from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


@dataclass(frozen=True)
class ConfigurazioneEsperimento:
    data_root: str = "data"
    equation: str = "advection"
    structure: str = "cloud"
    resolution: str = "low"
    split: str = "train"
    t0: float = 0.0
    t_finale: float = 200.0
    dt: float = 1.0
    velocita_x: float = 1.0
    velocita_y: float = 1.0
    hidden_dim: int = 64
    num_layers: int = 4
    dropout: float = 0.05
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    numero_valutazioni_training: int = 5000
    numero_punti_l2: int = 4096
    batch_traiettorie_data: int = 4
    batch_tempi_data: int = 4
    batch_traiettorie_fisica: int = 2
    batch_tempi_fisica: int = 3
    batch_intervalli_dmd: int = 4
    lambda_data: float = 0.0
    lambda_fisica: float = 0.0
    lambda_dmd: float = 1.0
    seed: int = 42
    device: str = "cpu"


def aggiungi_dropout_al_modello(
    modello: torch.nn.Module,
    probabilita: float,
) -> torch.nn.Module:
    if probabilita < 0.0 or probabilita >= 1.0:
        raise ValueError("dropout deve essere in [0, 1).")
    if probabilita == 0.0:
        return modello
    if not hasattr(modello, "net") or not isinstance(modello.net, torch.nn.Sequential):
        raise TypeError("Il modello deve esporre una rete torch.nn.Sequential in .net.")

    layers = []
    for layer in modello.net:
        layers.append(layer)
        if isinstance(layer, torch.nn.Tanh):
            layers.append(torch.nn.Dropout(p=probabilita))
    modello.net = torch.nn.Sequential(*layers)
    return modello


class DatasetDynabenchAvvezione:
    """
    Lettore lazy per Dynabench advection.

    I file locali hanno la struttura:
    data   -> (traiettorie, 201, 225, 1)
    points -> (traiettorie, 225, 2)

    I tempi data sono t_i = 0, 1, ..., 200. Gli intervalli DMD sono i
    200 intervalli centrati in t_i + 0.5.
    """

    def __init__(self, configurazione: ConfigurazioneEsperimento):
        self.configurazione = configurazione
        directory = (
            Path(configurazione.data_root)
            / configurazione.equation
            / configurazione.structure
            / configurazione.resolution
        )
        pattern = (
            f"{configurazione.equation}_{configurazione.split}_"
            f"{configurazione.structure}_{configurazione.resolution}_*.h5"
        )
        self.paths = sorted(directory.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(
                f"Nessun file Dynabench trovato con pattern {directory / pattern}"
            )

        self._handles: dict[int, h5py.File] = {}
        self.traiettorie_per_file: list[int] = []
        self._starts = [0]

        n_tempi = None
        n_punti = None
        n_variabili = None
        for path in self.paths:
            with h5py.File(path, "r") as file:
                data_shape = file["data"].shape
                points_shape = file["points"].shape

            if len(data_shape) != 4 or len(points_shape) != 3:
                raise ValueError(f"Struttura HDF5 non supportata in {path}.")
            if data_shape[0] != points_shape[0] or data_shape[2] != points_shape[1]:
                raise ValueError(f"Shape incompatibili tra data e points in {path}.")

            if n_tempi is None:
                n_tempi = data_shape[1]
                n_punti = data_shape[2]
                n_variabili = data_shape[3]
            elif (n_tempi, n_punti, n_variabili) != data_shape[1:]:
                raise ValueError(f"Shape data non omogenea in {path}.")

            self.traiettorie_per_file.append(data_shape[0])
            self._starts.append(self._starts[-1] + data_shape[0])

        self.numero_tempi = int(n_tempi)
        self.numero_punti_spaziali = int(n_punti)
        self.numero_variabili = int(n_variabili)
        self.numero_traiettorie = self._starts[-1]
        self.numero_rollback = self.numero_traiettorie * (self.numero_tempi - 1)
        self.tempi_data = (
            configurazione.t0
            + configurazione.dt * np.arange(self.numero_tempi, dtype=np.float32)
        )
        self.tempi_dmd = self.tempi_data[:-1] + 0.5 * configurazione.dt

        if not np.isclose(self.tempi_data[0], configurazione.t0):
            raise ValueError("Il primo tempo del dataset non coincide con t0.")
        if not np.isclose(self.tempi_data[-1], configurazione.t_finale):
            raise ValueError("Il dataset non copre l'intervallo temporale richiesto.")

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _handle(self, indice_file: int) -> h5py.File:
        if indice_file not in self._handles:
            self._handles[indice_file] = h5py.File(self.paths[indice_file], "r")
        return self._handles[indice_file]

    def _mappa_traiettoria(self, indice_globale: int) -> tuple[int, int]:
        indice_file = int(np.searchsorted(self._starts[1:], indice_globale, side="right"))
        indice_locale = indice_globale - self._starts[indice_file]
        return indice_file, int(indice_locale)

    def campiona_snapshot(
        self,
        rng: np.random.Generator,
        numero_traiettorie: int,
        numero_tempi: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Campiona punti data/fisica sui tempi interi t_i = 0, 1, ..., 200.

        Per ogni traiettoria e tempo campionati usa tutti i punti spaziali
        disponibili nel cloud Dynabench, cioe tutti i 225 punti della soluzione.
        """
        coords = []
        valori = []
        indici_traiettoria = rng.integers(
            0,
            self.numero_traiettorie,
            size=numero_traiettorie,
            endpoint=False,
        )

        for indice_globale in indici_traiettoria:
            indice_file, indice_locale = self._mappa_traiettoria(int(indice_globale))
            file = self._handle(indice_file)
            punti = np.asarray(file["points"][indice_locale], dtype=np.float32)
            indici_tempo = rng.integers(
                0,
                self.numero_tempi,
                size=numero_tempi,
                endpoint=False,
            )

            for indice_tempo in indici_tempo:
                tempo = self.tempi_data[indice_tempo]
                soluzione = np.asarray(
                    file["data"][indice_locale, indice_tempo, :, 0],
                    dtype=np.float32,
                )
                tempo_colonna = np.full(
                    (self.numero_punti_spaziali, 1),
                    tempo,
                    dtype=np.float32,
                )
                coords.append(np.concatenate([punti, tempo_colonna], axis=1))
                valori.append(soluzione.reshape(-1, 1))

        x = torch.as_tensor(np.vstack(coords), device=device, dtype=dtype)
        y = torch.as_tensor(np.vstack(valori), device=device, dtype=dtype)
        return x, y

    def campiona_punti_l2(
        self,
        rng: np.random.Generator,
        numero_punti: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        numero_punti = max(1001, numero_punti)
        numero_blocchi = int(np.ceil(numero_punti / self.numero_punti_spaziali))
        x, y = self.campiona_snapshot(
            rng=rng,
            numero_traiettorie=numero_blocchi,
            numero_tempi=1,
            device=device,
            dtype=dtype,
        )
        return x[:numero_punti], y[:numero_punti]

    def prima_traiettoria(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        file = self._handle(0)
        tempi = self.tempi_data.copy()
        snapshot = np.asarray(file["data"][0, :, :, 0], dtype=np.float32)
        punti = np.asarray(file["points"][0], dtype=np.float32)
        return tempi, snapshot, punti


def crea_griglia_spaziale_dmd(numero_punti: int) -> np.ndarray:
    lato = int(round(np.sqrt(numero_punti)))
    if lato * lato != numero_punti:
        raise ValueError(
            "Per costruire la griglia DMD serve un numero quadrato di punti."
        )

    coordinate = (np.arange(lato, dtype=np.float32) + 0.5) / lato
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="ij")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float32)


def coordinate_spazio_tempo(
    punti_spaziali: torch.Tensor,
    tempi: torch.Tensor,
) -> torch.Tensor:
    numero_tempi = tempi.numel()
    numero_punti = punti_spaziali.shape[0]
    punti = punti_spaziali.unsqueeze(0).expand(numero_tempi, numero_punti, 2)
    colonna_tempi = tempi.reshape(-1, 1, 1).expand(numero_tempi, numero_punti, 1)
    return torch.cat([punti, colonna_tempi], dim=-1).reshape(-1, 3)


def campiona_batch_dmd(
    dataset: DatasetDynabenchAvvezione,
    punti_spaziali_dmd: torch.Tensor,
    rng: np.random.Generator,
    configurazione: ConfigurazioneEsperimento,
) -> dict[str, torch.Tensor | int]:
    indici = rng.integers(
        0,
        dataset.numero_tempi - 1,
        size=configurazione.batch_intervalli_dmd,
        endpoint=False,
    )
    device = punti_spaziali_dmd.device
    tempi_iniziali = torch.as_tensor(
        dataset.tempi_data[indici],
        device=device,
        dtype=punti_spaziali_dmd.dtype,
    )
    tempi_finali = tempi_iniziali + configurazione.dt
    tempi_mezzi = tempi_iniziali + 0.5 * configurazione.dt

    return {
        "x_iniziali": coordinate_spazio_tempo(punti_spaziali_dmd, tempi_iniziali),
        "x_finali": coordinate_spazio_tempo(punti_spaziali_dmd, tempi_finali),
        "x_mezzi": coordinate_spazio_tempo(punti_spaziali_dmd, tempi_mezzi),
        "numero_intervalli": len(indici),
        "numero_punti": punti_spaziali_dmd.shape[0],
    }


def operatore_differenziale_avvezione(
    u: torch.Tensor,
    collocation_points: torch.Tensor,
    configurazione: ConfigurazioneEsperimento,
) -> torch.Tensor:
    gradiente = torch.autograd.grad(
        u,
        collocation_points,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    du_dx = gradiente[:, 0:1]
    du_dy = gradiente[:, 1:2]
    du_dt = gradiente[:, 2:3]

    return (
        du_dt
        + configurazione.velocita_x * du_dx
        + configurazione.velocita_y * du_dy
    )


def fisica_avvezione(
    termine_differenziale: torch.Tensor,
    u: torch.Tensor,
    collocation_points: torch.Tensor,
) -> torch.Tensor:
    return termine_differenziale


def loss_dmd_midpoint(
    modello: torch.nn.Module,
    dmd_operator: torch.Tensor,
    batch_dmd: dict[str, torch.Tensor | int],
    configurazione: ConfigurazioneEsperimento,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from pinn import loss_toeplitz

    numero_intervalli = int(batch_dmd["numero_intervalli"])
    numero_punti = int(batch_dmd["numero_punti"])

    u_iniziali = modello(batch_dmd["x_iniziali"]).reshape(numero_intervalli, numero_punti)
    u_finali = modello(batch_dmd["x_finali"]).reshape(numero_intervalli, numero_punti)
    u_mezzi = modello(batch_dmd["x_mezzi"]).reshape(numero_intervalli, numero_punti)

    derivata_temporale = (u_finali - u_iniziali) / configurazione.dt
    derivata_dmd = u_mezzi @ dmd_operator.T
    loss_dinamica = torch.mean((derivata_dmd - derivata_temporale) ** 2)
    loss_toeplitz_a = loss_toeplitz(dmd_operator)
    return loss_dinamica + loss_toeplitz_a, loss_dinamica, loss_toeplitz_a


def crea_esperimento(
    configurazione: ConfigurazioneEsperimento | None = None,
    create_model: bool = True,
) -> dict[str, object]:
    if configurazione is None:
        configurazione = ConfigurazioneEsperimento()

    np.random.seed(configurazione.seed)
    torch.manual_seed(configurazione.seed)

    device = torch.device(configurazione.device)
    dataset = DatasetDynabenchAvvezione(configurazione)
    punti_dmd_np = crea_griglia_spaziale_dmd(dataset.numero_punti_spaziali)
    punti_dmd = torch.as_tensor(punti_dmd_np, device=device, dtype=torch.float32)

    modello = None
    dmd_operator = None
    optimizer = None
    if create_model:
        from pinn import PINN

        modello = PINN(
            input_dim=3,
            output_dim=dataset.numero_variabili,
            hidden_dim=configurazione.hidden_dim,
            num_layers=configurazione.num_layers,
        ).to(device)
        modello = aggiungi_dropout_al_modello(
            modello,
            probabilita=configurazione.dropout,
        )
        dmd_operator = torch.nn.Parameter(
            torch.zeros(
                dataset.numero_punti_spaziali,
                dataset.numero_punti_spaziali,
                device=device,
            )
        )
        optimizer = torch.optim.AdamW(
            list(modello.parameters()) + [dmd_operator],
            lr=configurazione.learning_rate,
            weight_decay=configurazione.weight_decay,
        )

    def operatore(u: torch.Tensor, punti: torch.Tensor) -> torch.Tensor:
        return operatore_differenziale_avvezione(u, punti, configurazione)

    return {
        "configurazione": configurazione,
        "dataset": dataset,
        "modello": modello,
        "dmd_operator": dmd_operator,
        "optimizer": optimizer,
        "rng": np.random.default_rng(configurazione.seed),
        "operatore_differenziale": operatore,
        "fisica": fisica_avvezione,
        "punti_spaziali_dmd": punti_dmd,
        "tempi_data": dataset.tempi_data,
        "tempi_dmd": dataset.tempi_dmd,
    }


def step_training(esperimento: dict[str, object]) -> dict[str, float]:
    configurazione = esperimento["configurazione"]
    dataset = esperimento["dataset"]
    modello = esperimento["modello"]
    dmd_operator = esperimento["dmd_operator"]
    optimizer = esperimento["optimizer"]
    rng = esperimento["rng"]
    device = torch.device(configurazione.device)

    if modello is None or dmd_operator is None or optimizer is None:
        raise ValueError("create_model=True e' necessario per il training.")

    modello.train()
    optimizer.zero_grad()

    x_data, y_data = dataset.campiona_snapshot(
        rng=rng,
        numero_traiettorie=configurazione.batch_traiettorie_data,
        numero_tempi=configurazione.batch_tempi_data,
        device=device,
    )
    loss_data = modello.data_loss(x_data, y_data)

    loss_fisica = loss_data.new_zeros(())
    if configurazione.lambda_fisica != 0.0:
        x_fisica, _ = dataset.campiona_snapshot(
            rng=rng,
            numero_traiettorie=configurazione.batch_traiettorie_fisica,
            numero_tempi=configurazione.batch_tempi_fisica,
            device=device,
        )
        loss_fisica = modello.physics_loss(
            x_fisica,
            esperimento["operatore_differenziale"],
            esperimento["fisica"],
        )

    loss_dmd = loss_data.new_zeros(())
    loss_dmd_dinamica = loss_data.new_zeros(())
    loss_toeplitz_a = loss_data.new_zeros(())
    if configurazione.lambda_dmd != 0.0:
        batch_dmd = campiona_batch_dmd(
            dataset=dataset,
            punti_spaziali_dmd=esperimento["punti_spaziali_dmd"],
            rng=rng,
            configurazione=configurazione,
        )
        loss_dmd, loss_dmd_dinamica, loss_toeplitz_a = loss_dmd_midpoint(
            modello,
            dmd_operator,
            batch_dmd,
            configurazione,
        )

    loss_totale = (
        configurazione.lambda_data * loss_data
        + configurazione.lambda_fisica * loss_fisica
        + configurazione.lambda_dmd * loss_dmd
    )
    loss_totale.backward()
    optimizer.step()

    return {
        "loss_totale": float(loss_totale.detach().cpu()),
        "loss_data": float(loss_data.detach().cpu()),
        "loss_fisica": float(loss_fisica.detach().cpu()),
        "loss_dmd": float(loss_dmd.detach().cpu()),
        "loss_dmd_dinamica": float(loss_dmd_dinamica.detach().cpu()),
        "loss_toeplitz": float(loss_toeplitz_a.detach().cpu()),
    }


@torch.no_grad()
def calcola_norma_l2(
    esperimento: dict[str, object],
    numero_punti: int | None = None,
) -> dict[str, float]:
    configurazione = esperimento["configurazione"]
    dataset = esperimento["dataset"]
    modello = esperimento["modello"]
    rng = esperimento["rng"]
    device = torch.device(configurazione.device)

    if modello is None:
        raise ValueError("create_model=True e' necessario per calcolare l'errore.")

    modello.eval()
    numero_punti = configurazione.numero_punti_l2 if numero_punti is None else numero_punti
    x_l2, y_l2 = dataset.campiona_punti_l2(
        rng=rng,
        numero_punti=numero_punti,
        device=device,
    )
    y_pred = modello(x_l2)
    errore_medio_quadratico = torch.mean((y_pred - y_l2) ** 2)
    energia_media = torch.mean(y_l2 ** 2)
    misura_spazio_tempo = configurazione.t_finale - configurazione.t0
    norma_l2 = torch.sqrt(misura_spazio_tempo * errore_medio_quadratico)
    norma_l2_relativa = torch.sqrt(
        errore_medio_quadratico / torch.clamp(energia_media, min=1e-12)
    )

    return {
        "norma_l2": float(norma_l2.cpu()),
        "norma_l2_relativa": float(norma_l2_relativa.cpu()),
        "punti_quadratura": int(max(1001, numero_punti)),
    }


def allena(
    esperimento: dict[str, object],
    numero_valutazioni: int | None = None,
    stampa_ogni: int = 100,
    valuta_l2_ogni: int = 500,
) -> list[dict[str, float]]:
    configurazione = esperimento["configurazione"]
    if numero_valutazioni is None:
        numero_valutazioni = configurazione.numero_valutazioni_training

    storico = []
    for valutazione in range(1, numero_valutazioni + 1):
        metriche = step_training(esperimento)
        metriche["valutazione"] = float(valutazione)

        if valutazione == 1 or valutazione % valuta_l2_ogni == 0:
            metriche.update(calcola_norma_l2(esperimento))

        storico.append(metriche)

        if stampa_ogni > 0 and (valutazione == 1 or valutazione % stampa_ogni == 0):
            messaggio = (
                f"[{valutazione:05d}/{numero_valutazioni}] "
                f"loss={metriche['loss_totale']:.6e} "
                f"data={metriche['loss_data']:.6e} "
                f"fisica={metriche['loss_fisica']:.6e} "
                f"dmd={metriche['loss_dmd']:.6e}"
            )
            if "norma_l2" in metriche:
                messaggio += (
                    f" l2={metriche['norma_l2']:.6e} "
                    f"l2_rel={metriche['norma_l2_relativa']:.6e}"
                )
            print(messaggio)

    return storico


def descrivi_esperimento(esperimento: dict[str, object]) -> None:
    configurazione = esperimento["configurazione"]
    dataset = esperimento["dataset"]
    print("Esperimento PINN su Dynabench advection")
    print("Dominio spaziale: [0, 1] x [0, 1]")
    print(f"Intervallo temporale: [{configurazione.t0}, {configurazione.t_finale}]")
    print(f"File split {configurazione.split}: {len(dataset.paths)}")
    print(f"Traiettorie totali: {dataset.numero_traiettorie}")
    print(f"Snapshot per traiettoria: {dataset.numero_tempi}")
    print(f"Rollback/intervalli consecutivi disponibili: {dataset.numero_rollback}")
    print(f"Punti spaziali per snapshot: {dataset.numero_punti_spaziali}")
    print(f"Tempi data: {dataset.tempi_data[0]} ... {dataset.tempi_data[-1]}")
    print(f"Tempi DMD: {dataset.tempi_dmd[0]} ... {dataset.tempi_dmd[-1]}")
    print(
        "Pesi loss: "
        f"data={configurazione.lambda_data}, "
        f"fisica={configurazione.lambda_fisica}, "
        f"dmd={configurazione.lambda_dmd}"
    )
    print(
        "Regolarizzazione: "
        f"dropout={configurazione.dropout}, "
        f"weight_decay={configurazione.weight_decay}"
    )
    print(f"Valutazioni training: {configurazione.numero_valutazioni_training}")
    print(f"Punti quadratura L2: {max(1001, configurazione.numero_punti_l2)}")


if __name__ == "__main__":
    esperimento = crea_esperimento()
    try:
        descrivi_esperimento(esperimento)
        allena(esperimento)
    finally:
        esperimento["dataset"].close()
