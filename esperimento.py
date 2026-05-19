from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pinn import PINN


@dataclass(frozen=True)
class ConfigurazioneEsperimento:
    data_root: str = "data"
    equation: str = "advection"
    structure: str = "cloud"
    resolution: str = "high"
    t0: float = 0.0
    dt: float = 1.0
    velocita_x: float = 1.0
    velocita_y: float = 1.0
    hidden_dim: int = 64
    num_layers: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    numero_valutazioni_training: int = 5000
    batch_collocation: int = 16384
    batch_dmd: int = 16
    batch_periodic: int = 4096
    validation_points: int = 200
    lambda_initial_boundary: float = 1.0
    lambda_fisica: float = 1.0
    lambda_dmd: float = 0.0
    dmd_operator_path: str = "results/per_sim_final"
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0
    seed: int = 42
    device: str = "auto"
    download: bool = False


@dataclass
class SimulazioneDynabench:
    valori: torch.Tensor
    pos: torch.Tensor
    tempi: torch.Tensor
    tipo_punti: str
    descrizione_punti: str

    @property
    def numero_tempi(self) -> int:
        return int(self.valori.shape[0])

    @property
    def numero_punti(self) -> int:
        return int(self.valori.shape[1])


@dataclass
class OperatoreDMD:
    matrix: torch.Tensor
    norm_min: torch.Tensor
    norm_max: torch.Tensor


class DatasetSimulazioniIntere:
    """Carica simulazioni intere Dynabench invece delle vecchie finestre temporali."""

    def __init__(self, configurazione: ConfigurazioneEsperimento):
        try:
            from koopman_pidmd import DynabenchSimulationIterator
        except ImportError as exc:
            raise RuntimeError(
                "dynabench non e' installato. Installa le dipendenze con "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        self.configurazione = configurazione
        download_dataset = configurazione.download or not self._file_train_presenti()
        if download_dataset and not configurazione.download:
            print(
                "Dataset Dynabench train high resolution non trovato: "
                "provo a scaricarlo."
            )
        self.iterator = DynabenchSimulationIterator(
            split="train",
            equation=configurazione.equation,
            structure=configurazione.structure,
            resolution=configurazione.resolution,
            base_path=configurazione.data_root,
            download=download_dataset,
            dtype=np.float32,
        )
        if len(self.iterator) == 0:
            raise RuntimeError("Nessuna simulazione Dynabench trovata nello split train.")

    def _file_train_presenti(self) -> bool:
        path = (
            Path(self.configurazione.data_root)
            / self.configurazione.equation
            / self.configurazione.structure
            / self.configurazione.resolution
        )
        return any(path.glob("*train*.h5"))

    def carica_prima_simulazione(self, device: torch.device) -> SimulazioneDynabench:
        sample = self.iterator[0]
        valori_np = np.asarray(sample.x, dtype=np.float32)
        pos_np = np.asarray(sample.pos, dtype=np.float32)
        if valori_np.ndim != 3 or valori_np.shape[-1] != 1:
            raise ValueError("Mi aspetto sample.x con shape (T, K, 1).")
        if pos_np.ndim != 2 or pos_np.shape[-1] != 2:
            raise ValueError("Mi aspetto sample.pos con shape (K, 2).")

        tempi_np = self._tempi_da_sample(sample, valori_np.shape[0])
        tipo_punti, descrizione_punti = self._descrivi_punti(pos_np)
        return SimulazioneDynabench(
            valori=torch.as_tensor(valori_np[:, :, 0], device=device),
            pos=torch.as_tensor(pos_np, device=device),
            tempi=torch.as_tensor(tempi_np, device=device),
            tipo_punti=tipo_punti,
            descrizione_punti=descrizione_punti,
        )

    def _tempi_da_sample(self, sample: Any, numero_tempi: int) -> np.ndarray:
        for nome in ("t", "time", "times", "tempi"):
            if hasattr(sample, nome):
                valore = np.asarray(getattr(sample, nome), dtype=np.float32).reshape(-1)
                if valore.shape[0] == numero_tempi:
                    return valore
        return (
            self.configurazione.t0
            + self.configurazione.dt * np.arange(numero_tempi, dtype=np.float32)
        )

    def _descrivi_punti(self, pos: np.ndarray) -> tuple[str, str]:
        pos = np.asarray(pos, dtype=np.float32)
        coordinate = np.round(pos, decimals=8)
        x_unici = np.unique(coordinate[:, 0])
        y_unici = np.unique(coordinate[:, 1])
        punti = {tuple(map(float, punto)) for punto in coordinate}
        griglia_completa = (
            len(x_unici) * len(y_unici) == coordinate.shape[0]
            and all((float(x), float(y)) in punti for x in x_unici for y in y_unici)
        )
        if griglia_completa:
            return (
                "griglia_tensoriale",
                f"{len(x_unici)}x{len(y_unici)} punti su prodotto cartesiano",
            )

        min_x, max_x = float(pos[:, 0].min()), float(pos[:, 0].max())
        min_y, max_y = float(pos[:, 1].min()), float(pos[:, 1].max())
        return (
            "cloud",
            (
                f"{pos.shape[0]} punti cloud non tensoriali, "
                f"x=[{min_x:.4f}, {max_x:.4f}], y=[{min_y:.4f}, {max_y:.4f}]"
            ),
        )


def risolvi_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def normalizza_tempo(tempi: torch.Tensor, _configurazione: ConfigurazioneEsperimento) -> torch.Tensor:
    t_iniziale = tempi[0]
    t_finale = tempi[-1]
    return (tempi - t_iniziale) / torch.clamp(t_finale - t_iniziale, min=1e-8)


def features_da_punti(
    pos: torch.Tensor,
    tempi_normalizzati: torch.Tensor,
    indici_punti: torch.Tensor,
    indici_tempi: torch.Tensor,
) -> torch.Tensor:
    punti = pos.index_select(0, indici_punti)
    tempi = tempi_normalizzati.index_select(0, indici_tempi).unsqueeze(1)
    return torch.cat([punti, tempi], dim=1)


def condizione_iniziale_completa(
    simulazione: SimulazioneDynabench,
    tempi_normalizzati: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = simulazione.valori.device
    indici_punti = torch.arange(simulazione.numero_punti, device=device)
    indici_tempi = torch.zeros(simulazione.numero_punti, device=device, dtype=torch.long)
    features = features_da_punti(
        simulazione.pos,
        tempi_normalizzati,
        indici_punti,
        indici_tempi,
    )
    valori = simulazione.valori[0].unsqueeze(1)
    return features, valori


def loss_fisica_avvezione(
    modello: PINN,
    features: torch.Tensor,
    tempo_scala: float,
    configurazione: ConfigurazioneEsperimento,
) -> torch.Tensor:
    features = features.detach().clone().requires_grad_(True)
    pred = modello(features)
    gradiente = torch.autograd.grad(
        pred,
        features,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
    )[0]
    du_dx = gradiente[:, 0:1]
    du_dy = gradiente[:, 1:2]
    du_dt_feature = gradiente[:, 2:3]
    du_dt = du_dt_feature / max(tempo_scala, 1e-8)
    residuo = du_dt + configurazione.velocita_x * du_dx + configurazione.velocita_y * du_dy
    return torch.mean(residuo**2)


def punti_periodic_boundary(
    tempi_norm: torch.Tensor,
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = tempi_norm.device
    dtype = tempi_norm.dtype
    indici_tempi = torch.as_tensor(
        rng.integers(0, int(tempi_norm.numel()), size=batch_size),
        device=device,
        dtype=torch.long,
    )
    tempi = tempi_norm.index_select(0, indici_tempi).unsqueeze(1)
    coordinate = torch.as_tensor(
        rng.random(batch_size),
        device=device,
        dtype=dtype,
    ).unsqueeze(1)

    bordo_x0 = torch.cat([torch.zeros_like(coordinate), coordinate, tempi], dim=1)
    bordo_x1 = torch.cat([torch.ones_like(coordinate), coordinate, tempi], dim=1)
    bordo_y0 = torch.cat([coordinate, torch.zeros_like(coordinate), tempi], dim=1)
    bordo_y1 = torch.cat([coordinate, torch.ones_like(coordinate), tempi], dim=1)
    return bordo_x0, bordo_x1, bordo_y0, bordo_y1


def carica_operatore_dmd(
    path: str | Path,
    simulazione: SimulazioneDynabench,
    device: torch.device,
) -> OperatoreDMD:
    path = Path(path)
    if path.is_dir():
        candidati = sorted(path.glob("*sim00000.npz"))
        if not candidati:
            raise FileNotFoundError(f"Nessun operatore DMD sim00000 trovato in {path}.")
        path = candidati[0]
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Per ora l'esperimento minimale legge operatori DMD .npz.")

    with np.load(path) as data:
        matrix_np = np.asarray(data["A"], dtype=np.float32)
        norm_min_np = np.asarray(data["norm_min"], dtype=np.float32).reshape(-1)[0]
        norm_max_np = np.asarray(data["norm_max"], dtype=np.float32).reshape(-1)[0]
        pts_np = np.asarray(data["pts"], dtype=np.float32) if "pts" in data else None

    if matrix_np.shape != (simulazione.numero_punti, simulazione.numero_punti):
        raise ValueError(
            "Operatore DMD incompatibile: "
            f"shape {matrix_np.shape}, attesa "
            f"({simulazione.numero_punti}, {simulazione.numero_punti})."
        )
    if pts_np is not None:
        pts = torch.as_tensor(pts_np, device=device, dtype=simulazione.pos.dtype)
        scarto = torch.max(torch.abs(pts - simulazione.pos))
        if float(scarto.detach().cpu()) > 1e-4:
            raise ValueError("I punti dell'operatore DMD non coincidono con Dynabench sim00000.")

    return OperatoreDMD(
        matrix=torch.as_tensor(matrix_np, device=device, dtype=simulazione.valori.dtype),
        norm_min=torch.as_tensor(norm_min_np, device=device, dtype=simulazione.valori.dtype),
        norm_max=torch.as_tensor(norm_max_np, device=device, dtype=simulazione.valori.dtype),
    )


def normalizza_dmd(stati: torch.Tensor, dmd: OperatoreDMD) -> torch.Tensor:
    return 2.0 * (stati - dmd.norm_min) / torch.clamp(dmd.norm_max - dmd.norm_min, min=1e-8) - 1.0


def predici_snapshot(
    modello: PINN,
    simulazione: SimulazioneDynabench,
    tempi_norm: torch.Tensor,
    indici_tempi: torch.Tensor,
) -> torch.Tensor:
    batch_tempi = int(indici_tempi.numel())
    device = simulazione.valori.device
    indici_punti = torch.arange(simulazione.numero_punti, device=device)
    indici_punti = indici_punti.repeat(batch_tempi)
    indici_tempi_estesi = indici_tempi.repeat_interleave(simulazione.numero_punti)
    features = features_da_punti(
        simulazione.pos,
        tempi_norm,
        indici_punti,
        indici_tempi_estesi,
    )
    return modello(features).reshape(batch_tempi, simulazione.numero_punti)


def loss_dmd(
    modello: PINN,
    simulazione: SimulazioneDynabench,
    tempi_norm: torch.Tensor,
    dmd: OperatoreDMD,
    indici_tempi: torch.Tensor,
) -> torch.Tensor:
    pred_corrente = predici_snapshot(modello, simulazione, tempi_norm, indici_tempi)
    pred_successivo = predici_snapshot(modello, simulazione, tempi_norm, indici_tempi + 1)
    pred_corrente_norm = normalizza_dmd(pred_corrente, dmd)
    pred_successivo_norm = normalizza_dmd(pred_successivo, dmd)
    successivo_dmd = pred_corrente_norm @ dmd.matrix.T
    return torch.mean((successivo_dmd - pred_successivo_norm) ** 2)


def crea_esperimento(configurazione: ConfigurazioneEsperimento | None = None) -> dict[str, Any]:
    if configurazione is None:
        configurazione = ConfigurazioneEsperimento()

    np.random.seed(configurazione.seed)
    torch.manual_seed(configurazione.seed)
    device = risolvi_device(configurazione.device)
    rng = np.random.default_rng(configurazione.seed)

    dataset = DatasetSimulazioniIntere(configurazione)
    simulazione = dataset.carica_prima_simulazione(device)
    dmd = (
        carica_operatore_dmd(configurazione.dmd_operator_path, simulazione, device)
        if configurazione.lambda_dmd != 0.0
        else None
    )
    modello = PINN(
        input_dim=3,
        output_dim=1,
        hidden_dim=configurazione.hidden_dim,
        num_layers=configurazione.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        modello.parameters(),
        lr=configurazione.learning_rate,
        weight_decay=configurazione.weight_decay,
    )

    return {
        "configurazione": configurazione,
        "device": device,
        "rng": rng,
        "dataset": dataset,
        "simulazione": simulazione,
        "dmd": dmd,
        "modello": modello,
        "optimizer": optimizer,
    }


def step_training(esperimento: dict[str, Any]) -> dict[str, float]:
    configurazione: ConfigurazioneEsperimento = esperimento["configurazione"]
    rng: np.random.Generator = esperimento["rng"]
    simulazione: SimulazioneDynabench = esperimento["simulazione"]
    dmd: OperatoreDMD | None = esperimento["dmd"]
    modello: PINN = esperimento["modello"]
    optimizer: torch.optim.Optimizer = esperimento["optimizer"]

    modello.train()
    tempi_norm = normalizza_tempo(simulazione.tempi, configurazione)
    tempo_scala = float((simulazione.tempi[-1] - simulazione.tempi[0]).detach().cpu())
    device = simulazione.valori.device

    x_ic, y_ic = condizione_iniziale_completa(simulazione, tempi_norm)
    bordo_x0, bordo_x1, bordo_y0, bordo_y1 = punti_periodic_boundary(
        tempi_norm,
        rng,
        configurazione.batch_periodic,
    )
    loss_initial_boundary, loss_ic, loss_periodic = modello.boundary_condition_loss(
        x_ic,
        y_ic,
        bordo_x0,
        bordo_x1,
        bordo_y0,
        bordo_y1,
    )

    indici_punti_fisica = torch.as_tensor(
        rng.integers(0, simulazione.numero_punti, size=configurazione.batch_collocation),
        device=device,
        dtype=torch.long,
    )
    indici_tempi_fisica = torch.as_tensor(
        rng.integers(0, simulazione.numero_tempi, size=configurazione.batch_collocation),
        device=device,
        dtype=torch.long,
    )
    x_fisica = features_da_punti(
        simulazione.pos,
        tempi_norm,
        indici_punti_fisica,
        indici_tempi_fisica,
    )
    loss_fisica = loss_fisica_avvezione(modello, x_fisica, tempo_scala, configurazione)

    if configurazione.lambda_dmd != 0.0:
        if dmd is None:
            raise RuntimeError("lambda_dmd e' attiva ma l'operatore DMD non e' stato caricato.")
        indici_tempi_dmd = torch.as_tensor(
            rng.integers(0, simulazione.numero_tempi - 1, size=configurazione.batch_dmd),
            device=device,
            dtype=torch.long,
        )
        loss_dmd_valore = loss_dmd(modello, simulazione, tempi_norm, dmd, indici_tempi_dmd)
    else:
        loss_dmd_valore = loss_ic.new_zeros(())

    loss_totale = (
        configurazione.lambda_initial_boundary * loss_initial_boundary
        + configurazione.lambda_fisica * loss_fisica
        + configurazione.lambda_dmd * loss_dmd_valore
    )

    optimizer.zero_grad(set_to_none=True)
    loss_totale.backward()
    optimizer.step()

    return {
        "loss_totale": float(loss_totale.detach().cpu()),
        "loss_initial_boundary": float(loss_initial_boundary.detach().cpu()),
        "loss_ic": float(loss_ic.detach().cpu()),
        "loss_periodic": float(loss_periodic.detach().cpu()),
        "loss_fisica": float(loss_fisica.detach().cpu()),
        "loss_dmd": float(loss_dmd_valore.detach().cpu()),
    }


@torch.no_grad()
def valuta_su_dynabench(esperimento: dict[str, Any]) -> dict[str, float]:
    configurazione: ConfigurazioneEsperimento = esperimento["configurazione"]
    rng: np.random.Generator = esperimento["rng"]
    simulazione: SimulazioneDynabench = esperimento["simulazione"]
    modello: PINN = esperimento["modello"]

    modello.eval()
    tempi_norm = normalizza_tempo(simulazione.tempi, configurazione)
    device = simulazione.valori.device
    indici_punti = torch.as_tensor(
        rng.integers(0, simulazione.numero_punti, size=configurazione.validation_points),
        device=device,
        dtype=torch.long,
    )
    indici_tempi = torch.as_tensor(
        rng.integers(0, simulazione.numero_tempi, size=configurazione.validation_points),
        device=device,
        dtype=torch.long,
    )
    features = features_da_punti(simulazione.pos, tempi_norm, indici_punti, indici_tempi)
    pred = modello(features)
    target = simulazione.valori[indici_tempi, indici_punti].unsqueeze(1)
    mse = torch.mean((pred - target) ** 2)
    return {
        "validation_mse_200": float(mse.detach().cpu()),
    }


def allena(
    esperimento: dict[str, Any],
    numero_valutazioni: int | None = None,
    stampa_ogni: int = 50,
    valida_ogni: int = 500,
) -> list[dict[str, float]]:
    configurazione: ConfigurazioneEsperimento = esperimento["configurazione"]
    if numero_valutazioni is None:
        numero_valutazioni = configurazione.numero_valutazioni_training

    storico: list[dict[str, float]] = []
    best_validation = float("inf")
    validation_senza_miglioramento = 0
    for valutazione in range(1, numero_valutazioni + 1):
        metriche = step_training(esperimento)
        metriche["valutazione"] = float(valutazione)
        if valida_ogni > 0 and valutazione % valida_ogni == 0:
            metriche.update(valuta_su_dynabench(esperimento))
            validation = metriche["validation_mse_200"]
            if validation < best_validation - configurazione.early_stopping_min_delta:
                best_validation = validation
                validation_senza_miglioramento = 0
            else:
                validation_senza_miglioramento += 1
            metriche["best_validation_mse_200"] = float(best_validation)
            metriche["early_stopping_wait"] = float(validation_senza_miglioramento)
        storico.append(metriche)

        if stampa_ogni > 0 and (valutazione == 1 or valutazione % stampa_ogni == 0):
            messaggio = (
                f"[{valutazione:05d}/{numero_valutazioni}] "
                f"loss={metriche['loss_totale']:.6e} "
                f"ic_bc={metriche['loss_initial_boundary']:.6e} "
                f"ic={metriche['loss_ic']:.6e} "
                f"periodic={metriche['loss_periodic']:.6e} "
                f"fisica={metriche['loss_fisica']:.6e} "
                f"dmd={metriche['loss_dmd']:.6e}"
            )
            if "validation_mse_200" in metriche:
                messaggio += (
                    f" validation_mse_200={metriche['validation_mse_200']:.6e}"
                    f" best={metriche['best_validation_mse_200']:.6e}"
                    f" wait={int(metriche['early_stopping_wait'])}"
                )
            print(messaggio)

        if (
            configurazione.early_stopping_patience > 0
            and "validation_mse_200" in metriche
            and validation_senza_miglioramento >= configurazione.early_stopping_patience
        ):
            print(
                "Early stopping: validation_mse_200 non migliora da "
                f"{configurazione.early_stopping_patience} validation."
            )
            break
    return storico


def salva_storico_csv(storico: list[dict[str, float]], path: str | Path) -> None:
    if not storico:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    campi: list[str] = []
    for riga in storico:
        for chiave in riga:
            if chiave not in campi:
                campi.append(chiave)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=campi)
        writer.writeheader()
        writer.writerows(storico)


def descrivi_esperimento(esperimento: dict[str, Any]) -> None:
    configurazione: ConfigurazioneEsperimento = esperimento["configurazione"]
    simulazione: SimulazioneDynabench = esperimento["simulazione"]
    device: torch.device = esperimento["device"]
    print("Esperimento PINN minimale su una simulazione Dynabench advection")
    print(f"Device: {device}")
    print(
        "Dataset: "
        f"{configurazione.equation}/{configurazione.structure}/{configurazione.resolution}"
    )
    print("Simulazione allenata: 0")
    print(
        "Shape simulazione: "
        f"T={simulazione.numero_tempi}, K={simulazione.numero_punti}"
    )
    print(f"Punti Dynabench: {simulazione.tipo_punti} ({simulazione.descrizione_punti})")
    print("Condizione iniziale: tutti i punti disponibili del primo snapshot")
    print(
        "Pesi loss: "
        f"initial_boundary={configurazione.lambda_initial_boundary}, "
        f"fisica={configurazione.lambda_fisica}, "
        f"dmd={configurazione.lambda_dmd}"
    )
    if configurazione.lambda_dmd != 0.0:
        print(f"Operatore DMD: {configurazione.dmd_operator_path}")
    print(f"Validation: {configurazione.validation_points} punti Dynabench campionati")
    print(f"Periodic boundary: {configurazione.batch_periodic} punti campionati")
    if configurazione.early_stopping_patience > 0:
        print(
            "Early stopping: "
            f"patience={configurazione.early_stopping_patience}, "
            f"min_delta={configurazione.early_stopping_min_delta}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PINN su simulazioni intere Dynabench advection.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--resolution", choices=["low", "medium", "high"], help="Risoluzione Dynabench da usare.")
    parser.add_argument("--evals", type=int, help="Numero di step di training.")
    parser.add_argument("--batch-collocation", type=int, help="Punti PDE campionati per step.")
    parser.add_argument("--batch-dmd", type=int, help="Transizioni temporali usate nella DMD loss.")
    parser.add_argument("--batch-periodic", type=int, help="Punti campionati sulle condizioni periodiche.")
    parser.add_argument("--validation-points", type=int, help="Punti Dynabench usati per la validation.")
    parser.add_argument("--dmd-operator", help="File .npz o directory con operatore DMD sim00000.")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0 oppure auto.")
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lambda-initial-boundary", type=float)
    parser.add_argument("--lambda-fisica", type=float)
    parser.add_argument("--lambda-dmd", type=float)
    parser.add_argument("--early-stopping-patience", type=int, help="Numero di validation senza miglioramento prima dello stop; 0 disattiva.")
    parser.add_argument("--early-stopping-min-delta", type=float, help="Miglioramento minimo richiesto sulla validation.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--history-csv", default="results/training_history.csv")
    parser.add_argument("--save-model", help="Percorso .pt in cui salvare pesi e configurazione.")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configurazione = ConfigurazioneEsperimento(download=args.download)
    sostituzioni = {
        "data_root": args.data_root,
        "resolution": args.resolution,
        "numero_valutazioni_training": args.evals,
        "batch_collocation": args.batch_collocation,
        "batch_dmd": args.batch_dmd,
        "batch_periodic": args.batch_periodic,
        "validation_points": args.validation_points,
        "dmd_operator_path": args.dmd_operator,
        "device": args.device,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lambda_initial_boundary": args.lambda_initial_boundary,
        "lambda_fisica": args.lambda_fisica,
        "lambda_dmd": args.lambda_dmd,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
    }
    configurazione = replace(
        configurazione,
        **{chiave: valore for chiave, valore in sostituzioni.items() if valore is not None},
    )

    esperimento = crea_esperimento(configurazione)
    descrivi_esperimento(esperimento)
    storico = allena(
        esperimento,
        numero_valutazioni=configurazione.numero_valutazioni_training,
        stampa_ogni=args.print_every,
        valida_ogni=args.validate_every,
    )
    if args.history_csv:
        salva_storico_csv(storico, args.history_csv)
        print(f"Storico salvato in {args.history_csv}")
    if args.save_model:
        path = Path(args.save_model)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": esperimento["modello"].state_dict(),
                "configurazione": configurazione.__dict__,
                "storico": storico,
            },
            path,
        )
        print(f"Modello salvato in {path}")


if __name__ == "__main__":
    main()
