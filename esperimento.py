from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dynabench.dataset import DynabenchIterator
from dynabench.grid import Grid

from pinn import PINN


@dataclass(frozen=True)
class ConfigurazioneEsperimento:
    data_root: str = "data"
    equation: str = "advection"
    structure: str = "cloud"
    resolution: str = "low"
    lookback: int = 1
    rollout: int = 16
    t0: float = 0.0
    t_finale: float = 200.0
    dt: float = 1.0
    velocita_x: float = 1.0
    velocita_y: float = 1.0
    hidden_dim: int = 64
    num_layers: int = 4
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    numero_valutazioni_training: int = 5000
    batch_size: int = 4
    batch_size_dmd: int = 16
    dmd_punti_per_snapshot: int | None = 100
    batch_validation: int = 8
    validation_batches: int = 8
    lambda_data: float = 1.0
    lambda_fisica: float = 0.0
    lambda_dmd: float = 0.0
    seed: int = 42
    device: str = "auto"
    dmd_operator_path: str | None = "results/per_sim_final"
    download: bool = False
    clip_grad_norm: float | None = 1.0
    dmd_cache_size: int = 10


@dataclass
class BatchRollout:
    u0: torch.Tensor
    target: torch.Tensor
    pos: torch.Tensor
    tempi_rollout: torch.Tensor
    indici_iterator: np.ndarray
    indici_simulazione: np.ndarray
    indici_temporali: np.ndarray


@dataclass
class DMDRecord:
    matrix: np.ndarray
    norm_min: float | None = None
    norm_max: float | None = None
    pts: np.ndarray | None = None


@dataclass
class DMDTensors:
    matrix: torch.Tensor
    norm_min: torch.Tensor | None = None
    norm_max: torch.Tensor | None = None
    pts: torch.Tensor | None = None


def risolvi_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def normalizza_tempo(tempi: torch.Tensor, configurazione: ConfigurazioneEsperimento) -> torch.Tensor:
    scala = configurazione.t_finale - configurazione.t0
    if scala <= 0:
        raise ValueError("t_finale deve essere maggiore di t0.")
    return (tempi - configurazione.t0) / scala


class DynabenchRolloutDataset:
    """Wrapper leggero attorno a DynabenchIterator per finestre (1, 16)."""

    def __init__(self, configurazione: ConfigurazioneEsperimento, split: str):
        self.configurazione = configurazione
        self.split = split
        self.iterator = DynabenchIterator(
            split=split,
            equation=configurazione.equation,
            structure=configurazione.structure,
            resolution=configurazione.resolution,
            base_path=configurazione.data_root,
            lookback=configurazione.lookback,
            rollout=configurazione.rollout,
            squeeze_lookback_dim=False,
            download=configurazione.download,
            dtype=np.float32,
        )
        if len(self.iterator) == 0:
            raise RuntimeError(f"DynabenchIterator vuoto per split={split!r}.")

        sample = self.iterator[0]
        if sample.x.ndim != 3 or sample.y.ndim != 3 or sample.pos.ndim != 2:
            raise ValueError(
                "Questo esperimento si aspetta cloud data con x=(L,K,F), "
                "y=(R,K,F), pos=(K,2)."
            )
        if sample.x.shape[0] != configurazione.lookback:
            raise ValueError("lookback del sample non coerente con la configurazione.")
        if sample.y.shape[0] != configurazione.rollout:
            raise ValueError("rollout del sample non coerente con la configurazione.")
        if sample.x.shape[-1] != 1 or sample.y.shape[-1] != 1:
            raise ValueError("L'esperimento advection qui usa una sola variabile scalare.")

        self.numero_punti = int(sample.pos.shape[0])
        self.numero_variabili = int(sample.x.shape[-1])
        lato = int(round(math.sqrt(self.numero_punti)))
        if lato * lato != self.numero_punti:
            raise ValueError("Per low/cloud ci si aspetta un numero quadrato di punti.")

        self.griglia_dominio = Grid(
            grid_size=(lato, lato),
            grid_limits=((0.0, 1.0), (0.0, 1.0)),
        )
        self.offset_simulazioni = np.cumsum(self.iterator.number_of_simulations) - self.iterator.number_of_simulations[0]
        self._controlla_punti_nel_dominio(sample.pos)

    def __len__(self) -> int:
        return len(self.iterator)

    def _controlla_punti_nel_dominio(self, pos: np.ndarray) -> None:
        x_lim, y_lim = self.griglia_dominio.grid_limits
        eps = 1e-6
        if (
            np.min(pos[:, 0]) < x_lim[0] - eps
            or np.max(pos[:, 0]) > x_lim[1] + eps
            or np.min(pos[:, 1]) < y_lim[0] - eps
            or np.max(pos[:, 1]) > y_lim[1] + eps
        ):
            raise ValueError("I punti cloud Dynabench non cadono nel dominio [0,1]x[0,1].")

    def punti_dmd_da_griglia(self) -> np.ndarray:
        xx, yy = self.griglia_dominio.get_meshgrid()
        return np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float32)

    def decodifica_indice(self, index: int) -> tuple[int, int, int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index fuori range.")

        starting = np.asarray(self.iterator.starting_indices)
        indice_file = int(np.searchsorted(starting, index, side="right") - 1)
        raw_idx = int(index - starting[indice_file])
        usable_len = int(self.iterator.usable_simulation_lengths[indice_file])
        indice_sim_locale = raw_idx // usable_len
        indice_temporale = raw_idx % usable_len
        indice_sim_globale = int(self.offset_simulazioni[indice_file] + indice_sim_locale)
        return indice_file, int(indice_sim_locale), int(indice_temporale), indice_sim_globale

    def sample_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> BatchRollout:
        indici = rng.integers(0, len(self), size=batch_size, endpoint=False)

        u0_list: list[np.ndarray] = []
        target_list: list[np.ndarray] = []
        pos_list: list[np.ndarray] = []
        tempi_list: list[np.ndarray] = []
        sim_indices: list[int] = []
        time_indices: list[int] = []

        for indice in indici:
            _, _, indice_temporale, indice_sim_globale = self.decodifica_indice(int(indice))
            item = self.iterator[int(indice)]
            self._controlla_punti_nel_dominio(item.pos)

            u0_list.append(np.asarray(item.x[:, :, 0].T, dtype=np.float32))
            target_list.append(np.asarray(item.y[:, :, 0], dtype=np.float32))
            pos_list.append(np.asarray(item.pos, dtype=np.float32))

            tempi = (
                self.configurazione.t0
                + self.configurazione.dt
                * (
                    indice_temporale
                    + self.configurazione.lookback
                    + np.arange(self.configurazione.rollout, dtype=np.float32)
                )
            )
            tempi_list.append(tempi.astype(np.float32))
            sim_indices.append(indice_sim_globale)
            time_indices.append(indice_temporale)

        return BatchRollout(
            u0=torch.as_tensor(np.stack(u0_list), device=device, dtype=dtype),
            target=torch.as_tensor(np.stack(target_list), device=device, dtype=dtype),
            pos=torch.as_tensor(np.stack(pos_list), device=device, dtype=dtype),
            tempi_rollout=torch.as_tensor(np.stack(tempi_list), device=device, dtype=dtype),
            indici_iterator=np.asarray(indici, dtype=np.int64),
            indici_simulazione=np.asarray(sim_indices, dtype=np.int64),
            indici_temporali=np.asarray(time_indices, dtype=np.int64),
        )


class DMDOperatorProvider:
    def __init__(self, source: Any = None, cache_size: int = 32):
        self.source = source
        self.cache_size = cache_size
        self._cache: OrderedDict[int, DMDRecord] = OrderedDict()
        self._fixed_record: DMDRecord | None = None
        self._directory: Path | None = None
        self._missing_path: Path | None = None

        if source is None:
            return
        if isinstance(source, (np.ndarray, torch.Tensor)):
            matrix = source.detach().cpu().numpy() if isinstance(source, torch.Tensor) else source
            self._fixed_record = DMDRecord(matrix=self._as_float_array(matrix))
            return

        path = Path(source)
        if not path.exists():
            self._missing_path = path
            return
        if path.is_dir():
            if not any(path.glob("*.npz")):
                self._missing_path = path
                return
            self._directory = path
            return
        self._fixed_record = self._load_record_from_file(path)

    @property
    def has_operator(self) -> bool:
        return self._fixed_record is not None or self._directory is not None

    @property
    def missing_path(self) -> Path | None:
        return self._missing_path

    @property
    def directory(self) -> Path | None:
        return self._directory

    def operator_count(self) -> int:
        if self._directory is None:
            return 1 if self._fixed_record is not None else 0
        return len(list(self._directory.glob("*sim*.npz")))

    def get_for_batch(
        self,
        batch: BatchRollout,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DMDTensors:
        if self._fixed_record is not None:
            return self._record_to_tensors(self._fixed_record, device, dtype)
        if self._directory is None:
            raise ValueError(
                "lambda_dmd e' diverso da zero, ma nessun dmd_operator e' stato fornito."
            )

        records = [self._load_record_for_simulation(int(i)) for i in batch.indici_simulazione]
        matrix = torch.as_tensor(
            np.stack([record.matrix for record in records]),
            device=device,
            dtype=dtype,
        )

        norm_min = self._stack_optional_scalars(records, "norm_min", device, dtype)
        norm_max = self._stack_optional_scalars(records, "norm_max", device, dtype)
        pts = None
        if all(record.pts is not None for record in records):
            pts = torch.as_tensor(
                np.stack([record.pts for record in records if record.pts is not None]),
                device=device,
                dtype=dtype,
            )
        return DMDTensors(matrix=matrix, norm_min=norm_min, norm_max=norm_max, pts=pts)

    def _stack_optional_scalars(
        self,
        records: list[DMDRecord],
        field: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        values = [getattr(record, field) for record in records]
        if any(value is None for value in values):
            return None
        return torch.as_tensor(values, device=device, dtype=dtype)

    def _load_record_for_simulation(self, indice_simulazione: int) -> DMDRecord:
        if indice_simulazione in self._cache:
            record = self._cache.pop(indice_simulazione)
            self._cache[indice_simulazione] = record
            return record

        if self._directory is None:
            raise RuntimeError("Directory DMD non configurata.")

        candidates = sorted(self._directory.glob(f"*sim{indice_simulazione:05d}.npz"))
        if not candidates:
            raise FileNotFoundError(
                f"Nessun operatore DMD trovato per sim {indice_simulazione:05d} "
                f"in {self._directory}"
            )

        record = self._load_record_from_file(candidates[0])
        self._cache[indice_simulazione] = record
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return record

    def _record_to_tensors(
        self,
        record: DMDRecord,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DMDTensors:
        matrix = torch.as_tensor(record.matrix, device=device, dtype=dtype)
        norm_min = (
            torch.as_tensor(record.norm_min, device=device, dtype=dtype)
            if record.norm_min is not None
            else None
        )
        norm_max = (
            torch.as_tensor(record.norm_max, device=device, dtype=dtype)
            if record.norm_max is not None
            else None
        )
        pts = (
            torch.as_tensor(record.pts, device=device, dtype=dtype)
            if record.pts is not None
            else None
        )
        return DMDTensors(matrix=matrix, norm_min=norm_min, norm_max=norm_max, pts=pts)

    def _load_record_from_file(self, path: Path) -> DMDRecord:
        if not path.exists():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()
        if suffix == ".npz":
            with np.load(path) as data:
                matrix = self._first_present(data, ["A", "dmd_operator", "dmd_operatore", "operator", "matrix", "arr_0"])
                norm_min = self._optional_scalar(data, "norm_min")
                norm_max = self._optional_scalar(data, "norm_max")
                pts = self._as_float_array(data["pts"]) if "pts" in data else None
            return DMDRecord(
                matrix=self._as_float_array(matrix),
                norm_min=norm_min,
                norm_max=norm_max,
                pts=pts,
            )
        if suffix == ".npy":
            return DMDRecord(matrix=self._as_float_array(np.load(path)))
        if suffix in {".csv", ".txt"}:
            return DMDRecord(matrix=self._as_float_array(np.loadtxt(path, delimiter=",")))
        if suffix in {".pt", ".pth"}:
            loaded = torch.load(path, map_location="cpu")
            if isinstance(loaded, torch.Tensor):
                return DMDRecord(matrix=self._as_float_array(loaded))
            if isinstance(loaded, dict):
                matrix = self._first_present(loaded, ["A", "dmd_operator", "dmd_operatore", "operator", "matrix"])
                norm_min = self._optional_scalar(loaded, "norm_min")
                norm_max = self._optional_scalar(loaded, "norm_max")
                pts = loaded.get("pts")
                pts_np = self._as_float_array(pts) if pts is not None else None
                return DMDRecord(
                    matrix=self._as_float_array(matrix),
                    norm_min=norm_min,
                    norm_max=norm_max,
                    pts=pts_np,
                )
        raise ValueError(f"Formato operatore DMD non supportato: {path}")

    def _as_float_array(self, value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if np.iscomplexobj(array):
            array = array.real
        return array.astype(np.float32)

    def _first_present(self, container: Any, keys: list[str]) -> Any:
        for key in keys:
            if key in container:
                return container[key]
        raise KeyError(f"Nessuna delle chiavi {keys} e' presente nell'operatore DMD.")

    def _optional_scalar(self, container: Any, key: str) -> float | None:
        if key not in container:
            return None
        value = container[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return float(np.asarray(value).reshape(-1)[0])


def costruisci_feature_rollout(
    batch: BatchRollout,
    configurazione: ConfigurazioneEsperimento,
    require_grad: bool = False,
) -> torch.Tensor:
    batch_size, rollout, numero_punti = batch.target.shape
    pos = batch.pos[:, None, :, :].expand(batch_size, rollout, numero_punti, 2)
    tempo = normalizza_tempo(batch.tempi_rollout, configurazione)
    tempo = tempo[:, :, None, None].expand(batch_size, rollout, numero_punti, 1)
    lookback = batch.u0.shape[-1]
    u0 = batch.u0[:, None, :, :].expand(batch_size, rollout, numero_punti, lookback)
    features = torch.cat([pos, tempo, u0], dim=-1).reshape(-1, 3 + lookback)
    if require_grad:
        features = features.detach().clone().requires_grad_(True)
    return features


def forward_rollout(
    modello: torch.nn.Module,
    batch: BatchRollout,
    configurazione: ConfigurazioneEsperimento,
    require_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = costruisci_feature_rollout(batch, configurazione, require_grad=require_grad)
    pred_flat = modello(features)
    pred = pred_flat.reshape(batch.target.shape)
    return pred, features


def loss_fisica_avvezione(
    pred_flat: torch.Tensor,
    features: torch.Tensor,
    configurazione: ConfigurazioneEsperimento,
) -> torch.Tensor:
    gradiente = torch.autograd.grad(
        pred_flat,
        features,
        grad_outputs=torch.ones_like(pred_flat),
        create_graph=True,
        retain_graph=True,
    )[0]

    du_dx = gradiente[:, 0:1]
    du_dy = gradiente[:, 1:2]
    du_dt_feature = gradiente[:, 2:3]
    du_dt = du_dt_feature / (configurazione.t_finale - configurazione.t0)
    residuo = du_dt + configurazione.velocita_x * du_dx + configurazione.velocita_y * du_dy
    return torch.mean(residuo ** 2)


def _normalizza_stati_dmd(stati: torch.Tensor, dmd: DMDTensors) -> torch.Tensor:
    if dmd.norm_min is None or dmd.norm_max is None:
        return stati

    norm_min = dmd.norm_min
    norm_max = dmd.norm_max
    if norm_min.ndim == 0:
        return 2.0 * (stati - norm_min) / torch.clamp(norm_max - norm_min, min=1e-8) - 1.0

    view_shape = (norm_min.shape[0],) + (1,) * (stati.ndim - 1)
    norm_min = norm_min.reshape(view_shape)
    norm_max = norm_max.reshape(view_shape)
    return 2.0 * (stati - norm_min) / torch.clamp(norm_max - norm_min, min=1e-8) - 1.0


def loss_dmd_rollout(
    pred_rollout: torch.Tensor,
    batch: BatchRollout,
    dmd: DMDTensors,
    punti_per_snapshot: int | None = None,
) -> torch.Tensor:
    numero_punti = batch.u0.shape[-2]
    if dmd.matrix.shape[-2:] != (numero_punti, numero_punti):
        raise ValueError(
            f"Operatore DMD con shape {tuple(dmd.matrix.shape)} incompatibile "
            f"con {numero_punti} punti."
        )

    if dmd.pts is not None:
        if dmd.pts.ndim == 2:
            scarto_punti = torch.max(torch.abs(batch.pos - dmd.pts[None, :, :]))
        elif dmd.pts.ndim == 3:
            scarto_punti = torch.max(torch.abs(batch.pos - dmd.pts))
        else:
            raise ValueError("dmd.pts deve avere shape (K,2) oppure (B,K,2).")
        if float(scarto_punti.detach().cpu()) > 1e-4:
            raise ValueError(
                "I punti cloud del batch non coincidono con quelli dell'operatore DMD. "
                "Usa gli operatori per-simulazione generati da Copia 05 nello stesso ordine."
            )

    stati_correnti = torch.cat([batch.u0[:, None, :, -1], pred_rollout[:, :-1, :]], dim=1)
    stati_successivi = pred_rollout
    stati_correnti = _normalizza_stati_dmd(stati_correnti, dmd)
    stati_successivi = _normalizza_stati_dmd(stati_successivi, dmd)

    if dmd.matrix.ndim == 2:
        successivi_dmd = stati_correnti @ dmd.matrix.T
    elif dmd.matrix.ndim == 3:
        successivi_dmd = torch.einsum("brn,bmn->brm", stati_correnti, dmd.matrix)
    else:
        raise ValueError("dmd.matrix deve avere shape (K,K) oppure (B,K,K).")

    if punti_per_snapshot is not None and punti_per_snapshot < numero_punti:
        if punti_per_snapshot < 1:
            raise ValueError("punti_per_snapshot deve essere positivo oppure None.")
        indici_punti = torch.randperm(
            numero_punti,
            device=successivi_dmd.device,
        )[:punti_per_snapshot]
        successivi_dmd = successivi_dmd.index_select(dim=-1, index=indici_punti)
        stati_successivi = stati_successivi.index_select(dim=-1, index=indici_punti)

    return torch.mean((successivi_dmd - stati_successivi) ** 2)


def crea_esperimento(
    configurazione: ConfigurazioneEsperimento | None = None,
    dmd_operator: Any = None,
) -> dict[str, Any]:
    if configurazione is None:
        configurazione = ConfigurazioneEsperimento()

    np.random.seed(configurazione.seed)
    torch.manual_seed(configurazione.seed)

    device = risolvi_device(configurazione.device)
    rng = np.random.default_rng(configurazione.seed)

    train_dataset = DynabenchRolloutDataset(configurazione, split="train")
    val_dataset = DynabenchRolloutDataset(configurazione, split="val")

    source = dmd_operator if dmd_operator is not None else configurazione.dmd_operator_path
    dmd_provider = DMDOperatorProvider(source, cache_size=configurazione.dmd_cache_size)
    if configurazione.lambda_dmd != 0.0 and not dmd_provider.has_operator:
        if dmd_provider.missing_path is not None:
            raise ValueError(
                "La loss DMD ha peso diverso da zero, ma non trovo gli operatori "
                f"piDMD in {dmd_provider.missing_path}. Generali dal notebook "
                "Copia 05 oppure con: python koopman_pidmd.py --split train"
            )
        raise ValueError(
            "La loss DMD ha peso diverso da zero, quindi devi fornire una matrice "
            "con dmd_operator=... oppure --dmd-operator PERCORSO."
        )
    if configurazione.lambda_dmd != 0.0 and configurazione.batch_size_dmd < 1:
        raise ValueError("batch_size_dmd deve essere almeno 1 quando lambda_dmd != 0.")
    if (
        configurazione.lambda_dmd != 0.0
        and configurazione.dmd_punti_per_snapshot is not None
        and configurazione.dmd_punti_per_snapshot < 1
    ):
        raise ValueError("dmd_punti_per_snapshot deve essere positivo oppure None.")
    if configurazione.lambda_dmd != 0.0 and dmd_provider.directory is not None:
        numero_sim_train = int(sum(train_dataset.iterator.number_of_simulations))
        numero_operatori = dmd_provider.operator_count()
        if numero_operatori < numero_sim_train:
            raise ValueError(
                "Gli operatori piDMD sono incompleti: "
                f"trovati {numero_operatori}, richiesti {numero_sim_train}. "
                "Genera l'intero split train con: "
                "python koopman_pidmd.py --split train --out-dir results/per_sim_final"
            )

    modello = PINN(
        input_dim=3 + configurazione.lookback,
        output_dim=1,
        hidden_dim=configurazione.hidden_dim,
        num_layers=configurazione.num_layers,
        dropout=configurazione.dropout,
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
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "modello": modello,
        "optimizer": optimizer,
        "dmd_provider": dmd_provider,
    }


def step_training(esperimento: dict[str, Any]) -> dict[str, float]:
    configurazione: ConfigurazioneEsperimento = esperimento["configurazione"]
    device: torch.device = esperimento["device"]
    rng: np.random.Generator = esperimento["rng"]
    train_dataset: DynabenchRolloutDataset = esperimento["train_dataset"]
    modello: torch.nn.Module = esperimento["modello"]
    optimizer: torch.optim.Optimizer = esperimento["optimizer"]
    dmd_provider: DMDOperatorProvider = esperimento["dmd_provider"]

    modello.train()
    batch = train_dataset.sample_batch(rng, configurazione.batch_size, device)

    optimizer.zero_grad(set_to_none=True)
    fisica_attiva = configurazione.lambda_fisica != 0.0
    pred_rollout, features = forward_rollout(
        modello,
        batch,
        configurazione,
        require_grad=fisica_attiva,
    )
    loss_data = torch.mean((pred_rollout - batch.target) ** 2)

    if fisica_attiva:
        loss_fisica = loss_fisica_avvezione(
            pred_rollout.reshape(-1, 1),
            features,
            configurazione,
        )
    else:
        loss_fisica = loss_data.new_zeros(())

    if configurazione.lambda_dmd != 0.0:
        batch_dmd = train_dataset.sample_batch(rng, configurazione.batch_size_dmd, device)
        pred_dmd, _ = forward_rollout(
            modello,
            batch_dmd,
            configurazione,
            require_grad=False,
        )
        dmd = dmd_provider.get_for_batch(batch_dmd, device=device, dtype=batch_dmd.target.dtype)
        loss_dmd = loss_dmd_rollout(
            pred_dmd,
            batch_dmd,
            dmd,
            punti_per_snapshot=configurazione.dmd_punti_per_snapshot,
        )
    else:
        loss_dmd = loss_data.new_zeros(())

    loss_totale = (
        configurazione.lambda_data * loss_data
        + configurazione.lambda_fisica * loss_fisica
        + configurazione.lambda_dmd * loss_dmd
    )
    loss_totale.backward()
    if configurazione.clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(modello.parameters(), configurazione.clip_grad_norm)
    optimizer.step()

    return {
        "loss_totale": float(loss_totale.detach().cpu()),
        "loss_data": float(loss_data.detach().cpu()),
        "loss_fisica": float(loss_fisica.detach().cpu()),
        "loss_dmd": float(loss_dmd.detach().cpu()),
    }


@torch.no_grad()
def valida(esperimento: dict[str, Any], numero_batch: int | None = None) -> dict[str, float]:
    configurazione: ConfigurazioneEsperimento = esperimento["configurazione"]
    device: torch.device = esperimento["device"]
    rng: np.random.Generator = esperimento["rng"]
    val_dataset: DynabenchRolloutDataset = esperimento["val_dataset"]
    modello: torch.nn.Module = esperimento["modello"]

    numero_batch = configurazione.validation_batches if numero_batch is None else numero_batch
    modello.eval()
    somma_quadrati_per_tempo = torch.zeros(
        configurazione.rollout,
        device=device,
        dtype=torch.float64,
    )
    elementi_per_tempo = 0
    for _ in range(numero_batch):
        batch = val_dataset.sample_batch(rng, configurazione.batch_validation, device)
        pred_rollout, _ = forward_rollout(modello, batch, configurazione, require_grad=False)
        differenza = pred_rollout - batch.target
        somma_quadrati_per_tempo += torch.sum((differenza.double() ** 2), dim=(0, 2))
        elementi_per_tempo += differenza.shape[0] * differenza.shape[2]

    mse_per_tempo = somma_quadrati_per_tempo / max(elementi_per_tempo, 1)
    metriche = {
        "validation_mse_mean": float(torch.mean(mse_per_tempo).detach().cpu()),
        "validation_mse_rollout16": float(torch.mean(mse_per_tempo).detach().cpu()),
    }
    for indice, valore in enumerate(mse_per_tempo.detach().cpu().tolist(), start=1):
        metriche[f"validation_mse_step_{indice:02d}"] = float(valore)
    return metriche


def allena(
    esperimento: dict[str, Any],
    numero_valutazioni: int | None = None,
    stampa_ogni: int = 100,
    valida_ogni: int = 500,
) -> list[dict[str, float]]:
    configurazione: ConfigurazioneEsperimento = esperimento["configurazione"]
    if numero_valutazioni is None:
        numero_valutazioni = configurazione.numero_valutazioni_training

    storico: list[dict[str, float]] = []
    for valutazione in range(1, numero_valutazioni + 1):
        metriche = step_training(esperimento)
        metriche["valutazione"] = float(valutazione)

        if valutazione == 1 or (valida_ogni > 0 and valutazione % valida_ogni == 0):
            metriche.update(valida(esperimento))
            stampa_validation_terminal(metriche, valutazione, numero_valutazioni)

        storico.append(metriche)
        if stampa_ogni > 0 and (valutazione == 1 or valutazione % stampa_ogni == 0):
            messaggio = (
                f"[{valutazione:05d}/{numero_valutazioni}] "
                f"loss={metriche['loss_totale']:.6e} "
                f"data={metriche['loss_data']:.6e} "
                f"fisica={metriche['loss_fisica']:.6e} "
                f"dmd={metriche['loss_dmd']:.6e}"
            )
            if "validation_mse_rollout16" in metriche:
                messaggio += f" val_mse16={metriche['validation_mse_rollout16']:.6e}"
            print(messaggio)

    return storico


def stampa_validation_terminal(
    metriche: dict[str, float],
    valutazione: int,
    numero_valutazioni: int,
) -> None:
    step_keys = sorted(k for k in metriche if k.startswith("validation_mse_step_"))
    if not step_keys:
        return

    valori = [metriche[k] for k in step_keys]
    print(
        f"\nValidation [{valutazione:05d}/{numero_valutazioni}] "
        f"mean={metriche['validation_mse_mean']:.6e} "
        f"t+1={valori[0]:.6e} "
        f"t+{len(valori)}={valori[-1]:.6e} "
        f"min={min(valori):.6e} "
        f"max={max(valori):.6e}"
    )
    print("  MSE per rollout step:")
    for start in range(0, len(valori), 4):
        pezzi = []
        for offset, valore in enumerate(valori[start:start + 4], start=start + 1):
            pezzi.append(f"t+{offset:02d}={valore:.3e}")
        print("    " + "  ".join(pezzi))
    print()


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


def plotta_errori_validation(
    storico: list[dict[str, float]],
    path: str | Path = "results/validation_errors.png",
) -> None:
    righe_validation = [riga for riga in storico if "validation_mse_mean" in riga]
    if not righe_validation:
        print("Nessuna metrica di validation trovata: grafico non generato.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    steps = np.asarray([riga["valutazione"] for riga in righe_validation], dtype=np.float64)
    mse_mean = np.asarray([riga["validation_mse_mean"] for riga in righe_validation], dtype=np.float64)
    rollout_keys = [
        f"validation_mse_step_{indice:02d}"
        for indice in range(1, 1 + len([k for k in righe_validation[0] if k.startswith("validation_mse_step_")]))
    ]

    mse_matrix = np.asarray(
        [[riga[key] for riga in righe_validation] for key in rollout_keys],
        dtype=np.float64,
    )
    mse_log = np.log10(np.maximum(mse_matrix, 1e-12))
    rollout_steps = np.arange(1, len(rollout_keys) + 1)

    fig = plt.figure(figsize=(11, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.8, 1.0])
    ax_mean = fig.add_subplot(gs[0])
    ax_heatmap = fig.add_subplot(gs[1])
    ax_profile = fig.add_subplot(gs[2])

    ax_mean.plot(steps, mse_mean, marker="o", linewidth=2.0, color="black", label="media t+1..t+16")
    ax_mean.plot(
        steps,
        mse_matrix[0],
        marker=".",
        linewidth=1.2,
        color="#3b82f6",
        label="t+1",
    )
    ax_mean.plot(
        steps,
        mse_matrix[-1],
        marker=".",
        linewidth=1.2,
        color="#ef4444",
        label=f"t+{len(rollout_keys)}",
    )
    ax_mean.set_ylabel("Validation MSE")
    ax_mean.set_title("Validation: media rollout, primo e ultimo passo")
    if np.all(mse_mean > 0):
        ax_mean.set_yscale("log")
    ax_mean.grid(alpha=0.3)
    ax_mean.legend(loc="best")

    if len(steps) == 1:
        extent = [steps[0] - 0.5, steps[0] + 0.5, 0.5, len(rollout_keys) + 0.5]
    else:
        delta = float(np.median(np.diff(steps)))
        extent = [steps[0] - 0.5 * delta, steps[-1] + 0.5 * delta, 0.5, len(rollout_keys) + 0.5]
    image = ax_heatmap.imshow(
        mse_log,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    ax_heatmap.set_ylabel("Rollout step")
    ax_heatmap.set_title("Heatmap validation: log10(MSE) per orizzonte")
    ax_heatmap.set_yticks(rollout_steps)
    colorbar = fig.colorbar(image, ax=ax_heatmap)
    colorbar.set_label("log10(MSE)")

    ultimo_profilo = mse_matrix[:, -1]
    ax_profile.plot(rollout_steps, ultimo_profilo, marker="o", linewidth=2.0, color="#0f766e")
    ax_profile.set_xlabel("Rollout step futuro")
    ax_profile.set_ylabel("Validation MSE")
    ax_profile.set_title(f"Profilo ultimo checkpoint: step training {int(steps[-1])}")
    if np.all(ultimo_profilo > 0):
        ax_profile.set_yscale("log")
    ax_profile.set_xticks(rollout_steps)
    ax_profile.grid(alpha=0.3)

    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Grafico validation salvato in {path}")


def descrivi_esperimento(esperimento: dict[str, Any]) -> None:
    configurazione: ConfigurazioneEsperimento = esperimento["configurazione"]
    train_dataset: DynabenchRolloutDataset = esperimento["train_dataset"]
    val_dataset: DynabenchRolloutDataset = esperimento["val_dataset"]
    device: torch.device = esperimento["device"]

    print("Esperimento PINN/DMD su Dynabench advection")
    print(f"Device: {device}")
    print("Dominio spaziale: [0, 1] x [0, 1]")
    print(f"Dominio temporale: [{configurazione.t0}, {configurazione.t_finale}]")
    print(f"Iterator train: lookback={configurazione.lookback}, rollout={configurazione.rollout}")
    print(f"Finestre train: {len(train_dataset)}")
    print(f"Finestre val: {len(val_dataset)}")
    print(f"Punti cloud per snapshot: {train_dataset.numero_punti}")
    punti_dmd = (
        train_dataset.numero_punti
        if configurazione.dmd_punti_per_snapshot is None
        else min(configurazione.dmd_punti_per_snapshot, train_dataset.numero_punti)
    )
    print(
        "Collocation DMD per step: "
        f"{configurazione.batch_size_dmd} finestre x "
        f"{configurazione.rollout} tempi x "
        f"{punti_dmd} punti"
    )
    print(f"Griglia dynabench.grid: {train_dataset.griglia_dominio}")
    print(
        "Pesi loss: "
        f"data={configurazione.lambda_data}, "
        f"fisica={configurazione.lambda_fisica}, "
        f"dmd={configurazione.lambda_dmd}"
    )
    print(f"Valutazioni training: {configurazione.numero_valutazioni_training}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training PINN su Dynabench advection (lookback=1, rollout=16).")
    parser.add_argument("--dmd-operator", help="File .npy/.npz/.pt/.csv o directory di operatori per-simulazione.")
    parser.add_argument("--evals", type=int, help="Numero di valutazioni/step di training.")
    parser.add_argument("--batch-size", type=int, help="Batch size train in finestre Dynabench.")
    parser.add_argument("--batch-size-dmd", type=int, help="Finestre Dynabench usate solo per la DMD loss.")
    parser.add_argument("--dmd-punti-per-snapshot", type=int, help="Punti spaziali usati nella DMD loss; 225 usa tutto.")
    parser.add_argument("--validation-batches", type=int, help="Numero batch usati per la validation.")
    parser.add_argument("--batch-validation", type=int, help="Batch size validation in finestre Dynabench.")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0 oppure auto.")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--save-model", help="Percorso .pt in cui salvare pesi e configurazione.")
    parser.add_argument("--history-csv", default="results/training_history.csv", help="CSV dello storico training/validation.")
    parser.add_argument("--validation-plot", default="results/validation_errors.png", help="PNG con errori validation.")
    parser.add_argument("--download", action="store_true", help="Scarica il dataset se non presente.")
    parser.add_argument("--lambda-data", type=float, help="Peso della data loss.")
    parser.add_argument("--lambda-fisica", type=float, help="Peso della physics loss.")
    parser.add_argument("--lambda-dmd", type=float, help="Peso della DMD loss.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configurazione = ConfigurazioneEsperimento(download=args.download)
    if args.dmd_operator is not None:
        configurazione = replace(configurazione, dmd_operator_path=args.dmd_operator)
    if args.evals is not None:
        configurazione = replace(configurazione, numero_valutazioni_training=args.evals)
    if args.batch_size is not None:
        configurazione = replace(configurazione, batch_size=args.batch_size)
    if args.batch_size_dmd is not None:
        configurazione = replace(configurazione, batch_size_dmd=args.batch_size_dmd)
    if args.dmd_punti_per_snapshot is not None:
        configurazione = replace(configurazione, dmd_punti_per_snapshot=args.dmd_punti_per_snapshot)
    if args.validation_batches is not None:
        configurazione = replace(configurazione, validation_batches=args.validation_batches)
    if args.batch_validation is not None:
        configurazione = replace(configurazione, batch_validation=args.batch_validation)
    if args.device is not None:
        configurazione = replace(configurazione, device=args.device)
    if args.lambda_data is not None:
        configurazione = replace(configurazione, lambda_data=args.lambda_data)
    if args.lambda_fisica is not None:
        configurazione = replace(configurazione, lambda_fisica=args.lambda_fisica)
    if args.lambda_dmd is not None:
        configurazione = replace(configurazione, lambda_dmd=args.lambda_dmd)

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
    if args.validation_plot:
        plotta_errori_validation(storico, args.validation_plot)

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
