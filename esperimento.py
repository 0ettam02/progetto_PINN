from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from pinn import AdvectionCNN, PINN


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
    model_architecture: str = "cnn"
    hidden_dim: int = 64
    num_layers: int = 6
    dropout: float = 0.0
    optimizer_name: str = "adamw_amsgrad"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    lr_scheduler: str = "onecycle"
    onecycle_pct_start: float = 0.15
    onecycle_div_factor: float = 20.0
    onecycle_final_div_factor: float = 100.0
    numero_valutazioni_training: int = 5000
    batch_size: int = 32
    batch_size_dmd: int = 16
    dmd_punti_per_snapshot: int | None = None
    batch_validation: int = 32
    validation_batches: int | None = 20
    lambda_data: float = 1.0
    lambda_fisica: float = 0.0
    lambda_dmd: float = 1.0
    seed: int = 42
    device: str = "auto"
    dmd_operator_path: str | None = "results/per_sim_final"
    dmd_svd_rank: int = 58
    dmd_auto_generate: bool = True
    download: bool = False
    clip_grad_norm: float | None = 1.0
    dmd_cache_size: int = 10
    cnn_interpolation_neighbors: int = 8
    cnn_use_coordinates: bool = True


class CampionatoreQuasiMonteCarlo:
    """Campionatore Sobol con stream separati per train, validation e DMD."""

    def __init__(self, seed: int):
        self.seed = seed
        self._engines: dict[tuple[str, int], torch.quasirandom.SobolEngine] = {}

    def draw(self, stream: str, dimensione: int, n: int) -> np.ndarray:
        if dimensione < 1:
            raise ValueError("dimensione deve essere positiva.")
        if n < 1:
            raise ValueError("n deve essere positivo.")

        key = (stream, dimensione)
        if key not in self._engines:
            self._engines[key] = torch.quasirandom.SobolEngine(
                dimension=dimensione,
                scramble=True,
                seed=self.seed + self._stream_offset(stream, dimensione),
            )
        return self._engines[key].draw(n).cpu().numpy()

    def indici(self, stream: str, massimo: int, n: int) -> np.ndarray:
        if massimo < 1:
            raise ValueError("massimo deve essere positivo.")
        valori = self.draw(stream, 1, n)[:, 0]
        indici = np.floor(valori * massimo).astype(np.int64)
        return np.clip(indici, 0, massimo - 1)

    def indici_unici(
        self,
        stream: str,
        massimo: int,
        n: int,
    ) -> np.ndarray:
        if n >= massimo:
            return np.arange(massimo, dtype=np.int64)

        indici = np.empty(0, dtype=np.int64)
        while indici.size < n:
            nuovi = self.indici(stream, massimo, max(2 * n, 16))
            indici = np.unique(np.concatenate([indici, nuovi]))
        return np.sort(indici[:n])

    def _stream_offset(self, stream: str, dimensione: int) -> int:
        offset = 9973 * dimensione
        for indice, carattere in enumerate(stream, start=1):
            offset += indice * ord(carattere)
        return offset


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


class KoopmanPIDMDOperatorFactory:
    """Genera gli operatori piDMD mancanti usando la pipeline del notebook Copia 05."""

    def __init__(
        self,
        out_dir: str | Path,
        configurazione: ConfigurazioneEsperimento,
        split: str = "train",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.configurazione = configurazione
        self.split = split
        self._sim_iterator = None
        self._tools_loaded = False
        self._iterator_cls: Any = None
        self._extract_features: Any = None
        self._save_features: Any = None

    def __call__(self, indice_simulazione: int) -> Path:
        output_path = self.out_dir / self._filename(indice_simulazione)
        if output_path.exists():
            return output_path

        sim_iterator = self._get_sim_iterator()
        if indice_simulazione < 0 or indice_simulazione >= len(sim_iterator):
            raise IndexError(
                "Indice simulazione fuori range per generazione piDMD: "
                f"{indice_simulazione}"
            )

        print(
            "Operatore piDMD mancante: "
            f"genero sim {indice_simulazione:05d} e salvo in {output_path}"
        )
        features = self._extract_features(
            indice_simulazione,
            sim_iterator,
            svd_rank=self.configurazione.dmd_svd_rank,
        )
        return self._save_features(
            features,
            out_dir=self.out_dir,
            svd_rank=self.configurazione.dmd_svd_rank,
        )

    def _filename(self, indice_simulazione: int) -> str:
        return (
            f"koopman_pidmd_r{self.configurazione.dmd_svd_rank}_"
            f"sim{indice_simulazione:05d}.npz"
        )

    def _get_sim_iterator(self) -> Any:
        self._load_tools()
        if self._sim_iterator is None:
            self._sim_iterator = self._iterator_cls(
                split=self.split,
                equation=self.configurazione.equation,
                structure=self.configurazione.structure,
                resolution=self.configurazione.resolution,
                base_path=self.configurazione.data_root,
                download=self.configurazione.download,
            )
        return self._sim_iterator

    def _load_tools(self) -> None:
        if self._tools_loaded:
            return
        try:
            from koopman_pidmd import (
                DynabenchSimulationIterator,
                extract_koopman_features,
                save_koopman_features,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Non posso generare gli operatori piDMD: installa le dipendenze "
                "con `python -m pip install -r requirements.txt`."
            ) from exc

        self._iterator_cls = DynabenchSimulationIterator
        self._extract_features = extract_koopman_features
        self._save_features = save_koopman_features
        self._tools_loaded = True


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
        try:
            from dynabench.dataset import DynabenchIterator
            from dynabench.grid import Grid
        except ImportError as exc:
            raise RuntimeError(
                "dynabench non e' installato. Installa le dipendenze con "
                "`python -m pip install -r requirements.txt`."
            ) from exc

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
        self.grid_shape = (lato, lato)
        self.grid_points = self._costruisci_griglia_regolare(lato)

        self.griglia_dominio = Grid(
            grid_size=(lato, lato),
            grid_limits=((0.0, 1.0), (0.0, 1.0)),
        )
        self.simulation_starts = np.concatenate(
            [
                np.array([0], dtype=np.int64),
                np.cumsum(self.iterator.number_of_simulations, dtype=np.int64),
            ]
        )
        self.offset_simulazioni = self.simulation_starts[:-1]
        self.numero_simulazioni_totali = int(self.simulation_starts[-1])
        self._controlla_punti_nel_dominio(sample.pos)

    def __len__(self) -> int:
        return len(self.iterator)

    def _costruisci_griglia_regolare(self, lato: int) -> np.ndarray:
        coordinate = np.linspace(0.0, 1.0, lato, dtype=np.float32)
        xx, yy = np.meshgrid(coordinate, coordinate, indexing="xy")
        return np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float32)

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

    def codifica_indice(self, indice_sim_globale: int, indice_temporale: int) -> int:
        if indice_sim_globale < 0 or indice_sim_globale >= self.numero_simulazioni_totali:
            raise IndexError("Indice simulazione fuori range.")

        indice_file = int(np.searchsorted(self.simulation_starts[1:], indice_sim_globale, side="right"))
        indice_sim_locale = int(indice_sim_globale - self.simulation_starts[indice_file])
        usable_len = int(self.iterator.usable_simulation_lengths[indice_file])
        indice_temporale = min(max(int(indice_temporale), 0), usable_len - 1)
        return int(
            self.iterator.starting_indices[indice_file]
            + indice_sim_locale * usable_len
            + indice_temporale
        )

    def sample_indici_quasi_montecarlo(
        self,
        qmc: CampionatoreQuasiMonteCarlo,
        stream: str,
        batch_size: int,
    ) -> np.ndarray:
        punti = qmc.draw(stream, 2, batch_size)
        indici_simulazione = np.floor(
            punti[:, 0] * self.numero_simulazioni_totali
        ).astype(np.int64)
        indici_simulazione = np.clip(
            indici_simulazione,
            0,
            self.numero_simulazioni_totali - 1,
        )

        indici = []
        for valore_tempo, indice_sim_globale in zip(punti[:, 1], indici_simulazione):
            indice_file = int(np.searchsorted(self.simulation_starts[1:], indice_sim_globale, side="right"))
            usable_len = int(self.iterator.usable_simulation_lengths[indice_file])
            indice_temporale = int(np.floor(valore_tempo * usable_len))
            indice_temporale = min(max(indice_temporale, 0), usable_len - 1)
            indici.append(self.codifica_indice(int(indice_sim_globale), indice_temporale))

        return np.asarray(indici, dtype=np.int64)

    def sample_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        qmc: CampionatoreQuasiMonteCarlo | None = None,
        stream: str = "train",
    ) -> BatchRollout:
        if qmc is None:
            indici = rng.integers(0, len(self), size=batch_size, endpoint=False)
        else:
            indici = self.sample_indici_quasi_montecarlo(qmc, stream, batch_size)

        return self.batch_da_indici(indici, device=device, dtype=dtype)

    def batch_da_indici(
        self,
        indici: np.ndarray,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> BatchRollout:
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
    def __init__(
        self,
        source: Any = None,
        cache_size: int = 32,
        missing_operator_factory: Callable[[int], Path] | None = None,
        allow_missing_directory: bool = False,
    ):
        self.source = source
        self.cache_size = cache_size
        self._missing_operator_factory = missing_operator_factory
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
            if allow_missing_directory and path.suffix == "":
                self._directory = path
                return
            self._missing_path = path
            return
        if path.is_dir():
            if not any(path.glob("*.npz")) and not allow_missing_directory:
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

    @property
    def can_generate_missing(self) -> bool:
        return self._directory is not None and self._missing_operator_factory is not None

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
        if not candidates and self._missing_operator_factory is not None:
            generated_path = self._missing_operator_factory(indice_simulazione)
            if generated_path.exists():
                candidates = [generated_path]
            else:
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


def _dmd_source_supports_auto_generation(source: Any) -> bool:
    if source is None or isinstance(source, (np.ndarray, torch.Tensor)):
        return False

    path = Path(source)
    if path.exists():
        return path.is_dir()

    operator_file_suffixes = {".npy", ".npz", ".pt", ".pth", ".csv", ".txt"}
    return path.suffix.lower() not in operator_file_suffixes


def costruisci_optimizer(
    parametri: list[torch.nn.Parameter],
    configurazione: ConfigurazioneEsperimento,
) -> torch.optim.Optimizer:
    if configurazione.optimizer_name == "adamw_amsgrad":
        return torch.optim.AdamW(
            parametri,
            lr=configurazione.learning_rate,
            betas=(configurazione.adam_beta1, configurazione.adam_beta2),
            weight_decay=configurazione.weight_decay,
            amsgrad=True,
        )
    if configurazione.optimizer_name == "adamw":
        return torch.optim.AdamW(
            parametri,
            lr=configurazione.learning_rate,
            betas=(configurazione.adam_beta1, configurazione.adam_beta2),
            weight_decay=configurazione.weight_decay,
        )
    raise ValueError(
        "optimizer_name non supportato: "
        f"{configurazione.optimizer_name!r}. Usa 'adamw_amsgrad' oppure 'adamw'."
    )


def costruisci_scheduler(
    optimizer: torch.optim.Optimizer,
    configurazione: ConfigurazioneEsperimento,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if configurazione.lr_scheduler == "none":
        return None
    if configurazione.numero_valutazioni_training < 1:
        return None
    if configurazione.lr_scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=configurazione.learning_rate,
            total_steps=configurazione.numero_valutazioni_training,
            pct_start=configurazione.onecycle_pct_start,
            anneal_strategy="cos",
            div_factor=configurazione.onecycle_div_factor,
            final_div_factor=configurazione.onecycle_final_div_factor,
        )
    raise ValueError(
        "lr_scheduler non supportato: "
        f"{configurazione.lr_scheduler!r}. Usa 'onecycle' oppure 'none'."
    )


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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if hasattr(modello, "forward_cloud"):
        if require_grad:
            raise ValueError("La loss fisica pointwise non e' supportata dal modello CNN.")
        return modello.forward_cloud(batch.u0, batch.pos), None

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
    indici_punti: torch.Tensor | None = None,
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
        if indici_punti is None:
            indici_punti = torch.linspace(
                0,
                numero_punti - 1,
                punti_per_snapshot,
                device=successivi_dmd.device,
            ).round().long()
        else:
            indici_punti = indici_punti.to(device=successivi_dmd.device, dtype=torch.long)
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
    qmc = CampionatoreQuasiMonteCarlo(configurazione.seed)

    train_dataset = DynabenchRolloutDataset(configurazione, split="train")
    val_dataset = DynabenchRolloutDataset(configurazione, split="val")
    if configurazione.model_architecture == "cnn" and configurazione.lambda_fisica != 0.0:
        raise ValueError(
            "La loss fisica pointwise non e' supportata con model_architecture='cnn'. "
            "Usa --lambda-fisica 0 oppure --model pinn."
        )

    source = dmd_operator if dmd_operator is not None else configurazione.dmd_operator_path
    dmd_factory = None
    allow_missing_dmd_directory = False
    if (
        configurazione.lambda_dmd != 0.0
        and configurazione.dmd_auto_generate
        and _dmd_source_supports_auto_generation(source)
    ):
        dmd_factory = KoopmanPIDMDOperatorFactory(
            out_dir=Path(source),
            configurazione=configurazione,
            split="train",
        )
        allow_missing_dmd_directory = True

    dmd_provider = DMDOperatorProvider(
        source,
        cache_size=configurazione.dmd_cache_size,
        missing_operator_factory=dmd_factory,
        allow_missing_directory=allow_missing_dmd_directory,
    )
    dmd_operator_apprendibile = None
    if configurazione.lambda_dmd != 0.0 and not dmd_provider.has_operator:
        dmd_operator_apprendibile = torch.nn.Parameter(
            torch.eye(
                train_dataset.numero_punti,
                device=device,
                dtype=torch.float32,
            )
        )
        sorgente_mancante = (
            f" in {dmd_provider.missing_path}"
            if dmd_provider.missing_path is not None
            else ""
        )
        print(
            "Avviso: operatori piDMD non trovati"
            f"{sorgente_mancante}. Uso un operatore DMD globale apprendibile."
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
            if dmd_provider.can_generate_missing:
                print(
                    "Operatori piDMD incompleti: "
                    f"trovati {numero_operatori}, richiesti {numero_sim_train}. "
                    "I file mancanti verranno generati e salvati quando richiesti."
                )
            else:
                raise ValueError(
                    "Gli operatori piDMD sono incompleti: "
                    f"trovati {numero_operatori}, richiesti {numero_sim_train}. "
                    "Genera l'intero split train con: "
                    "python koopman_pidmd.py --split train --out-dir results/per_sim_final"
                )

    if configurazione.model_architecture == "cnn":
        modello = AdvectionCNN(
            lookback=configurazione.lookback,
            rollout=configurazione.rollout,
            grid_points=torch.as_tensor(train_dataset.grid_points, dtype=torch.float32),
            grid_shape=train_dataset.grid_shape,
            hidden_dim=configurazione.hidden_dim,
            num_layers=configurazione.num_layers,
            dropout=configurazione.dropout,
            interpolation_neighbors=configurazione.cnn_interpolation_neighbors,
            use_coordinates=configurazione.cnn_use_coordinates,
        ).to(device)
    elif configurazione.model_architecture == "pinn":
        modello = PINN(
            input_dim=3 + configurazione.lookback,
            output_dim=1,
            hidden_dim=configurazione.hidden_dim,
            num_layers=configurazione.num_layers,
            dropout=configurazione.dropout,
        ).to(device)
    else:
        raise ValueError(
            "model_architecture non supportata: "
            f"{configurazione.model_architecture!r}. Usa 'cnn' oppure 'pinn'."
        )
    parametri_ottimizzazione = list(modello.parameters())
    if dmd_operator_apprendibile is not None:
        parametri_ottimizzazione.append(dmd_operator_apprendibile)

    optimizer = costruisci_optimizer(parametri_ottimizzazione, configurazione)
    scheduler = costruisci_scheduler(optimizer, configurazione)

    return {
        "configurazione": configurazione,
        "device": device,
        "rng": rng,
        "qmc": qmc,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "modello": modello,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scheduler_step": 0,
        "dmd_provider": dmd_provider,
        "dmd_operator_apprendibile": dmd_operator_apprendibile,
    }


def step_training(esperimento: dict[str, Any]) -> dict[str, float]:
    configurazione: ConfigurazioneEsperimento = esperimento["configurazione"]
    device: torch.device = esperimento["device"]
    rng: np.random.Generator = esperimento["rng"]
    qmc: CampionatoreQuasiMonteCarlo = esperimento["qmc"]
    train_dataset: DynabenchRolloutDataset = esperimento["train_dataset"]
    modello: torch.nn.Module = esperimento["modello"]
    optimizer: torch.optim.Optimizer = esperimento["optimizer"]
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = esperimento["scheduler"]
    dmd_provider: DMDOperatorProvider = esperimento["dmd_provider"]
    dmd_operator_apprendibile: torch.nn.Parameter | None = esperimento["dmd_operator_apprendibile"]

    modello.train()
    batch = train_dataset.sample_batch(
        rng,
        configurazione.batch_size,
        device,
        qmc=qmc,
        stream="train_data",
    )

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
        if features is None:
            raise ValueError("La loss fisica richiede feature pointwise; usa --model pinn.")
        loss_fisica = loss_fisica_avvezione(
            pred_rollout.reshape(-1, 1),
            features,
            configurazione,
        )
    else:
        loss_fisica = loss_data.new_zeros(())

    if configurazione.lambda_dmd != 0.0:
        batch_dmd = train_dataset.sample_batch(
            rng,
            configurazione.batch_size_dmd,
            device,
            qmc=qmc,
            stream="train_dmd",
        )
        pred_dmd, _ = forward_rollout(
            modello,
            batch_dmd,
            configurazione,
            require_grad=False,
        )
        if dmd_operator_apprendibile is None:
            dmd = dmd_provider.get_for_batch(batch_dmd, device=device, dtype=batch_dmd.target.dtype)
        else:
            dmd = DMDTensors(matrix=dmd_operator_apprendibile.to(dtype=batch_dmd.target.dtype))
        indici_punti = None
        if (
            configurazione.dmd_punti_per_snapshot is not None
            and configurazione.dmd_punti_per_snapshot < batch_dmd.u0.shape[-2]
        ):
            indici_punti = torch.as_tensor(
                qmc.indici_unici(
                    "train_dmd_points",
                    batch_dmd.u0.shape[-2],
                    configurazione.dmd_punti_per_snapshot,
                ),
                device=device,
                dtype=torch.long,
            )
        loss_dmd = loss_dmd_rollout(
            pred_dmd,
            batch_dmd,
            dmd,
            punti_per_snapshot=configurazione.dmd_punti_per_snapshot,
            indici_punti=indici_punti,
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
    if scheduler is not None and esperimento["scheduler_step"] < configurazione.numero_valutazioni_training:
        scheduler.step()
        esperimento["scheduler_step"] += 1

    return {
        "loss_totale": float(loss_totale.detach().cpu()),
        "loss_data": float(loss_data.detach().cpu()),
        "loss_fisica": float(loss_fisica.detach().cpu()),
        "loss_dmd": float(loss_dmd.detach().cpu()),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


@torch.no_grad()
def valida(esperimento: dict[str, Any], numero_batch: int | None = None) -> dict[str, float]:
    configurazione: ConfigurazioneEsperimento = esperimento["configurazione"]
    device: torch.device = esperimento["device"]
    val_dataset: DynabenchRolloutDataset = esperimento["val_dataset"]
    modello: torch.nn.Module = esperimento["modello"]

    if numero_batch is None:
        numero_batch = configurazione.validation_batches
    if numero_batch == 0:
        numero_batch = None
    numero_finestre = len(val_dataset)
    if numero_batch is not None:
        numero_finestre = min(numero_finestre, numero_batch * configurazione.batch_validation)

    modello.eval()
    somma_quadrati_per_tempo = torch.zeros(
        configurazione.rollout,
        device=device,
        dtype=torch.float64,
    )
    elementi_per_tempo = 0
    for inizio in range(0, numero_finestre, configurazione.batch_validation):
        fine = min(inizio + configurazione.batch_validation, numero_finestre)
        indici = np.arange(inizio, fine, dtype=np.int64)
        batch = val_dataset.batch_da_indici(
            indici,
            device=device,
        )
        pred_rollout, _ = forward_rollout(modello, batch, configurazione, require_grad=False)
        differenza = pred_rollout - batch.target
        somma_quadrati_per_tempo += torch.sum((differenza.double() ** 2), dim=(0, 2))
        elementi_per_tempo += differenza.shape[0] * differenza.shape[2]

    mse_per_tempo = somma_quadrati_per_tempo / max(elementi_per_tempo, 1)
    metriche = {
        "validation_mse_mean": float(torch.mean(mse_per_tempo).detach().cpu()),
        "validation_mse_rollout16": float(torch.mean(mse_per_tempo).detach().cpu()),
        "validation_windows": float(numero_finestre),
    }
    for indice, valore in enumerate(mse_per_tempo.detach().cpu().tolist(), start=1):
        metriche[f"validation_mse_step_{indice:02d}"] = float(valore)
    return metriche


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
    for valutazione in range(1, numero_valutazioni + 1):
        metriche = step_training(esperimento)
        metriche["valutazione"] = float(valutazione)

        if valida_ogni > 0 and valutazione % valida_ogni == 0:
            metriche.update(valida(esperimento))
            stampa_validation_terminal(metriche, valutazione, numero_valutazioni)

        storico.append(metriche)
        if stampa_ogni > 0 and (valutazione == 1 or valutazione % stampa_ogni == 0):
            messaggio = (
                f"[{valutazione:05d}/{numero_valutazioni}] "
                f"loss={metriche['loss_totale']:.6e} "
                f"data={metriche['loss_data']:.6e} "
                f"fisica={metriche['loss_fisica']:.6e} "
                f"dmd={metriche['loss_dmd']:.6e} "
                f"lr={metriche['learning_rate']:.3e}"
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
        f"windows={int(metriche['validation_windows'])} "
        f"t+1={valori[0]:.6e} "
        f"t+{len(valori)}={valori[-1]:.6e} "
        f"min={min(valori):.6e} "
        f"max={max(valori):.6e}"
    )
    print("  MSE validation per t di rollout:")
    for offset, valore in enumerate(valori, start=1):
        print(f"    t+{offset:02d}: {valore:.6e}")
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
    dmd_provider: DMDOperatorProvider = esperimento["dmd_provider"]
    dmd_operator_apprendibile = esperimento["dmd_operator_apprendibile"]

    print("Esperimento su Dynabench advection")
    print(f"Device: {device}")
    print(
        "Modello: "
        f"{configurazione.model_architecture}, "
        f"hidden_dim={configurazione.hidden_dim}, "
        f"num_layers={configurazione.num_layers}, "
        f"dropout={configurazione.dropout}"
    )
    if configurazione.model_architecture == "cnn":
        print(
            "CNN cloud->grid: "
            f"griglia={train_dataset.grid_shape[0]}x{train_dataset.grid_shape[1]}, "
            f"vicini IDW={configurazione.cnn_interpolation_neighbors}, "
            f"coordinate={configurazione.cnn_use_coordinates}"
        )
    print("Dominio spaziale: [0, 1] x [0, 1]")
    print(f"Dominio temporale: [{configurazione.t0}, {configurazione.t_finale}]")
    print(f"Iterator train: lookback={configurazione.lookback}, rollout={configurazione.rollout}")
    print(f"Finestre train: {len(train_dataset)}")
    print(f"Finestre val: {len(val_dataset)}")
    if configurazione.validation_batches is None or configurazione.validation_batches == 0:
        print("Validation: dataset val completo, sequenziale")
    else:
        finestre_validation = min(
            len(val_dataset),
            configurazione.validation_batches * configurazione.batch_validation,
        )
        print(f"Validation: prime {finestre_validation} finestre del dataset val")
    print(f"Punti cloud per snapshot: {train_dataset.numero_punti}")
    print(
        "Data loss per step: "
        f"{configurazione.batch_size} finestre x "
        f"{configurazione.rollout} tempi x "
        f"{train_dataset.numero_punti} punti"
    )
    if configurazione.lambda_dmd != 0.0:
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
    else:
        print("Collocation DMD per step: disattivata")
    print(f"Griglia dynabench.grid: {train_dataset.griglia_dominio}")
    print(
        "Campionamento: Sobol quasi-Monte Carlo su finestre "
        "Dynabench, con dimensioni separate per simulazioni e tempi; "
        "ogni finestra usa tutti i punti spaziali del cloud."
    )
    print(
        "Pesi loss: "
        f"data={configurazione.lambda_data}, "
        f"fisica={configurazione.lambda_fisica}, "
        f"dmd={configurazione.lambda_dmd}"
    )
    print(
        "Optimizer: "
        f"{configurazione.optimizer_name}, "
        f"max_lr={configurazione.learning_rate}, "
        f"weight_decay={configurazione.weight_decay}, "
        f"betas=({configurazione.adam_beta1}, {configurazione.adam_beta2}), "
        f"scheduler={configurazione.lr_scheduler}, "
        f"clip_grad_norm={configurazione.clip_grad_norm}"
    )
    if configurazione.lambda_dmd != 0.0:
        sorgente_dmd = (
            "operatore globale apprendibile"
            if dmd_operator_apprendibile is not None
            else (
                "operatori piDMD da file, con generazione automatica dei mancanti"
                if dmd_provider.can_generate_missing
                else "operatori piDMD caricati da file"
            )
        )
        print(f"Sorgente DMD: {sorgente_dmd}")
    print(f"Valutazioni training: {configurazione.numero_valutazioni_training}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training su Dynabench advection (lookback=1, rollout=16).")
    parser.add_argument("--dmd-operator", help="File .npy/.npz/.pt/.csv o directory di operatori per-simulazione.")
    parser.add_argument("--dmd-svd-rank", type=int, help="Rank SVD usato per generare operatori piDMD mancanti.")
    parser.add_argument("--no-dmd-auto-generate", action="store_true", help="Disabilita la generazione automatica degli operatori piDMD mancanti.")
    parser.add_argument("--evals", type=int, help="Numero di valutazioni/step di training.")
    parser.add_argument("--batch-size", type=int, help="Batch size train in finestre Dynabench.")
    parser.add_argument("--batch-size-dmd", type=int, help="Finestre Dynabench usate solo per la DMD loss.")
    parser.add_argument("--dmd-punti-per-snapshot", type=int, help="Punti spaziali usati nella DMD loss; se omesso usa tutta la griglia.")
    parser.add_argument("--validation-batches", type=int, help="Limita il numero di batch sequenziali usati per la validation; 0 usa tutto il dataset val.")
    parser.add_argument("--batch-validation", type=int, help="Batch size validation in finestre Dynabench.")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0 oppure auto.")
    parser.add_argument("--model", choices=["cnn", "pinn"], help="Architettura del modello.")
    parser.add_argument("--hidden-dim", type=int, help="Canali CNN o neuroni hidden della PINN.")
    parser.add_argument("--num-layers", type=int, help="Blocchi residuali CNN o layer hidden della PINN.")
    parser.add_argument("--dropout", type=float, help="Dropout del modello.")
    parser.add_argument("--cnn-neighbors", type=int, help="Numero di vicini IDW per interpolare cloud e griglia.")
    parser.add_argument("--no-cnn-coordinates", action="store_true", help="Non aggiunge canali coordinate x/y alla CNN.")
    parser.add_argument("--optimizer", choices=["adamw_amsgrad", "adamw"], help="Ottimizzatore usato per il modello.")
    parser.add_argument("--learning-rate", type=float, help="Learning rate massimo; con onecycle e' il max_lr.")
    parser.add_argument("--weight-decay", type=float, help="Weight decay dell'ottimizzatore.")
    parser.add_argument("--lr-scheduler", choices=["onecycle", "none"], help="Scheduler del learning rate.")
    parser.add_argument("--clip-grad-norm", type=float, help="Clipping L2 del gradiente; usa 0 per disattivarlo.")
    parser.add_argument("--print-every", type=int, default=50)
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
    if args.dmd_svd_rank is not None:
        configurazione = replace(configurazione, dmd_svd_rank=args.dmd_svd_rank)
    if args.no_dmd_auto_generate:
        configurazione = replace(configurazione, dmd_auto_generate=False)
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
    if args.model is not None:
        configurazione = replace(configurazione, model_architecture=args.model)
    if args.hidden_dim is not None:
        configurazione = replace(configurazione, hidden_dim=args.hidden_dim)
    if args.num_layers is not None:
        configurazione = replace(configurazione, num_layers=args.num_layers)
    if args.dropout is not None:
        configurazione = replace(configurazione, dropout=args.dropout)
    if args.cnn_neighbors is not None:
        configurazione = replace(configurazione, cnn_interpolation_neighbors=args.cnn_neighbors)
    if args.no_cnn_coordinates:
        configurazione = replace(configurazione, cnn_use_coordinates=False)
    if args.optimizer is not None:
        configurazione = replace(configurazione, optimizer_name=args.optimizer)
    if args.learning_rate is not None:
        configurazione = replace(configurazione, learning_rate=args.learning_rate)
    if args.weight_decay is not None:
        configurazione = replace(configurazione, weight_decay=args.weight_decay)
    if args.lr_scheduler is not None:
        configurazione = replace(configurazione, lr_scheduler=args.lr_scheduler)
    if args.clip_grad_norm is not None:
        configurazione = replace(
            configurazione,
            clip_grad_norm=None if args.clip_grad_norm == 0 else args.clip_grad_norm,
        )
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
                "dmd_operator_apprendibile": (
                    esperimento["dmd_operator_apprendibile"].detach().cpu()
                    if esperimento["dmd_operator_apprendibile"] is not None
                    else None
                ),
                "configurazione": configurazione.__dict__,
                "storico": storico,
            },
            path,
        )
        print(f"Modello salvato in {path}")


if __name__ == "__main__":
    main()
