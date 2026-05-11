from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from dynabench.dataset import download_equation
from dynabench.dataset._base import BaseListSimulationIterator
from dynabench.dataset.transforms import DefaultTransform


SVD_RANK = 58


class DynabenchSimulationIterator(BaseListSimulationIterator):
    """Simulation iterator used by the Copia 05 Koopman-piDMD notebook."""

    def __init__(
        self,
        split: str = "train",
        equation: str = "advection",
        structure: str = "cloud",
        resolution: str = "low",
        transforms=DefaultTransform(),
        base_path: str = "data",
        download: bool = False,
        dtype: np.dtype = np.float32,
    ) -> None:
        if download:
            download_equation(equation, structure, resolution, data_dir=base_path)

        self.file_list = glob.glob(
            os.path.join(base_path, equation, structure, resolution, f"*{split}*.h5")
        )
        super().__init__(
            data_paths=self.file_list,
            is_batched=True,
            transforms=transforms,
            dtype=dtype,
        )


def build_snapshot_matrix(simulation_data: np.ndarray, delta_t: int = 1) -> np.ndarray:
    data = simulation_data[::delta_t]
    return data[:, :, 0].T.astype(np.float64)


def normalize_minmax(
    x: np.ndarray,
    feature_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, dict[str, float | tuple[float, float]]]:
    a, b = feature_range
    x_min = float(x.min())
    x_max = float(x.max())
    x_norm = (b - a) * (x - x_min) / (x_max - x_min + 1e-8) + a
    return x_norm, {"min": x_min, "max": x_max, "range": feature_range}


def extract_koopman_features(
    sim_index: int,
    sim_iterator: DynabenchSimulationIterator,
    svd_rank: int = SVD_RANK,
) -> dict[str, np.ndarray | int | dict[str, float | tuple[float, float]]]:
    try:
        from pydmd import PiDMD
    except ImportError as exc:
        raise RuntimeError(
            "pydmd non e' installato. Esegui: python -m pip install -r requirements.txt"
        ) from exc

    sample = sim_iterator[sim_index]
    x_raw = build_snapshot_matrix(sample.x)
    x_norm, norm_params = normalize_minmax(x_raw)

    # The Copia 05 notebook reconstructs A from modes/eigenvalues below.
    # Keeping compute_A=False avoids an internal pseudo-inverse that can fail
    # on very ill-conditioned cloud trajectories even when modes/eigs are valid.
    model = PiDMD(manifold="unitary", svd_rank=svd_rank, compute_A=False)
    model.fit(x_norm)
    a_matrix = (model.modes @ np.diag(model.eigs) @ model.modes.conj().T).real

    return {
        "A": a_matrix,
        "modes": model.modes,
        "eigs": model.eigs,
        "omega": np.log(model.eigs),
        "pts": sample.pos,
        "norm_params": norm_params,
        "sim_index": sim_index,
    }


def save_koopman_features(
    features: dict[str, np.ndarray | int | dict[str, float | tuple[float, float]]],
    out_dir: str | Path = "results/per_sim_final",
    svd_rank: int = SVD_RANK,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    idx = int(features["sim_index"])
    modes = np.asarray(features["modes"])
    eigs = np.asarray(features["eigs"])
    omega = np.asarray(features["omega"])
    norm_params = features["norm_params"]
    assert isinstance(norm_params, dict)

    path = out_path / f"koopman_pidmd_r{svd_rank}_sim{idx:05d}.npz"
    np.savez(
        path,
        A=np.asarray(features["A"], dtype=np.float64),
        modes_re=modes.real,
        modes_im=modes.imag,
        eigs_re=eigs.real,
        eigs_im=eigs.imag,
        omega_re=omega.real,
        omega_im=omega.imag,
        norm_min=np.array([float(norm_params["min"])]),
        norm_max=np.array([float(norm_params["max"])]),
        pts=np.asarray(features["pts"], dtype=np.float64),
    )
    return path


def iter_indices(args: argparse.Namespace, total: int) -> Iterable[int]:
    if args.sim_index:
        for index in args.sim_index:
            if index < 0 or index >= total:
                raise IndexError(f"sim_index fuori range: {index}")
            yield index
        return

    end = total if args.num_sims is None else min(total, args.restart + args.num_sims)
    for index in range(args.restart, end):
        yield index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estrae gli operatori Koopman-piDMD usati dal notebook Copia 05."
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default="results/per_sim_final")
    parser.add_argument("--svd-rank", type=int, default=SVD_RANK)
    parser.add_argument("--restart", type=int, default=0)
    parser.add_argument("--num-sims", type=int)
    parser.add_argument("--sim-index", type=int, action="append")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_iterator = DynabenchSimulationIterator(
        split=args.split,
        equation="advection",
        structure="cloud",
        resolution="low",
        base_path=args.data_root,
        download=args.download,
    )

    print(f"Simulazioni {args.split}: {len(sim_iterator)}")
    print(f"Output: {args.out_dir}")
    print(f"svd_rank: {args.svd_rank}")

    errors: list[tuple[int, str]] = []
    for index in iter_indices(args, len(sim_iterator)):
        output_path = Path(args.out_dir) / f"koopman_pidmd_r{args.svd_rank}_sim{index:05d}.npz"
        if output_path.exists() and not args.force:
            print(f"SKIP sim {index:05d}: {output_path} esiste gia'")
            continue

        try:
            features = extract_koopman_features(index, sim_iterator, svd_rank=args.svd_rank)
            saved_path = save_koopman_features(features, args.out_dir, svd_rank=args.svd_rank)
            print(f"Salvato: {saved_path}")
        except Exception as exc:
            errors.append((index, str(exc)))
            print(f"ERRORE sim {index:05d}: {exc}")

    if errors:
        print("Errori:")
        for index, message in errors:
            print(f"  sim {index:05d}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
