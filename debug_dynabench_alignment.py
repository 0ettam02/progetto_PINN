from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from esperimento import ConfigurazioneEsperimento, DynabenchRolloutDataset


def normalizza(stati: np.ndarray, norm_min: float, norm_max: float) -> np.ndarray:
    return 2.0 * (stati - norm_min) / (norm_max - norm_min + 1e-8) - 1.0


def denormalizza(stati: np.ndarray, norm_min: float, norm_max: float) -> np.ndarray:
    return (stati + 1.0) * 0.5 * (norm_max - norm_min) + norm_min


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug allineamento Dynabench cloud/piDMD.")
    parser.add_argument("--sim-index", type=int, default=0)
    parser.add_argument("--temporal-index", type=int, default=0)
    parser.add_argument("--dmd-dir", default="results/per_sim_final")
    args = parser.parse_args()

    config = ConfigurazioneEsperimento(lookback=1, lambda_dmd=1.0)
    dataset = DynabenchRolloutDataset(config, split="train")
    usable = int(dataset.iterator.usable_simulation_lengths[0])
    iterator_index = args.sim_index * usable + args.temporal_index

    _, _, temporal_index, sim_index = dataset.decodifica_indice(iterator_index)
    item = dataset.iterator[iterator_index]

    dmd_path = Path(args.dmd_dir) / f"koopman_pidmd_r58_sim{sim_index:05d}.npz"
    dmd = np.load(dmd_path)
    A = dmd["A"].astype(np.float64)
    pts = dmd["pts"].astype(np.float64)
    norm_min = float(dmd["norm_min"][0])
    norm_max = float(dmd["norm_max"][0])

    item_points = item.pos.astype(np.float64)
    points_max_abs = float(np.max(np.abs(item_points - pts)))
    print(f"iterator_index        : {iterator_index}")
    print(f"sim_index decoded     : {sim_index}")
    print(f"temporal_index decoded: {temporal_index}")
    print(f"dmd file              : {dmd_path}")
    print(f"points max abs diff   : {points_max_abs:.6e}")

    u0 = item.x[-1, :, 0].astype(np.float64)
    y = item.y[:, :, 0].astype(np.float64)
    persistence = np.repeat(u0[None, :], y.shape[0], axis=0)
    print(f"persistence mse16     : {mse(persistence, y):.6e}")

    stati_norm = []
    stato = normalizza(u0, norm_min, norm_max)
    for _ in range(config.rollout):
        stato = A @ stato
        stati_norm.append(stato.copy())
    pred_norm = np.stack(stati_norm)
    pred = denormalizza(pred_norm, norm_min, norm_max)

    print(f"pidmd A mse16 physical: {mse(pred, y):.6e}")
    print(f"pidmd A mse16 norm    : {mse(pred_norm, normalizza(y, norm_min, norm_max)):.6e}")
    per_step = np.mean((pred - y) ** 2, axis=1)
    print("pidmd A mse per step  :", " ".join(f"{v:.3e}" for v in per_step))

    # Controllo equivalente con tensori e orientamento usato dalla training loss.
    A_t = torch.as_tensor(A, dtype=torch.float64)
    stato_t = torch.as_tensor(normalizza(u0, norm_min, norm_max), dtype=torch.float64)
    pred_t = []
    for _ in range(config.rollout):
        stato_t = stato_t @ A_t.T
        pred_t.append(stato_t)
    pred_t = torch.stack(pred_t).numpy()
    print(f"torch orientation diff: {float(np.max(np.abs(pred_t - pred_norm))):.6e}")


if __name__ == "__main__":
    main()
