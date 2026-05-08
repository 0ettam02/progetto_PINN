#!/usr/bin/env python3
"""
Script semplice per plottare snapshot della soluzione.

Modalità d'uso principali:
 - Mostra uno snapshot dalla prima traiettoria Dynabench locale:
     python plot_solution.py --index 10

 - Carica da file HDF5 / NumPy e plotta un frame (index o tempo):
     python plot_solution.py --input data/advection/cloud/low/advection_test_cloud_low_0_499.h5 --index 5 --out fig.png

Lo script gestisce sia matrici 2D/3D sia file Dynabench cloud con dataset
`data` e `points`.
"""

from __future__ import annotations

import argparse
import sys
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_from_h5(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Prova a leggere da un file HDF5 e ritorna (times, snapshots).
    snapshots shape: (nt, nx, ny) o (nt, ny, nx) - lo script cerca di normalizzare a (nt, nx, ny).
    Se il file non contiene tempi, times sarà np.arange(nt).
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py non installato. installalo con: pip install h5py") from exc

    with h5py.File(path, "r") as f:
        if "data" in f and "points" in f:
            data = np.asarray(f["data"][0])
            points = np.asarray(f["points"][0])
            data = np.squeeze(data)
            if data.ndim != 2:
                raise RuntimeError(
                    "Dataset Dynabench cloud con dimensione non supportata: "
                    + str(data.shape)
                )
            return np.arange(data.shape[0]), data, points

        # Cerca il primo dataset utile
        dataset = None
        for name, obj in f.items():
            if isinstance(obj, h5py.Dataset):
                data = obj[()]
                if data.ndim >= 2:
                    dataset = data
                    break
        if dataset is None:
            # cerca ricorsivamente
            def find_dataset(g):
                for k, v in g.items():
                    if isinstance(v, h5py.Dataset):
                        d = v[()]
                        if d.ndim >= 2:
                            return d
                    elif isinstance(v, h5py.Group):
                        r = find_dataset(v)
                        if r is not None:
                            return r
                return None

            dataset = find_dataset(f)

        if dataset is None:
            raise RuntimeError(f"Nessun dataset 2D/3D trovato in {path}")

        data = np.asarray(dataset)

        # Prova a trovare un vettore tempi (dataset chiamato 'time' o 'tempi')
        times = None
        for key in ("time", "times", "tempi", "t"):
            if key in f:
                times = np.asarray(f[key])
                break

        if data.ndim == 3:
            nt = data.shape[0]
            if times is None:
                times = np.arange(nt)
            return times, data, None
        elif data.ndim == 2:
            # singolo snapshot
            times = np.array([0.0])
            return times, data[np.newaxis, ...], None
        else:
            # cerca di ridurre a 3D
            data = np.squeeze(data)
            if data.ndim == 3:
                nt = data.shape[0]
                times = np.arange(nt) if times is None else times
                return times, data, None
            raise RuntimeError("Dataset con dimensione non supportata: "+str(data.shape))


def load_from_numpy(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = np.load(path)
    # .npz -> dict-like
    if isinstance(data, np.lib.npyio.NpzFile):
        # cerca chiavi utili
        for key in ("snapshot", "snapshots", "data", "arr_0"):
            if key in data:
                arr = data[key]
                break
        else:
            # prendi il primo elemento
            arr = data[list(data.files)[0]]
    else:
        arr = data

    arr = np.asarray(arr)
    if arr.ndim == 3:
        nt = arr.shape[0]
        times = np.arange(nt)
        return times, arr, None
    elif arr.ndim == 2:
        return np.array([0.0]), arr[np.newaxis, ...], None
    else:
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            return np.arange(arr.shape[0]), arr, None
        raise RuntimeError("Array numpy con dimensione non supportata: "+str(arr.shape))


def plot_snapshot(snapshot: np.ndarray, time: float | None = None, out: str | None = None, cmap: str = "viridis") -> None:
    # snapshot expected shape (nx, ny)
    if snapshot.ndim != 2:
        raise ValueError("snapshot deve essere 2D")

    nx, ny = snapshot.shape

    plt.figure(figsize=(6, 5))
    # imshow expects (ny, nx) so trasponiamo
    im = plt.imshow(snapshot.T, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, aspect="auto")
    plt.colorbar(im, label="u")
    title = "snapshot"
    if time is not None:
        title += f"  t={time:.4g}"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    if out:
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Figura salvata in {out}")
    else:
        plt.show()


def plot_snapshot_cloud(
    valori: np.ndarray,
    points: np.ndarray,
    time: float | None = None,
    out: str | None = None,
    cmap: str = "viridis",
) -> None:
    valori = np.squeeze(valori)
    if valori.ndim != 1:
        raise ValueError("per il cloud Dynabench lo snapshot deve essere 1D")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points deve avere shape (n_punti, 2)")

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(points[:, 0], points[:, 1], c=valori, s=16, cmap=cmap)
    plt.colorbar(sc, label="u")
    title = "snapshot"
    if time is not None:
        title += f"  t={time:.4g}"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")

    if out:
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Figura salvata in {out}")
    else:
        plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot snapshot solution")
    parser.add_argument("--input", "-i", help="File di input (.h5, .npy, .npz). Se omesso usa la prima traiettoria Dynabench via esperimento.")
    parser.add_argument("--index", "-n", type=int, help="Indice snapshot (0-based)")
    parser.add_argument("--time", "-t", type=float, help="Tempo reale: trova lo snapshot più vicino")
    parser.add_argument("--out", "-o", help="Percorso file per salvare la figura (png, pdf, ...)")
    parser.add_argument("--cmap", default="viridis", help="Colormap matplotlib")

    args = parser.parse_args(argv)

    times = None
    snapshots = None
    points = None

    if args.input:
        path = args.input
        if path.endswith(".h5") or path.endswith(".hdf5"):
            times, snapshots, points = load_from_h5(path)
        elif path.endswith(".npy") or path.endswith(".npz"):
            times, snapshots, points = load_from_numpy(path)
        else:
            # prova entrambi
            try:
                times, snapshots, points = load_from_h5(path)
            except Exception:
                times, snapshots, points = load_from_numpy(path)
    else:
        # usa la prima traiettoria Dynabench locale
        try:
            from esperimento import crea_esperimento

            exp = crea_esperimento(create_model=False)
            times, snapshots, points = exp["dataset"].prima_traiettoria()
            exp["dataset"].close()
        except Exception as exc:
            print("Impossibile caricare gli snapshot Dynabench: ", exc)
            sys.exit(1)

    times = np.asarray(times)
    snapshots = np.asarray(snapshots)

    if points is not None and snapshots.ndim == 2:
        nt = snapshots.shape[0]
    elif snapshots.ndim == 3:
        nt = snapshots.shape[0]
    elif snapshots.ndim == 2:
        snapshots = snapshots[np.newaxis, ...]
        nt = 1
    else:
        raise RuntimeError("snapshot array con dimensione non supportata: "+str(snapshots.shape))

    if args.index is not None:
        idx = args.index
        if idx < 0 or idx >= nt:
            raise IndexError("index fuori range")
    elif args.time is not None:
        idx = int(np.argmin(np.abs(times - args.time)))
    else:
        idx = 0

    chosen_time = float(times[idx]) if times is not None and len(times) > idx else None
    snapshot = snapshots[idx]

    if points is not None:
        plot_snapshot_cloud(snapshot, points, time=chosen_time, out=args.out, cmap=args.cmap)
    else:
        plot_snapshot(snapshot, time=chosen_time, out=args.out, cmap=args.cmap)


if __name__ == "__main__":
    main()
