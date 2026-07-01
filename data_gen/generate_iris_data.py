"""Generate (input voltages, output intensity) pairs for the iris dataset
by running the voltage_reservoir + FDTD pipeline on each sample.

Pipeline per iris sample:
  1. Normalize 4 iris features to [0, 1] (per-feature min-max over full set).
  2. Map each normalized value linearly to a voltage in [-10, +10] V.
  3. Set top + bottom electrodes identically (y-symmetric: y_min[k] = y_max[k]).
  4. Compute director via class_voltage_reservoir (Poisson + LC relax / shortcut).
  5. Write director to simulation/lc_fields.npz and run class_simulation_gpu FDTD.
  6. Accumulate I(y) sensor output.
Save single iris_dataset.npz with input/voltage/output triples.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np


HERE = Path(__file__).parent.resolve()
DESIGN_FOLDER = HERE / "data" / "1_2D_electrodes"
OUT_FOLDER = HERE / "data" / "NN_data" / "3_electrodes_iris"


def load_iris() -> tuple[np.ndarray, np.ndarray, list[str]]:
    from sklearn import datasets
    iris = datasets.load_iris()
    return (np.asarray(iris.data, dtype=np.float64),
            np.asarray(iris.target, dtype=np.int64),
            [str(s) for s in iris.target_names])


def normalize_and_map_to_voltage(X: np.ndarray, v_min: float = -10.0,
                                  v_max: float = 10.0
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature min-max normalize to [0,1], then linearly map to [v_min, v_max]."""
    x_min = X.min(axis=0); x_max = X.max(axis=0)
    rng = np.where(x_max > x_min, x_max - x_min, 1.0)
    X_norm = (X - x_min) / rng
    V = v_min + X_norm * (v_max - v_min)
    return X_norm, V


def prepare_design_folder(template_folder: Path, work_folder: Path,
                          n_electrodes: int) -> None:
    work_folder.mkdir(parents=True, exist_ok=True)
    with open(template_folder / "simulation_data.json") as f:
        d = json.load(f)
    d["reservoir"]["voltages_y_min"] = [0.0] * n_electrodes
    d["reservoir"]["voltages_y_max"] = [0.0] * n_electrodes
    d["reservoir"]["voltages_x_min"] = []
    d["reservoir"]["voltages_x_max"] = []
    with open(work_folder / "simulation_data.json", "w") as f:
        json.dump(d, f, indent=2)


def set_voltages_in_json(folder: Path, voltages: np.ndarray) -> None:
    vlist = [float(v) for v in voltages]
    p = folder / "simulation_data.json"
    with open(p) as f:
        d = json.load(f)
    d["reservoir"]["voltages_y_min"] = vlist
    d["reservoir"]["voltages_y_max"] = vlist
    with open(p, "w") as f:
        json.dump(d, f, indent=2)


def run_one_sample(work_folder: Path, voltages: np.ndarray) -> np.ndarray:
    set_voltages_in_json(work_folder, voltages)
    result = subprocess.run(
        [sys.executable, str(HERE / "run_voltage_reservoir.py"),
         "--path", str(work_folder)],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pipeline failed: {result.stderr[-2000:]}")
    sensor_npz = work_folder / "simulation" / "output_sensor.npz"
    if not sensor_npz.exists():
        raise RuntimeError(f"output_sensor.npz missing after run: {result.stdout[-800:]}")
    s = np.load(sensor_npz)
    Ex = s["Ex"][0] if s["Ex"].ndim == 2 else s["Ex"]
    Ey = s["Ey"][0] if s["Ey"].ndim == 2 else s["Ey"]
    return np.abs(Ex) ** 2 + np.abs(Ey) ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", type=str, default=str(DESIGN_FOLDER),
                    help="template design folder (JSON copied per-sample)")
    ap.add_argument("--out", type=str, default=str(OUT_FOLDER),
                    help="output folder for iris_dataset.npz")
    ap.add_argument("--n-samples", type=int, default=0,
                    help="cap on samples (0 = all 150). Useful for smoke tests.")
    ap.add_argument("--v-min", type=float, default=-10.0)
    ap.add_argument("--v-max", type=float, default=10.0)
    args = ap.parse_args()

    template = Path(args.template).resolve()
    out_folder = Path(args.out).resolve()
    out_folder.mkdir(parents=True, exist_ok=True)
    work_folder = out_folder / "_work_design"

    print(f"=== Loading iris ===")
    X, labels, label_names = load_iris()
    print(f"  X shape: {X.shape}, labels: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"  feature ranges: min {X.min(axis=0)}, max {X.max(axis=0)}")
    X_norm, V = normalize_and_map_to_voltage(X, args.v_min, args.v_max)
    print(f"  voltage range per feature: min {V.min(axis=0)}, max {V.max(axis=0)}")
    n_features = X.shape[1]

    if args.n_samples > 0:
        N = min(args.n_samples, X.shape[0])
        X, X_norm, V, labels = X[:N], X_norm[:N], V[:N], labels[:N]

    print(f"\n=== Preparing design folder: {work_folder} ===")
    prepare_design_folder(template, work_folder, n_electrodes=n_features)
    (work_folder / "simulation").mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running {X.shape[0]} samples ===")
    I_outs: list[np.ndarray] = []
    n_y: int | None = None
    t0 = time.time()
    for i in range(X.shape[0]):
        ts = time.time()
        I_out = run_one_sample(work_folder, V[i])
        if n_y is None:
            n_y = I_out.size
        elif I_out.size != n_y:
            raise RuntimeError(f"sample {i} has mismatched sensor length {I_out.size} vs {n_y}")
        I_outs.append(I_out)
        elapsed = time.time() - t0
        eta = (X.shape[0] - i - 1) * elapsed / (i + 1)
        print(f"  sample {i:3d}/{X.shape[0]}  label={int(labels[i])}  "
              f"V={V[i].round(2).tolist()}  peak={I_out.max():.3f}  "
              f"({time.time()-ts:.1f}s, ETA {eta/60:.1f}min)")

    I_out_arr = np.stack(I_outs, axis=0)

    assert n_y is not None
    with open(work_folder / "simulation_data.json") as f:
        d = json.load(f)
    sy = float(d["output_sensor"]["position"]["size"])
    y_axis = np.linspace(-sy / 2.0, sy / 2.0, n_y)

    out_path = out_folder / "iris_dataset.npz"
    np.savez(out_path,
             X=X, X_norm=X_norm, V=V,
             I_out=I_out_arr, y=y_axis,
             labels=labels, label_names=np.array(label_names))
    print(f"\n=== DONE ===")
    print(f"  total wall time: {(time.time()-t0)/60:.1f} min")
    print(f"  saved {out_path}  ({I_out_arr.shape} I_out array)")


if __name__ == "__main__":
    main()
