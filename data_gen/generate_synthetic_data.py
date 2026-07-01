"""Generate (input voltages, output intensity) pairs for a synthetic 4-input
3-class dataset that is NOT linearly separable, so the reservoir's nonlinearity
has a chance to add value (unlike iris/wine, where raw features were already
linearly separable).

Labelling: nested 4-d radius rings. Sample x_i ~ N(0, 1) i.i.d., compute
    r = sqrt(x1^2 + x2^2 + x3^2 + x4^2)
and bin r into 3 classes via tertiles of the empirical distribution. A linear
classifier on raw X cannot separate concentric shells in 4-d (the optimal
boundary is quadratic); a kernel SVM with rbf/quadratic kernel hits ~100%.

Same pipeline as generate_{iris,wine}_data.py: per-feature min-max to [0,1],
linearly to [v_min, v_max], y-symmetric voltages, run reservoir, collect I(y).
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
OUT_FOLDER = HERE / "data" / "NN_data" / "5_electrodes_synthetic"
N_FEATURES = 4
N_CLASSES = 3


def make_synthetic(n_samples: int, seed: int = 42
                   ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sample x_i ~ N(0,1) i.i.d., label by 4-d radius tertile."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, N_FEATURES)).astype(np.float64)
    r = np.linalg.norm(X, axis=1)
    edges = np.quantile(r, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.zeros(n_samples, dtype=np.int64)
    labels[r >= edges[0]] = 1
    labels[r >= edges[1]] = 2
    label_names = [f"r<{edges[0]:.2f}",
                   f"{edges[0]:.2f}<=r<{edges[1]:.2f}",
                   f"r>={edges[1]:.2f}"]
    return X, labels, label_names


def normalize_and_map_to_voltage(X: np.ndarray, v_min: float = -10.0,
                                  v_max: float = 10.0
                                  ) -> tuple[np.ndarray, np.ndarray]:
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
        raise RuntimeError(f"output_sensor.npz missing: {result.stdout[-800:]}")
    s = np.load(sensor_npz)
    Ex = s["Ex"][0] if s["Ex"].ndim == 2 else s["Ex"]
    Ey = s["Ey"][0] if s["Ey"].ndim == 2 else s["Ey"]
    return np.abs(Ex) ** 2 + np.abs(Ey) ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", type=str, default=str(DESIGN_FOLDER))
    ap.add_argument("--out", type=str, default=str(OUT_FOLDER))
    ap.add_argument("--n-samples", type=int, default=150,
                    help="total dataset size (default 150 ~ 50 per class)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--v-min", type=float, default=-10.0)
    ap.add_argument("--v-max", type=float, default=10.0)
    args = ap.parse_args()

    template = Path(args.template).resolve()
    out_folder = Path(args.out).resolve()
    out_folder.mkdir(parents=True, exist_ok=True)
    work_folder = out_folder / "_work_design"

    print(f"=== Generating synthetic dataset (nested 4-d radius rings) ===")
    X, labels, label_names = make_synthetic(args.n_samples, seed=args.seed)
    counts = dict(zip(*np.unique(labels, return_counts=True)))
    print(f"  X shape: {X.shape}, classes: {counts}")
    print(f"  label bins: {label_names}")
    print(f"  X range per feature: min {X.min(axis=0).round(3)}")
    print(f"                       max {X.max(axis=0).round(3)}")

    X_norm, V = normalize_and_map_to_voltage(X, args.v_min, args.v_max)
    print(f"  voltage range per feature: min {V.min(axis=0).round(2)}")
    print(f"                             max {V.max(axis=0).round(2)}")

    print(f"\n=== Preparing design folder: {work_folder} ===")
    prepare_design_folder(template, work_folder, n_electrodes=N_FEATURES)
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
            raise RuntimeError(f"sample {i} sensor len {I_out.size} vs {n_y}")
        I_outs.append(I_out)
        elapsed = time.time() - t0
        eta = (X.shape[0] - i - 1) * elapsed / (i + 1)
        print(f"  sample {i:3d}/{X.shape[0]}  label={int(labels[i])}  "
              f"peak={I_out.max():.3f}  ({time.time()-ts:.1f}s, ETA {eta/60:.1f}min)")

    I_out_arr = np.stack(I_outs, axis=0)

    assert n_y is not None
    with open(work_folder / "simulation_data.json") as f:
        d = json.load(f)
    sy = float(d["output_sensor"]["position"]["size"])
    y_axis = np.linspace(-sy / 2.0, sy / 2.0, n_y)

    out_path = out_folder / "synthetic_dataset.npz"
    np.savez(out_path,
             X=X, X_norm=X_norm, V=V,
             I_out=I_out_arr, y=y_axis,
             labels=labels, label_names=np.array(label_names))
    print(f"\n=== DONE ===")
    print(f"  total wall time: {(time.time()-t0)/60:.1f} min")
    print(f"  saved {out_path}  ({I_out_arr.shape} I_out array)")


if __name__ == "__main__":
    main()
