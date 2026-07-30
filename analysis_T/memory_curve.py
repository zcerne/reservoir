#!/usr/bin/env python
"""Memory curve m(d) from the symbol-drive runs (mem_* designs).

For each condition (p150 cavity / p0 cavity / mirrorless), pools the
per-symbol feature vectors across seeds, ridge-regresses them against the
input symbol d steps back, and reports m(d) = corr^2 on a held-out test
split. The input sequence is rebuilt exactly via symbols_source.symbol_sequence
(same seed, same n_sym as the run used) — nothing about u(n) is stored on disk.

    python analysis_T/memory_curve.py \
        --base data/memory_testing --out memory_curve

Feature vector per symbol window: mean of the |Ey|^2 envelope plus `--taps`
sub-window means (envelope = Ey^2 smoothed over ~2 optical periods).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from symbols_source import symbol_sequence  # noqa: E402

CONDITIONS = {          # label -> design-dir glob
    "cavity p150": "mem_R0.5_p150_s*",
    "cavity p0": "mem_R0.5_p0_s*",
    "mirrorless p150": "mem_Rnone_p150_s*",
}


def envelope(ey: np.ndarray, dt: float, periods: float = 2.0,
             lam: float = 0.55) -> np.ndarray:
    """|Ey|^2 smoothed over `periods` optical periods (period = lam t.u.)."""
    w = max(3, int(round(periods * lam / dt)))
    k = np.ones(w) / w
    return np.convolve(ey ** 2, k, mode="same")


def run_features(folder: str, taps: int, washout: int):
    """(features [n_windows, taps+1], targets u [n_windows]) for one run."""
    cfg = json.load(open(os.path.join(folder, "simulation_data.json")))
    src = next(v for v in cfg.values()
               if isinstance(v, dict) and v.get("source_type") == "symbols")
    T = float(src["symbol_length"])
    end_time = float(src.get("end_time", cfg["run_until"]))
    n_sym = int(np.ceil(end_time / T)) + 1
    u = symbol_sequence(src.get("seed", 0), n_sym,
                        src.get("amp_range", [0.5, 1.5]))

    z = np.load(os.path.join(folder, "simulation_meep", "point_snap.npz"))
    t, ey = z["t"], z["Ey"][:, 0]
    dt = float(np.median(np.diff(t)))
    env = envelope(ey, dt)

    feats, targs = [], []
    n_windows = int(t[-1] // T)
    for n in range(washout, n_windows):
        sel = (t >= n * T) & (t < (n + 1) * T)
        e = env[sel]
        if e.size < taps:
            continue
        sub = [s.mean() for s in np.array_split(e, taps)]
        feats.append([e.mean()] + sub)
        targs.append(n)          # symbol index; u looked up per delay later
    return np.asarray(feats), np.asarray(targs, dtype=int), u


def ridge_m(X, y, alpha, train_frac=0.7, seed=0):
    """Test-set corr^2 of ridge prediction (features standardized on train)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_tr = int(train_frac * len(y))
    tr, te = idx[:n_tr], idx[n_tr:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    Xtr = np.c_[np.ones(len(tr)), (X[tr] - mu) / sd]
    Xte = np.c_[np.ones(len(te)), (X[te] - mu) / sd]
    A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ y[tr])
    p = Xte @ w
    if p.std() < 1e-12 or y[te].std() < 1e-12:
        return 0.0
    return float(np.corrcoef(p, y[te])[0, 1] ** 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="data/memory_testing")
    ap.add_argument("--taps", type=int, default=8)
    ap.add_argument("--washout", type=int, default=3,
                    help="symbol windows discarded at the start of each run")
    ap.add_argument("--max-delay", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--out", default="memory_curve",
                    help="basename for <out>.png/<out>.json in --base")
    a = ap.parse_args()

    results = {}
    for label, pat in CONDITIONS.items():
        folders = sorted(glob.glob(os.path.join(a.base, pat)))
        pooled_X, pooled_u = [], {d: [] for d in range(a.max_delay + 1)}
        for f in folders:
            snap = os.path.join(f, "simulation_meep", "point_snap.npz")
            if not os.path.exists(snap):
                print(f"[skip] {f} (no point_snap yet)")
                continue
            X, widx, u = run_features(f, a.taps, a.washout)
            pooled_X.append(X)
            for d in range(a.max_delay + 1):
                pooled_u[d].append(u[widx - d])
        if not pooled_X:
            print(f"[{label}] no data")
            continue
        X = np.vstack(pooled_X)
        curve = [ridge_m(X, np.concatenate(pooled_u[d]), a.ridge)
                 for d in range(a.max_delay + 1)]
        results[label] = curve
        mc = sum(curve[1:])          # capacity excludes d=0 (present input)
        print(f"[{label}] n={len(X)} windows  m(d)="
              + " ".join(f"{v:.3f}" for v in curve) + f"  MC(d>=1)={mc:.2f}")

    with open(os.path.join(a.base, a.out + ".json"), "w") as f:
        json.dump(results, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, curve in results.items():
        ax.plot(range(len(curve)), curve, "o-", label=label)
    ax.set_xlabel("delay d [symbols = round trips]")
    ax.set_ylabel("m(d) = test corr²")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.set_title("Linear memory curve — R0.5 cavity reservoir")
    fig.tight_layout()
    fig.savefig(os.path.join(a.base, a.out + ".png"), dpi=150)
    print("saved", os.path.join(a.base, a.out + ".png"))


if __name__ == "__main__":
    main()
