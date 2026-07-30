#!/usr/bin/env python
"""Random-sigmoid reference "reservoir": 4 inputs → 200 sigmoid units, fixed
random weights (ELM-style). Runs the SAME two probes as the physical devices:

1. Harmonics (Method D): the generate_harmonics_data phase-sweep drive
   E(t_j) = Σₖ Aₖ cos(toneₖ·t_j)·eₖ on channels 0,1 (tones 3,5, N_t=64),
   outputs → n4_harmonics_distortion.harmonic_specter.
2. Balance-scale: the dataset's raw 4 inputs → NN → 200 features → the same
   softmax-ridge readout protocol as scripts/train_balance_scale_readout.py
   (70/30 split, 5 repeats, detector-count sweep by subsampling outputs).

Inputs are standardized before the layer; --gain sets how deep into the
sigmoid's nonlinear regime the layer operates (h = σ(gain·(Wx+b)), W,b ~ N(0,1),
W scaled by 1/√4).

    python characterization/random_sigmoid_reference.py \
        --balance /home/ziga/Lips_project/reservoir_runs/04_LC_4src/datasets/balance_scale.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))
from n4_harmonics_distortion import harmonic_specter          # noqa: E402
from train_balance_scale_readout import softmax_ridge         # noqa: E402


def make_nn(n_in=4, n_out=200, gain=1.5, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_out, n_in)) / np.sqrt(n_in)
    b = rng.normal(size=n_out)
    def nn(x):
        return 1.0 / (1.0 + np.exp(-gain * (x @ W.T + b)))
    return nn


def harmonics_probe(nn, tones=(3, 5), chans=(0, 1), n_t=64, amp=1.0):
    t = 2.0 * np.pi * np.arange(n_t) / n_t
    U = np.zeros((len(tones), 4))
    for k, s in enumerate(chans):
        U[k, s] = 1.0
    X = np.zeros((n_t, 4))
    for k, tone in enumerate(tones):
        X += amp * np.cos(tone * t)[:, None] * U[k]           # Re() as the source does
    Y = nn(X)
    return {"outputs": Y.astype(complex), "inputs": X.astype(complex),
            "t": t, "tones": np.asarray(tones), "amps": np.full(len(tones), amp)}


def balance_probe(nn, npz, n_points=(10, 25, 50, 100, 200), repeats=5,
                  test_frac=0.3, seed=0):
    z = np.load(npz, allow_pickle=True)
    X_raw, y = z["inputs"].astype(float), z["labels"].astype(int)
    X_std = (X_raw - X_raw.mean(0)) / X_raw.std(0)
    H = nn(X_std)                                             # (625, 200)
    rng = np.random.default_rng(seed)
    rows = []
    for N in n_points:
        accs = []
        for _ in range(repeats):
            cols = rng.choice(H.shape[1], size=min(N, H.shape[1]), replace=False)
            idx = rng.permutation(len(y))
            n_te = int(test_frac * len(y))
            te, tr = idx[:n_te], idx[n_te:]
            _, acc = softmax_ridge(H[tr][:, cols], y[tr], H[te][:, cols], y[te])
            accs.append(acc)
        rows.append((N, float(np.mean(accs)), float(np.std(accs))))
        print(f"  N={N:4d} sigmoid units: test {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--balance", required=True, help="balance_scale.npz path")
    ap.add_argument("--gain", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_t", type=int, default=64)
    a = ap.parse_args()

    nn = make_nn(gain=a.gain, seed=a.seed)

    print(f"[random-sigmoid 4->200, gain={a.gain}, seed={a.seed}]")
    print("== harmonics probe (tones 3,5 on ch 0,1) ==")
    spec = harmonic_specter(harmonics_probe(nn, n_t=a.n_t))
    for k, v in spec.items():
        if np.isscalar(v) or isinstance(v, (int, float)):
            print(f"  {k}: {v}")

    print("== balance-scale readout (same protocol as physical) ==")
    balance_probe(nn, a.balance)
    print("  reference: raw-4 logistic ~0.89 | MLP ~0.96 | "
          "LC peak 0.937 | iso peak 0.927")


if __name__ == "__main__":
    main()
