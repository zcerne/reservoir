"""XOR task accuracy vs readout size, for one encoding, both readouts.

  python plotting/xor_encoding_02.py --npz <readouts npz> --label "intensity encoding"

Input is an {inputs, outputs} npz (the IPC probe set, or just its readout columns
pulled from the parts). Labels are the parity of the signs of the drive variables u,
so XOR2 uses two channels and XOR4 all four.

WHAT THE SIGN MEANS DEPENDS ON THE ENCODING, and it is worth being explicit. Under
amplitude encoding u < 0 is a pi phase flip, and XOR is then an EVEN function of u —
which is why an intensity readout (also even) could do it while a field readout had to
lean on whatever small even component the medium had. Under intensity encoding the
drive is u -> sqrt((u+1)/2), monotonic and non-negative, so the sign is merely
"intensity above or below half-max" and no parity structure exists: both readouts sit
on equal footing, and the task instead demands a non-monotonic response in each input.

Channels are PCA components of the readout, not raw detectors: with 61 wavelengths x
200 points x 3 components the raw feature count (73,200 real) vastly exceeds the 1000
probes, so a detector subset would measure the subset, not the reservoir.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
from xor_01_vs_05 import ridge_accuracy, labels     # noqa: E402

COL = {"field": "#2a72c4", "intensity": "#e08a1e"}
MARK = {"field": "o", "intensity": "s"}
KS = [10, 25, 50, 100, 200, 400]
N_SPLIT = 8
TRAIN = 0.7


def pca(F, k):
    Fc = F - F.mean(0)
    _, _, Vt = np.linalg.svd(Fc, full_matrices=False)
    return Fc @ Vt[:k].T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--label", default="intensity encoding")
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--out", default=str(REPO / "data" / "reservoir_types"
                                         / "xor_encoding_02.png"))
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    U, X = np.asarray(d["inputs"]), np.asarray(d["outputs"])
    reps = {"field": np.concatenate([X.real, X.imag], axis=1),
            "intensity": np.abs(X) ** 2}
    M = len(U); n_tr = int(TRAIN * M)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    rows = []
    for ax, (task, bits) in zip(axes, (("XOR2", 2), ("XOR4", 4))):
        y = labels(U, bits)
        for mode, F in reps.items():
            mu, sd = [], []
            for k in KS:
                Z = pca(F, min(k, F.shape[1]))
                rng = np.random.default_rng(0)
                acc = []
                for _ in range(N_SPLIT):
                    pm = rng.permutation(M); tr, te = pm[:n_tr], pm[n_tr:]
                    acc.append(ridge_accuracy(Z[tr], y[tr], Z[te], y[te], a.lam)[0])
                mu.append(np.mean(acc)); sd.append(np.std(acc))
                rows.append((task, mode, k, np.mean(acc), np.std(acc)))
            ax.errorbar(KS, mu, yerr=sd, fmt=MARK[mode] + "-", ms=5, lw=1.7, capsize=3,
                        color=COL[mode], label=f"{mode} readout")
        ax.axhline(0.5, color="0.45", ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("PCA channels used")
        ax.set_title(f"{task}  (parity of {bits} drive signs)", fontsize=10)
        ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
        ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    axes[0].set_ylabel("test accuracy")
    axes[0].annotate("chance", xy=(10, 0.5), xytext=(10.5, 0.515), fontsize=8, color="0.35")

    fig.suptitle(f"XOR on block_iso_gain/02 — {a.label}", fontsize=11)
    fig.tight_layout()
    out = Path(a.out)
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"{'task':>6}{'readout':>11}{'k':>6}{'test':>9}{'sd':>8}")
    for t, m, k, u_, s in rows:
        print(f"{t:>6}{m:>11}{k:6d}{u_:9.3f}{s:8.3f}")
    print(f"\nwrote {out} and .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
