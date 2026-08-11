"""XOR readout task on 01 block+gain vs 05 cavity, from the existing IPC probe sets.

  python plotting/xor_01_vs_05.py
  python plotting/xor_01_vs_05.py --drive 100 --lam 1e-3

No new FDTD: XOR is a derived task. The IPC set already drives the reservoir with
1000 distinct amplitude vectors u in [-1,1]^4, so a label is built from the signs of
those amplitudes and a ridge readout is fitted to the recorded state.

WHY THIS TASK IS THE RIGHT TEST. A linear reservoir cannot do XOR at all: the label
is not an affine function of u, so any readout of an affine state is at chance. XOR
accuracy therefore measures usable NONLINEAR capacity directly, which is exactly the
quantity that kernel rank and KR - GR fail to report (both are participation
dimensions of the raw state, and the raw state's variance is dominated by its linear
part -- see plot_kr_gr_spectra.py). Two controls make that concrete and are plotted
alongside: the same ridge fitted to the 4 raw drive amplitudes, and to the state's
affine part alone. Both must sit at chance; if they do not, the task has leaked.

XOR2 uses the signs of the first two drives, XOR4 the parity of all four -- the
harder task, since it needs a fourth-order interaction rather than a second-order one.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUNS = Path("/home/ziga/Lips_project/reservoir_runs")
DEFAULT_PATHS = [RUNS / "reservoir_types" / "block_iso_gain" / "01", RUNS / "05_cav_4src"]
DEFAULT_LABELS = ["01 block + gain", "05 cavity"]
COLORS = ["#0a6ebd", "#d1701a"]
N_LIST = [10, 25, 50, 100, 200, 400, 600]
N_DRAWS = 12          # random detector subsets per N; the spread over them is the sd
TRAIN_FRAC = 0.7
SEED = 0


def features(out):
    out = np.asarray(out)
    return np.concatenate([out.real, out.imag], axis=1)


def ridge_accuracy(Xtr, ytr, Xte, yte, lam):
    """Ridge on +-1 labels, sign at test. Returns (test acc, train acc, d_eff).

    Features are standardised on TRAIN only -- standardising on the pooled set would
    leak test statistics into the fit and inflate the accuracy.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    M, F = Xtr.shape

    # Solve in whichever space is smaller; identical solution, no F x F blow-up.
    if F <= M:
        A = Xtr.T @ Xtr + lam * M * np.eye(F)
        w = np.linalg.solve(A, Xtr.T @ ytr)
    else:
        K = Xtr @ Xtr.T + lam * M * np.eye(M)
        w = Xtr.T @ np.linalg.solve(K, ytr)
    b = ytr.mean() - (Xtr @ w).mean()

    s = np.linalg.svd(Xtr, compute_uv=False)
    d_eff = float((s ** 2 / (s ** 2 + lam * M)).sum())
    acc = lambda X, y: float((np.sign(X @ w + b) == y).mean())
    return acc(Xte, yte), acc(Xtr, ytr), d_eff


def labels(U, n_bits):
    """XOR of the signs of the first n_bits drive amplitudes, as +-1."""
    bits = (U[:, :n_bits] > 0).astype(int)
    return np.where(bits.sum(1) % 2 == 0, 1.0, -1.0)


def evaluate(X, y, rng, lam, n_list, n_draws):
    """Mean/sd test accuracy vs detector count, over random detector subsets."""
    M = len(y)
    n_tr = int(TRAIN_FRAC * M)
    n_det = X.shape[1] // 2          # [Re|Im] stacking: half the columns are detectors
    out = []
    for N in n_list:
        if N > n_det:
            continue
        accs, trs, deffs = [], [], []
        for _ in range(n_draws):
            perm = rng.permutation(M)
            tr, te = perm[:n_tr], perm[n_tr:]
            det = rng.choice(n_det, size=N, replace=False)
            cols = np.concatenate([det, det + n_det])
            a, t, d = ridge_accuracy(X[np.ix_(tr, cols)], y[tr],
                                     X[np.ix_(te, cols)], y[te], lam)
            accs.append(a); trs.append(t); deffs.append(d)
        out.append({"N": N, "test": float(np.mean(accs)), "sd": float(np.std(accs)),
                    "train": float(np.mean(trs)), "d_eff": float(np.mean(deffs)),
                    "gap": float(np.mean(trs) - np.mean(accs))})
    return out


def control_accuracy(U, X, y, rng, lam):
    """Chance-level controls: the drive itself, and the state's affine part alone."""
    M = len(y)
    n_tr = int(TRAIN_FRAC * M)
    A = np.concatenate([U, np.ones((M, 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A[:n_tr], X[:n_tr], rcond=None)
    lin = A @ coef                      # state with all nonlinearity removed
    res = []
    for name, F in (("drive u", U), ("affine part of state", lin)):
        accs = []
        for _ in range(N_DRAWS):
            perm = rng.permutation(M)
            tr, te = perm[:n_tr], perm[n_tr:]
            a, _, _ = ridge_accuracy(F[tr], y[tr], F[te], y[te], lam)
            accs.append(a)
        res.append({"control": name, "test": float(np.mean(accs)),
                    "sd": float(np.std(accs))})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", nargs="+", default=[str(p) for p in DEFAULT_PATHS])
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--drive", type=int, default=100)
    ap.add_argument("--lam", type=float, default=1e-3, help="ridge strength (default 1e-3)")
    ap.add_argument("--out", default=str(REPO / "data" / "reservoir_types" / "xor_01_vs_05.png"))
    a = ap.parse_args()

    tasks = [("XOR2", 2), ("XOR4", 4)]
    records, controls = [], []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    for path, label, colour in zip(a.paths, a.labels, COLORS):
        ds = Path(path) / "datasets" / f"ipc_4src_a{a.drive}.npz"
        with np.load(ds, allow_pickle=True) as d:
            U = np.asarray(d["inputs"]); X = features(np.asarray(d["outputs"]))
        for (task, bits), ax in zip(tasks, axes):
            y = labels(U, bits)
            rng = np.random.default_rng(SEED)
            rows = evaluate(X, y, rng, a.lam, N_LIST, N_DRAWS)
            ax.errorbar([r["N"] for r in rows], [r["test"] for r in rows],
                        yerr=[r["sd"] for r in rows], fmt="o-", ms=4, lw=1.6,
                        capsize=3, color=colour, label=label)
            for r in rows:
                records.append({"design": label, "task": task, **r})
            for c in control_accuracy(U, X, y, np.random.default_rng(SEED), a.lam):
                controls.append({"design": label, "task": task, **c})

    for (task, bits), ax in zip(tasks, axes):
        ax.axhline(0.5, color="0.45", ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("detectors used")
        ax.set_title(f"{task}  (parity of {bits} drive signs)", fontsize=10)
        ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
        # Opaque legend: the chance line runs behind it and otherwise strikes
        # through the labels.
        ax.legend(fontsize=8.5, loc="lower right", frameon=True,
                  facecolor="white", edgecolor="none", framealpha=1.0)
    axes[0].set_ylabel("test accuracy")
    axes[0].annotate("chance", xy=(10, 0.5), xytext=(11, 0.515), fontsize=8, color="0.35")
    fig.suptitle(f"XOR readout at drive a={a.drive}, ridge $\\lambda$={a.lam:g}", fontsize=11)
    fig.tight_layout()

    out = Path(a.out)
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({"results": records, "controls": controls,
                   "drive": a.drive, "lam": a.lam}, f, indent=1)

    print(f"\n{'design':<18}{'task':>6}{'N':>6}{'test':>9}{'sd':>8}{'train':>8}{'gap':>8}{'d_eff':>9}")
    for r in records:
        print(f"{r['design']:<18}{r['task']:>6}{r['N']:6d}{r['test']:9.3f}{r['sd']:8.3f}"
              f"{r['train']:8.3f}{r['gap']:8.3f}{r['d_eff']:9.1f}")
    print("\ncontrols (must sit at chance = 0.5):")
    for c in controls:
        print(f"  {c['design']:<18}{c['task']:>6}  {c['control']:<22}"
              f"{c['test']:.3f} +- {c['sd']:.3f}")
    print(f"\nwrote {out}, .pdf and .json")


if __name__ == "__main__":
    raise SystemExit(main())
