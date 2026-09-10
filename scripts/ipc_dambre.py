"""Dambre information processing capacity (IPC) on the static IPC dataset,
compared across output interpretations of the same complex exit field:

    |E|           amplitude (phase-insensitive detector)
    |E|^2         intensity (detector square — even-heavy by construction)
    sign(phase)|E| signed amplitude (encode-symmetric decode, odd-faithful)
    Re/Im         strictly linear in the field (reference)

Static Dambre: inputs u ~ U(-1,1)^4 i.i.d., targets = normalized products of
Legendre polynomials prod_i P_{d_i}(u_i), total degree 1..MAX_DEG. Capacity of
one target = OUT-OF-SAMPLE R^2 of its linear reconstruction from the features
(NEVER in-sample: 252 features vs 1000 probes would fake it). Noise floor
calibrated by shuffled targets; capacities below the floor count as 0.
Total capacity is bounded by the feature rank.

    python scripts/ipc_dambre.py --npz <ipc.npz>
"""
import argparse
import itertools
import numpy as np
from numpy.polynomial import legendre as L

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="/home/ziga/Lips/resevoir/data/signal_modulation/05d_ampsweep/datasets/ipc.npz")
ap.add_argument("--max-deg", type=int, default=5)
ap.add_argument("--test", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fig", default=None, help="output figure path")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
d = np.load(args.npz)
U = np.asarray(d["inputs"])
out = np.asarray(d["outputs"])
freqs = np.asarray(d["m2_freqs"]); nlam = len(freqs)
M = len(U); ny = out.shape[1] // nlam
Ek = out.reshape(M, nlam, ny)[:, int(np.argmin(np.abs(freqs - 1 / 1.064))), :]

idx = rng.permutation(M)
tr, te = idx[:-args.test], idx[-args.test:]

# phase reference for the signed decode: intercept of the linear field fit
A = np.concatenate([np.ones((M, 1)), U], axis=1)
coef, *_ = np.linalg.lstsq(A[tr], Ek[tr], rcond=None)
ref = coef[0] / np.abs(coef[0])

FEATURES = {
    "|E|": np.abs(Ek),
    "|E|^2": np.abs(Ek) ** 2,
    "sign(phase)|E|": np.sign((Ek * np.conj(ref)).real) * np.abs(Ek),
    "Re/Im (linear)": np.concatenate([Ek.real, Ek.imag], axis=1),
}

# Legendre target bank: all multi-indices, total degree 1..max_deg
combos = [c for deg in range(1, args.max_deg + 1)
          for c in itertools.product(range(deg + 1), repeat=4) if sum(c) == deg]
def leg(n, x):
    return L.legval(x, [0] * n + [1])
Z = np.empty((M, len(combos)))
for j, c in enumerate(combos):
    z = np.ones(M)
    for i, n in enumerate(c):
        if n:
            z = z * leg(n, U[:, i])
    Z[:, j] = z
Z = (Z - Z[tr].mean(0)) / Z[tr].std(0)          # normalize targets on train
degs = np.array([sum(c) for c in combos])

LAMBDAS = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])

def _r2(pred, z):
    return 1 - ((z - pred) ** 2).sum(0) / ((z - z.mean(0)) ** 2).sum(0)

def capacities(X, Zt):
    """Ridge regression, lambda picked on a validation split of the train set
    (unregularized OLS with 252 ill-conditioned features overfits into
    uniformly negative OOS R^2 — regularization is part of the protocol)."""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    Xs = (X - mu) / sd
    fit_i, val_i = tr[:-200], tr[-200:]

    def ridge_pred(ii, jj, lam):
        Xi = Xs[ii]
        Uu, s, Vt = np.linalg.svd(Xi, full_matrices=False)
        W = Vt.T @ ((s / (s ** 2 + lam))[:, None] * (Uu.T @ (Zt[ii] - Zt[ii].mean(0))))
        return Zt[ii].mean(0) + Xs[jj] @ W

    best_lam, best = None, -np.inf
    for lam in LAMBDAS:
        tot = np.clip(_r2(ridge_pred(fit_i, val_i, lam), Zt[val_i]), 0, 1).sum()
        if tot > best:
            best, best_lam = tot, lam
    return _r2(ridge_pred(tr, te, best_lam), Zt[te]), best_lam

print(f"targets: {len(combos)} (deg 1..{args.max_deg}), train {M-args.test}/test {args.test}")
results = {}
for name, X in FEATURES.items():
    rank = np.linalg.matrix_rank(X[tr] - X[tr].mean(0))
    c, lam = capacities(X, Z)
    # noise floor: same pipeline, row-shuffled targets
    Zs = Z[rng.permutation(M)]
    cs, _ = capacities(X, Zs)
    floor = max(0.0, np.quantile(cs, 0.99))
    cap = np.where(c > floor, np.clip(c, 0, 1), 0.0)
    results[name] = (cap, floor, rank)
    tot = cap.sum()
    by_deg = {int(dd): float(cap[degs == dd].sum()) for dd in range(1, args.max_deg + 1)}
    odd = sum(v for k, v in by_deg.items() if k % 2)
    even = sum(v for k, v in by_deg.items() if not k % 2)
    print(f"{name:16s} rank {rank:3d}  lam {lam:g}  floor {floor:.3f}  TOTAL {tot:6.2f}   "
          f"by degree " + " ".join(f"{k}:{v:5.2f}" for k, v in by_deg.items())
          + f"   odd {odd:5.2f} / even {even:5.2f}")
    # top targets
    top = np.argsort(cap)[::-1][:6]
    print("   top:", ", ".join(f"{combos[j]}={cap[j]:.2f}" for j in top if cap[j] > 0))

if args.fig:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(FEATURES)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bottom = np.zeros(len(names))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, args.max_deg))
    for dd in range(1, args.max_deg + 1):
        vals = [results[n][0][degs == dd].sum() for n in names]
        ax.bar(names, vals, bottom=bottom, color=colors[dd - 1], label=f"degree {dd}")
        bottom += vals
    ax.set_ylabel("capacity (sum of out-of-sample R$^2$)")
    ax.set_title("Dambre IPC by output interpretation — 05d IPC dataset (amp 70)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(args.fig, dpi=140)
    print("figure:", args.fig)
