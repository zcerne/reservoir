"""Balance Scale accuracy for block_iso 01, drive range 50-100.

Same softmax-ridge readout and split protocol as scripts/train_balance_scale_readout.py
and plotting/balance_a10_50.py (intensity readout, 3 components, both sensors,
swept over detector count).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/ziga/Nextcloud/Doktorski/Projects/Reservoir/gitcode"
sys.path[:0] = [os.path.join(REPO, "scripts"), REPO]
import train_balance_scale_readout as T                        # noqa: E402

PATH = ("/home/ziga/Orion/resevoir/data/reservoir_types/block_iso_gain/01/"
        "datasets/balance_scale_4src_a50_100.npz")
OUTDIR = os.path.join(REPO, "data/reservoir_types")
COMPS = ["Ex", "Ey", "Ez"]
NPTS = [10, 25, 50, 100, 200]
REPEATS, TEST_FRAC, RIDGE, LAM = 5, 0.3, 1e-3, 0.55

d = np.load(PATH, allow_pickle=True)
print("keys:", d.files, flush=True)
y = np.asarray(d["labels"]).astype(int)
n = len(y)
n2f = np.asarray(d["outputs"])
lam = 1.0 / np.asarray(d["m2_freqs"])
il = int(np.argmin(np.abs(lam - LAM)))
m2 = np.concatenate([np.asarray(d[f"m2_{c}"])[:, il, :] for c in COMPS], 1)
print(f"block 01 a50-100: {n} samples, n2f {n2f.shape}, monitor_2 {m2.shape}, "
      f"lam {lam[il]:.4f}, label counts {np.bincount(y)}", flush=True)

rng0 = np.random.default_rng(0)
perm = rng0.permutation(n); ntr = int(round((1 - TEST_FRAC) * n))
tr, te = perm[:ntr], perm[ntr:]

rows = {}
for sensor, X in (("n2f", n2f), ("out", m2)):
    npix = X.shape[1] // len(COMPS)
    for N in NPTS:
        accs = []
        for r in range(REPEATS):
            rng = np.random.default_rng(1000 * N + r)
            pts = rng.choice(npix, size=min(N, npix), replace=False)
            F = T.features(X, None, COMPS, pts, "intensity")
            _, a = T.softmax_ridge(F[tr], y[tr], F[te], y[te], ridge=RIDGE)
            accs.append(a)
        rows[(sensor, N)] = (float(np.mean(accs)), float(np.std(accs)), min(N, npix))
        print(f"  {sensor:4s} N={N:4d} (npix {npix:4d}) "
              f"acc {np.mean(accs):.4f} +- {np.std(accs):.4f}", flush=True)

np.savez(os.path.join(OUTDIR, "stats_balance_block01_a50_100.npz"),
         rows=np.array(rows, dtype=object))

fig, ax = plt.subplots(figsize=(6.4, 4))
for sensor, label in (("n2f", "n2f far screen"), ("out", "monitor_2 (output guide)")):
    m = [rows[(sensor, N)][0] for N in NPTS]
    s = [rows[(sensor, N)][1] for N in NPTS]
    ax.errorbar(NPTS, m, yerr=s, marker="o", capsize=3, label=label)
ax.axhline(1 / 3, color="gray", ls=":", lw=1, label="chance (majority ~0.46)")
ax.set_xscale("log")
ax.set_xlabel("detector count N")
ax.set_ylabel("test accuracy")
ax.set_title("Balance scale, block_iso 01, amp 50-100, linear softmax-ridge")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "balance_block01_a50_100.png"), dpi=140)
print("figure saved", flush=True)
