"""Balance Scale accuracy, drive range 10-50, iso 05 vs LC 05b.

Intensity readout, all 3 polarizations, both sensors, swept over detector count.
Same softmax-ridge readout and split protocol as scripts/train_balance_scale_readout.py.
"""
import os, sys
import numpy as np

REPO = "/home/ziga/Nextcloud/Doktorski/Projects/Reservoir/gitcode"
LIPS = "/home/ziga/Lips_project/reservoir_runs/reservoir_types"
sys.path[:0] = [os.path.join(REPO, "scripts"), REPO]
import train_balance_scale_readout as T                        # noqa: E402

SETS = {
    "iso 05": f"{REPO}/data/reservoir_types/res_iso_gain/05/datasets/"
              "balance_scale_4src_a10_50.npz",
    "LC 05b": f"{LIPS}/res_lc_gain/05b/datasets/balance_scale_a10_50.npz",
}
COMPS = ["Ex", "Ey", "Ez"]
NPTS = [10, 25, 50, 100, 200]
REPEATS, TEST_FRAC, RIDGE, LAM = 5, 0.3, 1e-3, 0.55

rows = {}
for design, path in SETS.items():
    d = np.load(path, allow_pickle=True)
    y = np.asarray(d["labels"]).astype(int)
    n = len(y)
    n2f = np.asarray(d["outputs"])                       # (n, 3*npix)
    lam = 1.0 / np.asarray(d["m2_freqs"])
    il = int(np.argmin(np.abs(lam - LAM)))
    m2 = np.concatenate([np.asarray(d[f"m2_{c}"])[:, il, :] for c in COMPS], 1)
    amp = np.asarray(d["amp_range"]) if "amp_range" in d.files else None
    print(f"\n=== {design}: {n} samples, amp_range {amp}, "
          f"n2f {n2f.shape}, monitor_2 {m2.shape}, lam {lam[il]:.4f}")

    rng0 = np.random.default_rng(0)
    perm = rng0.permutation(n); ntr = int(round((1 - TEST_FRAC) * n))
    tr, te = perm[:ntr], perm[ntr:]

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
            rows[(design, sensor, N)] = (float(np.mean(accs)), float(np.std(accs)),
                                         min(N, npix))
            print(f"  {sensor:4s} N={N:4d} (npix {npix:4d}) "
                  f"acc {np.mean(accs):.4f} +- {np.std(accs):.4f}")

np.savez(f"{REPO}/data/reservoir_types/stats_balance_a10_50.npz",
         rows=np.array(rows, dtype=object))

print("\n\n| N detectors | iso 05 n2f | iso 05 out | LC 05b n2f | LC 05b out |")
print("| --- | --- | --- | --- | --- |")
for N in NPTS:
    c = [f"{rows[(d, s, N)][0]:.4f} ± {rows[(d, s, N)][1]:.3f}"
         for d in SETS for s in ("n2f", "out")]
    print(f"| {N} | " + " | ".join(c) + " |")
