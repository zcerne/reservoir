"""Ez effect on Dambre IPC: I(Ex,Ey) vs I(Ex,Ey,Ez), and the same for fields.

iso 05 vs LC 05b, 1000 probes at drive 50, n2f sensor, max_degree 5.
Channels MATCHED at k=40 across every cell — dropping Ez removes a third of the
raw columns, and IPC total grows with channel count, so without matching this
measures the budget rather than Ez.
"""
import os, sys
import numpy as np

REPO = "/home/ziga/Nextcloud/Doktorski/Projects/Reservoir/gitcode"
LIPS = "/home/ziga/Lips_project/reservoir_runs/reservoir_types"
sys.path[:0] = [os.path.join(REPO, "characterization"), REPO]
import n6_dambre as n6                                          # noqa: E402

SETS = {
    "iso 05": f"{REPO}/data/reservoir_types/res_iso_gain/05/datasets/ipc_4src_a50.npz",
    "LC 05b": f"{LIPS}/res_lc_gain/05b/datasets/ipc_4src_a50.npz",
}
COMPS = ["Ex", "Ey", "Ez"]
KEEPS = {"Ex,Ey": ["Ex", "Ey"], "Ex,Ey,Ez": ["Ex", "Ey", "Ez"]}
MAXDEG, K_MATCH = 5, 40


def reduce_k(X, k):
    Xc = X - X.mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    rank = int((s / (s[0] + 1e-300) > 1e-8).sum())
    kk = min(k, max(4, rank))
    return Xc @ Vt[:kk].conj().T, kk, rank


rows = {}
for design, path in SETS.items():
    d = np.load(path, allow_pickle=True)
    U = np.asarray(d["inputs"]).real
    o = np.asarray(d["outputs"])
    npix = o.shape[1] // len(COMPS)
    cube = o.reshape(o.shape[0], len(COMPS), npix)
    print(f"\n=== {design}: {o.shape[0]} probes, {npix} px/comp")
    for tag, keep in KEEPS.items():
        ci = [COMPS.index(c) for c in keep]
        X = cube[:, ci, :].reshape(o.shape[0], -1)
        for readout in ("field", "intensity"):
            Xr = np.abs(X) ** 2 if readout == "intensity" else X
            Xk, kk, rank = reduce_k(Xr, K_MATCH)
            r = n6.dambre_ipc({"inputs": U, "outputs": Xk},
                              max_degree=MAXDEG, max_features=Xk.shape[1])
            rows[(design, tag, readout)] = dict(res=r, k=kk, rank=rank)
            byd = {dd: round(v, 2) for dd, v in sorted(r["ipc_by_degree"].items())}
            print(f"  {tag:9s} {readout:9s} k={kk} rank={rank:4d}  "
                  f"total={r['ipc_total']:7.3f}  NL={r['nonlinear_fraction']:.3f}  {byd}")

np.savez(f"{REPO}/data/reservoir_types/stats_dambre_ez_effect.npz",
         rows=np.array(rows, dtype=object))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEG = list(range(1, MAXDEG + 1))
fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharey="row")
W = 0.38
for r_i, readout in enumerate(("field", "intensity")):
    for c_i, design in enumerate(SETS):
        ax = axes[r_i, c_i]
        x = np.arange(len(DEG))
        for off, tag, col in ((-W / 2, "Ex,Ey", "C0"), (W / 2, "Ex,Ey,Ez", "C3")):
            v = rows[(design, tag, readout)]
            vals = [v["res"]["ipc_by_degree"].get(dd, 0.0) for dd in DEG]
            ax.bar(x + off, vals, W, color=col,
                   label=f"{tag}   total {v['res']['ipc_total']:.1f}")
            for xx, val in zip(x + off, vals):
                if val > 0.05:
                    ax.annotate(f"{val:.1f}", (xx, val), fontsize=7, ha="center",
                                textcoords="offset points", xytext=(0, 2))
        ax.set_xticks(x); ax.set_xticklabels([f"d{dd}" for dd in DEG])
        ax.set_title(f"{design} — {readout}", fontsize=10)
        ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
        if c_i == 0:
            ax.set_ylabel(f"IPC ({readout})")
fig.suptitle("Ez effect on Dambre IPC — with vs without the Ez channel\n"
             f"iso 05 vs LC 05b, 1000 probes, drive 50, n2f sensor, "
             f"max_degree {MAXDEG}, channels capped at k={K_MATCH} "
             "(Ex,Ey field is rank-limited to 23/24, so its k is lower — see note)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.91])
p = f"{REPO}/data/reservoir_types/dambre_ez_effect.png"
fig.savefig(p, dpi=140, bbox_inches="tight")
print("\nwrote", p)
