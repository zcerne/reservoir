"""IPC vs drive level — all available probe sets, field and intensity, to d5.

Also quantifies the honest answer to "can 400 probes go to d5?": the same 1000-probe
set is subsampled to 400 and re-measured, so the probe-count effect is isolated from
the drive effect.

n6_dambre applies a noise floor thr = 2*F/M, so M matters twice: directly through the
R^2 bias, and through the threshold that zeroes small capacities. At F=40 that is
thr=0.08 for M=1000 but thr=0.20 for M=400 — and high-degree capacity arrives as MANY
SMALL terms, exactly the ones a raised floor deletes.
"""
import os, sys, json
import numpy as np

REPO = "/home/ziga/Nextcloud/Doktorski/Projects/Reservoir/gitcode"
LIPS = "/home/ziga/Lips_project/reservoir_runs/reservoir_types"
sys.path[:0] = [os.path.join(REPO, "characterization"), REPO]
import n6_dambre as n6                                          # noqa: E402

ISO = f"{REPO}/data/reservoir_types/res_iso_gain/05/datasets"
LC = f"{LIPS}/res_lc_gain/05b/datasets"
SETS = [
    ("iso 05", 10,  f"{ISO}/ipc_4src_a10.npz"),
    ("iso 05", 50,  f"{ISO}/ipc_4src_a50.npz"),
    ("iso 05", 100, f"{ISO}/ipc_4src_a100.npz"),
    ("LC 05b", 50,  f"{LC}/ipc_4src_a50.npz"),
    ("LC 05b", 100, f"{LC}/ipc_4src_a100.npz"),
]
MAXDEG, K = 5, 40


def reduce_k(X, k):
    Xc = X - X.mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    rank = int((s / (s[0] + 1e-300) > 1e-8).sum())
    kk = min(k, max(4, rank))
    return Xc @ Vt[:kk].conj().T, kk, rank


def run(U, X, readout, maxdeg=MAXDEG):
    Xr = np.abs(X) ** 2 if readout == "intensity" else X
    Xk, kk, rank = reduce_k(Xr, K)
    r = n6.dambre_ipc({"inputs": U, "outputs": Xk},
                      max_degree=maxdeg, max_features=Xk.shape[1])
    return r, kk, rank


rows, missing = {}, []
for design, drive, path in SETS:
    if not os.path.exists(path):
        missing.append((design, drive, path)); continue
    d = np.load(path, allow_pickle=True)
    U = np.asarray(d["inputs"]).real
    X = np.asarray(d["outputs"])
    M = X.shape[0]
    print(f"\n=== {design} drive {drive}: {M} probes")
    md = MAXDEG if M >= 1000 else 3
    for readout in ("field", "intensity"):
        r, kk, rank = run(U, X, readout, maxdeg=md)
        rows[(design, drive, readout)] = dict(res=r, k=kk, rank=rank, M=M, md=md)
        byd = {dd: round(v, 2) for dd, v in sorted(r["ipc_by_degree"].items())}
        print(f"  {readout:9s} M={M:4d} maxdeg={md} k={kk:2d} rank={rank:3d} thr={r['threshold']:.3f}  "
              f"total={r['ipc_total']:7.3f} NL={r['nonlinear_fraction']:.3f} {byd}")
if missing:
    print("\nMISSING:", [(a, b) for a, b, _ in missing])

# --- probe-count control: the SAME data at 1000 and at 400 -------------------
print("\n\n=== probe-count control (iso 05 drive 50, identical data) ===")
d = np.load(f"{ISO}/ipc_4src_a50.npz", allow_pickle=True)
U = np.asarray(d["inputs"]).real
X = np.asarray(d["outputs"])
ctrl = {}
for M in (1000, 400):
    sub = slice(0, M)
    for readout in ("field", "intensity"):
        r, kk, _ = run(U[sub], X[sub], readout)
        ctrl[(M, readout)] = r
        byd = {dd: round(v, 2) for dd, v in sorted(r["ipc_by_degree"].items())}
        print(f"  M={M:4d} {readout:9s} thr={r['threshold']:.3f} "
              f"total={r['ipc_total']:7.3f}  {byd}")
np.savez(f"{REPO}/data/reservoir_types/stats_dambre_drive_sweep.npz",
         rows=np.array(rows, dtype=object), ctrl=np.array(ctrl, dtype=object))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEG = list(range(1, MAXDEG + 1))
keys = [k for k in rows]
designs = sorted({k[0] for k in keys})
fig, axes = plt.subplots(2, len(designs), figsize=(7 * len(designs), 8.6),
                         sharey="row", squeeze=False)
CMAP = {10: "C0", "10 (meep)": "C9", "10 (gpumeep)": "C5",
        50: "C1", 100: "C3"}
for r_i, readout in enumerate(("field", "intensity")):
    for c_i, design in enumerate(designs):
        ax = axes[r_i][c_i]
        drives = sorted(({k[1] for k in keys if k[0] == design}), key=str)
        W = 0.8 / len(drives)
        x = np.arange(len(DEG))
        for j, dr in enumerate(drives):
            v = rows[(design, dr, readout)]
            md = v.get("md", MAXDEG)
            dsel = [dd for dd in DEG if dd <= md]
            vals = [v["res"]["ipc_by_degree"].get(dd, 0.0) for dd in dsel]
            off = (j - (len(drives) - 1) / 2) * W
            lab = (f"drive {dr}  (M={v['M']})  total {v['res']['ipc_total']:.1f}"
                   + ("  [to d3]" if md < MAXDEG else ""))
            ax.bar(np.arange(len(dsel)) + off, vals, W,
                   color=CMAP.get(dr, "C4"), label=lab)
            for xx, val in zip(np.arange(len(dsel)) + off, vals):
                if val > 0.05:
                    ax.annotate(f"{val:.1f}", (xx, val), fontsize=6.5, ha="center",
                                textcoords="offset points", xytext=(0, 2))
        ax.set_xticks(x); ax.set_xticklabels([f"d{dd}" for dd in DEG])
        ax.set_title(f"{design} — {readout}", fontsize=10)
        ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
        if c_i == 0:
            ax.set_ylabel(f"IPC ({readout})")
fig.suptitle("IPC vs drive level — Dambre degree spectrum to d5\n"
             f"n2f sensor, all 3 components, channels capped at k={K}, "
             "noise floor 2F/M per capacity — 400-probe sets measured and drawn to d3 ONLY",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.91])
p = f"{REPO}/data/reservoir_types/dambre_drive_sweep.png"
fig.savefig(p, dpi=140, bbox_inches="tight")
print("\nwrote", p)
