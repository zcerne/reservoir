"""Drive-50 vs drive-100 IPC comparison for the 05 (isotropic) and 05b (LC patch) cavities.

  python plotting/plot_ipc_drive_comparison.py
  python plotting/plot_ipc_drive_comparison.py --max-degree 5 --figdir <dir>

Recomputes the Dambre spectra straight from the raw ipc_4src_a*.npz rather than reading
the cached n6 stats, so both designs are guaranteed to come out of the SAME pipeline —
mixing a max_degree=5 run with a max_degree=3 one is what made the earlier 05 numbers
(25.60 field total) irreproducible against a fresh max_degree=4 recompute (23.29).
State reduction mirrors Validator._reduce_state: PCA to min(rank, M/10) leading
channels, then dambre_ipc with its own even-spaced cap opted out.

Writes three figures:
  ipc_50_vs_100.png                    — design 05 alone, degree spectra + per-degree cost
  ipc_05_vs_05b_50_vs_100.png          — both designs, 2x2 small multiples
  ipc_linear_vs_nonlinear_50_vs_100.png — d1 vs d>=2 dumbbell, the headline result
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from characterization import n6_dambre as n6

REPO = Path(__file__).resolve().parent.parent
TYPES = REPO / "data" / "reservoir_types"
DEFAULT_FIGDIR = TYPES / "res_iso_gain" / "05" / "figures"

DESIGNS = [
    ("05 isotropic", TYPES / "res_iso_gain" / "05" / "datasets" / "ipc_4src_a{}.npz"),
    ("05b LC patch", TYPES / "res_lc_gain" / "05b" / "datasets" / "ipc_4src_a{}.npz"),
]
DRIVES = (50, 100)
MODES = ("field", "intensity")

# Drive is a two-category identity split, so the hues are assigned in fixed order and
# never cycled. Blue/orange validated: worst adjacent CVD dE 26.4 (protan), 29.9
# (tritan), normal-vision dE 32.6. Orange sits below 3:1 against white, which is why
# every bar carries a visible value label.
C50, C100 = "#3b6fd4", "#e07b39"
INK, MUTED, GRID, RULE = "#1f2328", "#6b7280", "#e5e7eb", "#d1d5db"


def _reduce_state(X):
    """Mirror of characterization.class_reservoir_validator.Validator._reduce_state."""
    M = X.shape[0]
    Xf = X.reshape(M, -1)
    F = Xf.shape[1]
    k = int(min(F, max(4, M // 10)))
    if F <= k:
        return Xf, F, F
    Xc = Xf - Xf.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    rank = int((S / (S[0] + 1e-300) > 1e-8).sum())
    k = min(k, max(4, rank))
    return Xc @ Vt[:k].conj().T, k, rank


def spectra(max_degree: int, degs: list[int]) -> dict:
    """{(design, drive, mode): {by, total, nl, d1, M, ch, rank}} for every combination."""
    out = {}
    for design, tmpl in DESIGNS:
        for drive in DRIVES:
            path = Path(str(tmpl).format(drive))
            if not path.exists():
                print(f"[ipc] MISSING {path} — skipping", flush=True)
                continue
            d = np.load(path, allow_pickle=True)
            U, X = d["inputs"], np.asarray(d["outputs"])
            for mode, Xv in (("field", X), ("intensity", np.abs(X) ** 2)):
                Xr, k, rank = _reduce_state(np.asarray(Xv))
                r = n6.dambre_ipc({"inputs": U, "outputs": Xr},
                                  max_degree=max_degree, max_features=Xr.shape[1])
                by = {int(a): float(b) for a, b in r["ipc_by_degree"].items()}
                out[(design, drive, mode)] = {
                    "by": [by.get(g, 0.0) for g in degs],
                    "total": float(r["ipc_total"]),
                    "nl": float(sum(v for g, v in by.items() if g >= 2)),
                    "d1": by.get(1, 0.0),
                    "M": int(U.shape[0]), "ch": k, "rank": rank,
                }
                print(f"[ipc] {design:14s} a{drive:<4d}{mode:10s} M={U.shape[0]:4d} "
                      f"ch={k:3d} total={r['ipc_total']:6.2f} "
                      f"d1={by.get(1, 0.0):5.2f} d>=2={out[(design, drive, mode)]['nl']:6.2f}",
                      flush=True)
    return out


def _style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, length=0)


def _grouped(ax, v50, v100, degs):
    """Grouped degree spectrum, drive 50 beside drive 100, with a gap between fills."""
    x = np.arange(len(degs))
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.012, v50, w, color=C50, label="drive 50", zorder=3)
    b2 = ax.bar(x + w / 2 + 0.012, v100, w, color=C100, label="drive 100", zorder=3)
    for bars, vals in ((b1, v50), (b2, v100)):
        for bar, v in zip(bars, vals):
            if v <= 0.005:                       # an exactly-dead degree needs no label
                continue
            ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=7, color=MUTED)
    ax.set_xticks(x, [f"d{d}" for d in degs])
    ax.set_ylim(0, max(max(v50), max(v100), 1e-9) * 1.22)
    _style(ax)


def _delta_note(a, b, key):
    return f"{a[key]:.2f} → {b[key]:.2f}  ({100 * (b[key] - a[key]) / max(a[key], 1e-9):+.1f}%)"


def fig_design_05(S, figdir, degs, max_degree):
    design = "05 isotropic"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    for ax, mode in zip(axes[:2], MODES):
        a, b = S[(design, 50, mode)], S[(design, 100, mode)]
        _grouped(ax, a["by"], b["by"], degs)
        ax.set_title(f"{mode} readout", fontsize=11, color=INK)
        ax.set_ylabel("IPC (Σ capacity)", color=MUTED, fontsize=9)
        ax.set_xlabel("target polynomial degree", color=MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        ax.text(0.02, 0.97, f"total  {_delta_note(a, b, 'total')}\nd≥2   {_delta_note(a, b, 'nl')}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                color=INK, linespacing=1.6)

    ax = axes[2]
    labels, deltas = [], []
    for mode in MODES:
        a, b = S[(design, 50, mode)], S[(design, 100, mode)]
        for i, dg in enumerate(degs):
            if max(a["by"][i], b["by"][i]) < 0.01:
                continue
            labels.append(f"d{dg}\n{mode[:3]}")
            deltas.append(b["by"][i] - a["by"][i])
    ax.bar(np.arange(len(deltas)), deltas, 0.62, color=C100, zorder=3)
    for i, v in enumerate(deltas):
        ax.annotate(f"{v:+.2f}", (i, v), textcoords="offset points",
                    xytext=(0, -11 if v < 0 else 4), ha="center", fontsize=7, color=MUTED)
    ax.axhline(0, color="#9ca3af", lw=1)
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=8)
    ax.set_ylim(min(deltas) * 1.32, max(0.0, max(deltas)) + 0.45)
    ax.set_ylabel("capacity change, drive 50 → 100", color=MUTED, fontsize=9)
    ax.set_title("what over-driving costs, per degree", fontsize=11, color=INK)
    _style(ax)

    fig.suptitle("05 isotropic cavity, 4-source IPC — drive 50 vs 100  "
                 f"(both MEEP, {S[(design, 50, 'field')]['M']} probes, max_degree={max_degree})",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, figdir / "ipc_50_vs_100.png")


def fig_cross_design(S, figdir, degs, max_degree):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    for r, (design, _) in enumerate(DESIGNS):
        for c, mode in enumerate(MODES):
            ax = axes[r][c]
            a, b = S[(design, 50, mode)], S[(design, 100, mode)]
            _grouped(ax, a["by"], b["by"], degs)
            ax.set_title(f"{design} — {mode} readout", fontsize=10.5, color=INK)
            if r == len(DESIGNS) - 1:
                ax.set_xlabel("target polynomial degree", color=MUTED, fontsize=9)
            if c == 0:
                ax.set_ylabel("IPC (Σ capacity)", color=MUTED, fontsize=9)
            if (r, c) == (0, 0):
                ax.legend(frameon=False, fontsize=9, loc="upper right")
            # |E|² is even in the drive, so it annihilates every odd-degree target —
            # the d1 it reports is noise-floor residue, not a linear channel. Quoting
            # its percentage change (05b: 0.47→0.21, "−55%") would be reading noise.
            tail = ("d1     parity-suppressed in |E|²" if mode == "intensity"
                    else f"d1     {_delta_note(a, b, 'd1')}")
            ax.text(0.02, 0.97,
                    f"total  {_delta_note(a, b, 'total')}\n"
                    f"d≥2   {_delta_note(a, b, 'nl')}\n{tail}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    color=INK, linespacing=1.6)

    fig.suptitle("Drive 50 beats drive 100 in both cavities — over-driving costs the "
                 "high-order degrees, not the linear one",
                 fontsize=13, color=INK, y=0.985)
    fig.text(0.5, 0.006,
             f"4-source IPC, max_degree={max_degree}.  Within-design 50↔100 is probe-matched; "
             f"across-design totals are NOT — 05 has {S[(DESIGNS[0][0], 50, 'field')]['M']} probes, "
             f"05b has {S[(DESIGNS[1][0], 50, 'field')]['M']}, which sets the PCA channel cap "
             "and hence the capacity ceiling.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    return _save(fig, figdir / "ipc_05_vs_05b_50_vs_100.png")


def fig_linear_vs_nonlinear(S, figdir):
    """Dumbbell: the linear channel survives over-driving, the nonlinear ones do not."""
    rows = [(d, m) for d, _ in DESIGNS for m in MODES]
    # The d1 panel is FIELD-ONLY on purpose: |E|² is even in the drive and annihilates
    # odd-degree targets, so an intensity d1 row would plot parity residue (0.00, 0.21)
    # next to real linear capacity and invite a "-55%" reading of pure noise.
    panels = (("nl", "nonlinear capacity  (d ≥ 2)", rows),
              ("d1", "linear capacity  (d1)  — field readout only",
               [r for r in rows if r[1] == "field"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)

    for ax, (key, title, prows) in zip(axes, panels):
        y = np.arange(len(rows))[::-1]
        keep = {r: yi for r, yi in zip(rows, y) if r in prows}
        for (design, mode), yi in keep.items():
            a, b = S[(design, 50, mode)], S[(design, 100, mode)]
            v50, v100 = a[key], b[key]
            ax.plot([v50, v100], [yi, yi], color=RULE, lw=2.5, zorder=2,
                    solid_capstyle="round")
            ax.scatter([v50], [yi], s=95, color=C50, zorder=3, edgecolor="white",
                       linewidth=2, label="drive 50" if yi == y[0] else None)
            ax.scatter([v100], [yi], s=95, color=C100, zorder=3, edgecolor="white",
                       linewidth=2, label="drive 100" if yi == y[0] else None)
            lo, hi = min(v50, v100), max(v50, v100)
            ax.annotate(f"{100 * (v100 - v50) / max(v50, 1e-9):+.1f}%", (hi, yi),
                        textcoords="offset points", xytext=(10, 0), va="center",
                        fontsize=8.5, color=MUTED)
            # clear BOTH markers: when the pair nearly coincides (05b d1, 3.91 vs 3.86)
            # a -9pt offset lands the value label on top of the other dot
            ax.annotate(f"{lo:.2f}", (lo, yi), textcoords="offset points",
                        xytext=(-15, 0), va="center", ha="right", fontsize=8.5, color=MUTED)
        ax.set_yticks(y, [f"{d} · {m}" for d, m in rows], fontsize=9.5)
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_xlabel("IPC (Σ capacity)", color=MUTED, fontsize=9)
        _style(ax, axis="x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", labelcolor=INK)

    axes[0].set_xlim(0, max(v["nl"] for v in S.values()) * 1.18)
    axes[1].set_xlim(0, max(v["d1"] for v in S.values()) * 1.34)
    # figure-level legend: an in-axes one lands on the bottom dumbbell at every x-limit
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=2,
               loc="upper left", bbox_to_anchor=(0.055, 0.945))
    fig.suptitle("Over-driving saturates the nonlinear channels and leaves the linear one intact",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, figdir / "ipc_linear_vs_nonlinear_50_vs_100.png")


def _save(fig, out: Path) -> Path:
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ipc] wrote {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figdir", default=str(DEFAULT_FIGDIR),
                    help="output directory (default: res_iso_gain/05/figures)")
    ap.add_argument("--max-degree", type=int, default=5,
                    help="highest total polynomial degree in the IPC target family")
    a = ap.parse_args()

    degs = list(range(1, a.max_degree + 1))
    figdir = Path(a.figdir)
    S = spectra(a.max_degree, degs)
    missing = [k for k in ((d, s, m) for d, _ in DESIGNS for s in DRIVES for m in MODES)
               if k not in S]
    if missing:
        print(f"[ipc] {len(missing)} combination(s) unavailable: {missing}", flush=True)
        return
    fig_design_05(S, figdir, degs, a.max_degree)
    fig_cross_design(S, figdir, degs, a.max_degree)
    fig_linear_vs_nonlinear(S, figdir)


if __name__ == "__main__":
    main()
