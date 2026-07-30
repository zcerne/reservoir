"""Plot for characterization/n4_harmonics_distortion.py's harmonic_specter() result."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# colour per bin class — shared by the spectrum and the power-by-kind bars
KIND_COLOR = {"dc": "gray", "fundamental": "C0", "harmonic": "C1",
              "intermod": "C2", "other": "C3"}


def plot_n4_harmonics_distortion(res: dict, fig_dir: str | Path, suffix: str = "",
                                 data: dict | None = None,
                                 normalize: bool = True) -> Path:
    """Where the nonlinear energy sits.

    Top (whenever the result carries the per-bin spectrum, i.e. `spec_nu` from
    harmonic_specter): the phase-sweep spectrum itself, each bin coloured by
    class and the identified ones labelled (f1, 2f1, f2-f1, ...). That panel
    shows WHICH products exist — the two summary bar charts below can only
    imply it, and the distinction matters because an |E|² readout manufactures
    the order-2 set on its own while only higher orders prove device
    nonlinearity. Older result dicts without the spectrum still plot (two
    panels, as before).

    `data` (the harmonic dataset itself) is optional; when given, its outputs
    are shown as |output|² over the sweep.
    """
    has_spec = "spec_nu" in res and len(np.asarray(res["spec_nu"])) > 0
    fund_p = float(res["power_by_kind"].get("fundamental", 0.0))
    norm = (fund_p / 100.0) if (normalize and fund_p > 0) else 1.0
    unit = " [% of fundamental]" if norm != 1.0 else ""
    n_bot = 3 if data is not None else 2
    if has_spec:
        fig = plt.figure(figsize=(5.0 * n_bot + 1.0, 8.0))
        gs = fig.add_gridspec(2, n_bot, height_ratios=(1.15, 1.0))
        ax_s = fig.add_subplot(gs[0, :])
        axes = [fig.add_subplot(gs[1, i]) for i in range(n_bot)]
    else:
        fig, axs = plt.subplots(1, n_bot, figsize=(5.0 * n_bot, 4.5))
        axes = list(np.atleast_1d(axs))
        ax_s = None
    ax1, ax2 = axes[0], axes[1]

    # ---------- the spectrum itself ----------
    if ax_s is not None:
        nu = np.asarray(res["spec_nu"])
        pw = np.asarray(res["spec_power"], dtype=float) / norm
        kinds_b = np.asarray(res["spec_kind"])
        labels = np.asarray(res["spec_label"])
        pmax = pw.max() if pw.size and pw.max() > 0 else 1.0
        # Pin the axis a decade below the smallest bin the analysis still counts
        # (rel_thresh 1e-9): bins under that are numerical zeros — sitting at
        # 1e-27 they would otherwise stretch the log axis over 30 decades and
        # squash every real product into a sliver at the top.
        floor = pmax * 1e-10
        # keep it readable: out to the last bin carrying real power
        sig = nu[pw > max(pmax * 1e-9, floor)]
        nu_max = int(sig.max()) + 2 if sig.size else int(nu.max())
        m = nu <= nu_max
        ax_s.bar(nu[m], np.maximum(pw[m], floor), width=0.62,
                 color=[KIND_COLOR.get(k, "0.7") for k in kinds_b[m]])
        for v, p, lb in zip(nu[m], pw[m], labels[m]):
            if lb and p > floor * 4:
                ax_s.annotate(lb, (v, p), textcoords="offset points",
                              xytext=(0, 4), ha="center", fontsize=7.5)
        ax_s.set_yscale("log")
        if norm != 1.0:
            ax_s.set_ylim(1e-7, 3000.0)
        else:
            ax_s.set_ylim(floor, pmax * 12)
        ax_s.set_xlim(-0.7, nu_max + 0.7)
        ax_s.set_xticks(range(0, nu_max + 1, max(1, (nu_max + 1) // 24)))
        ax_s.set_xlabel("sweep-frequency bin ν")
        ax_s.set_ylabel("power" + unit)
        tone_txt = ", ".join(f"f{i + 1}={t}" for i, t in enumerate(res["tones"]))
        ax_s.set_title(f"phase-sweep spectrum ({tone_txt})")
        ax_s.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=k)
                             for k, c in KIND_COLOR.items()],
                    fontsize=7.5, ncol=5, loc="upper right")

    # ---------- power by kind ----------
    kinds = ["dc", "fundamental", "harmonic", "intermod", "other"]
    vals = np.array([res["power_by_kind"][k] for k in kinds], dtype=float) / norm
    vpos = vals[vals > 0]
    kfloor = (vpos.min() * 0.1) if vpos.size else 1e-30
    ax1.bar(kinds, np.maximum(vals, kfloor), color=[KIND_COLOR[k] for k in kinds])
    ax1.set_yscale("log")
    if norm != 1.0:
        ax1.set_ylim(1e-7, 3000.0)
        for x, v in enumerate(vals):
            if v > 0:
                ax1.annotate(f"{v:.3g}%", (x, v), textcoords="offset points",
                             xytext=(0, 3), ha="center", fontsize=7.5)
    ax1.set_ylabel("power" + unit)
    ax1.set_title("power by kind")
    ax1.tick_params(axis="x", rotation=30)

    # ---------- power by order ----------
    orders = sorted(res["power_by_order"])
    ovals = np.array([res["power_by_order"][o] for o in orders], dtype=float) / norm
    opos = ovals[ovals > 0]
    ofloor = (opos.min() * 0.1) if opos.size else 1e-30
    ax2.bar([str(o) for o in orders], np.maximum(ovals, ofloor), color="C1")
    ax2.set_yscale("log")
    if norm != 1.0:
        ax2.set_ylim(1e-7, 3000.0)
        for x, v in enumerate(ovals):
            if v > 0:
                ax2.annotate(f"{v:.3g}%", (x, v), textcoords="offset points",
                             xytext=(0, 3), ha="center", fontsize=7.5)
    ax2.set_xlabel("order")
    ax2.set_ylabel("power" + unit)
    ax2.set_title(f"power by order (max nonlinear order = {res['max_order']})")

    # ---------- optional: the raw sweep ----------
    if data is not None:
        ax3 = axes[2]
        Y = np.asarray(data["outputs"])
        I = np.abs(Y.reshape(Y.shape[0], -1)) ** 2
        t = np.asarray(data["t"]) if "t" in data else np.arange(Y.shape[0])
        im = ax3.imshow(I.T, origin="lower", aspect="auto", cmap="magma",
                        extent=(float(t[0]), float(t[-1]), 0, I.shape[1]))
        ax3.set_xlabel("sweep phase t [rad]")
        ax3.set_ylabel("output feature")
        ax3.set_title("|output|² over the sweep")
        fig.colorbar(im, ax=ax3)

    verdict = "LINEAR" if res["linear"] else f"NONLINEAR (order {res['max_order']})"
    fig.suptitle(f"Harmonic/intermod distortion (Method D) | tones={res['tones']} | "
                 f"THD={100 * res['thd']:.2f}%  IMD={100 * res['imd']:.2f}%  "
                 f"distortion_frac={100 * res['distortion_frac']:.2f}% -> {verdict}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if ax_s is not None else None)
    out = Path(fig_dir) / f"n4_harmonics_distortion{suffix}.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out
