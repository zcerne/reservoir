import os
import numpy as np
import matplotlib.pyplot as plt

import class_reservoir_validator as cv


class PlotValidator(cv.Validator):
    """Validator + plotting. Inherits the analysis (modes/n1..n6) and adds figures."""

    def __init__(self, reservoir_path):
        super().__init__(reservoir_path)
        self.figdir = os.path.join(reservoir_path, "figures")

    # ------------------------------------------------------------ capacity
    def plot_capacity(self, save=True):
        """MODES / capacity figure: singular-value spectrum of G (absolute power on
        the left axis, normalized on a twin axis) + a table of the scalar capacity
        characteristics (n_eff, cond, throughput, rank, mixing, …)."""
        if "m1_bla" not in self.results:
            self.modes()                                  # builds G → m1/m2/m3
        m1 = self.results.get("m1_bla")
        if m1 is None:
            print("[plot] no field data for MODES — skipping capacity plot")
            return None
        s = np.asarray(m1["s"])
        s2 = s ** 2                                       # channel power
        s2n = s2 / (s2[0] + 1e-30)                        # normalized to strongest channel
        idx = np.arange(1, len(s) + 1)

        fig, (ax, ax_tbl) = plt.subplots(1, 2, figsize=(11, 4.5),
                                         gridspec_kw={"width_ratios": [2.2, 1]})

        # --- spectrum: absolute power (left) + normalized (right twin) ---
        ax.bar(idx, s2, color="steelblue", alpha=0.55, label="power $s_i^2$ (abs)")
        ax.set_xlabel("channel index $i$")
        ax.set_ylabel("absolute power $s_i^2$", color="steelblue")
        ax.tick_params(axis="y", labelcolor="steelblue")
        ax.set_yscale("log")

        axn = ax.twinx()
        axn.plot(idx, s2n, "o-", color="crimson", label="normalized $s_i^2/s_1^2$")
        axn.set_ylabel("normalized $s_i^2/s_1^2$", color="crimson")
        axn.tick_params(axis="y", labelcolor="crimson")
        axn.set_ylim(0, 1.05)
        # significant-channel threshold (1% of the strongest)
        axn.axhline(0.01, ls="--", lw=1, color="gray", alpha=0.7)
        # n_eff marker
        axn.axvline(m1["n_eff"], ls=":", lw=1.5, color="black",
                    label=f"$n_{{eff}}$ = {m1['n_eff']:.2f}")
        ax.set_title(f"Channel spectrum — {os.path.basename(self.path)}")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axn.get_legend_handles_labels()
        axn.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

        # --- table of scalar capacity characteristics ---
        m3s = self.results.get("m3_sum", {}); m3m = self.results.get("m3_mix", {})
        rows = [
            ("n_eff (usable channels)", f"{m1['n_eff']:.3f}"),
            ("rank", f"{m1['rank']}"),
            ("significant (|s|²≥1% max)", f"{m1['n_significant']}"),
            ("condition number", f"{m1['cond']:.3g}"),
            ("throughput Σ|s|²", f"{m3s.get('sum_rule', m1['sum_rule']):.4g}"),
            ("f_in / f_out", f"{m1['f_in']} / {m1['f_out']}"),
            ("probes", f"{m1['n_inputs']}"),
            ("mixing offdiag frac", "n/a" if m3m.get("offdiag_frac") is None
                                    else f"{m3m['offdiag_frac']:.3f}"),
            ("mode delocalization", f"{m3m.get('delocalization', float('nan')):.3f}"),
        ]
        ax_tbl.axis("off")
        tbl = ax_tbl.table(cellText=rows, colLabels=["capacity metric", "value"],
                           colWidths=[0.72, 0.28], loc="center", cellLoc="left")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
        ax_tbl.set_title("scalar characteristics", fontsize=10)

        fig.tight_layout()
        if save:
            os.makedirs(self.figdir, exist_ok=True)
            out = os.path.join(self.figdir, "capacity.png")
            fig.savefig(out, dpi=130, bbox_inches="tight")
            print(f"[plot] saved {out}")
        return fig
