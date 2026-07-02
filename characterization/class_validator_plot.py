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

    # -------------------------------------------------------- nonlinearity
    def plot_nonlinear_stats(self, save=True):
        """NONLINEARITY table — A. superposition (n1) + B. linear residual (n2),
        each in the field view (Maxwell-linear sanity ≈0) and the |E|² readout view
        (the reservoir's actual nonlinearity). Field should be LINEAR, |E|² NONLINEAR."""
        if "n1_field" not in self.results:
            self.superposition()
        if not any(k.startswith("n2") for k in self.results):
            self.linear_residual()
        if "n3_field" not in self.results:
            self.amplitude()
        R = self.results

        def _cell(x, fmt="{:.3g}"):
            return "n/a" if x is None else fmt.format(x)

        # n2 may be split (complex ipc → n2_field/n2_intensity) or single (intensity ipc → n2)
        n2f = R.get("n2_field"); n2i = R.get("n2_intensity") or R.get("n2")
        n1f = R.get("n1_field"); n1i = R.get("n1_intensity")
        n3f = R.get("n3_field"); n3i = R.get("n3_intensity")

        rows = [
            ("A. superposition  R²",
             _cell(n1f.get("r2") if n1f else None, "{:.4f}"),
             _cell(n1i.get("r2") if n1i else None, "{:.4f}")),
            ("A. superposition  mean violation",
             _cell(n1f.get("violation") if n1f else None, "{:.2e}"),
             _cell(n1i.get("violation") if n1i else None, "{:.2e}")),
            ("B. linear residual  1−R²",
             _cell(n2f.get("residual_fraction") if n2f else None, "{:.2e}"),
             _cell(n2i.get("residual_fraction") if n2i else None, "{:.3g}")),
            ("C. amplitude-BLA  max drift",
             _cell(n3f.get("max_drift") if n3f else None, "{:.2e}"),
             _cell(n3i.get("max_drift") if n3i else None, "{:.3g}")),
            ("verdict",
             "LINEAR" if (n1f and n1f.get("linear")) else "—",
             "NONLINEAR" if (n1i and not n1i.get("linear")) else "—"),
        ]

        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        ax.axis("off")
        tbl = ax.table(cellText=rows, colLabels=["nonlinearity metric", "field (E)", "readout |E|²"],
                       colWidths=[0.5, 0.25, 0.25], loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.7)
        # colour the verdict row
        for c in (1, 2):
            cell = tbl[len(rows), c]
            txt = cell.get_text().get_text()
            cell.set_facecolor("#d6f5d6" if txt == "LINEAR" else "#f8d6d6" if txt == "NONLINEAR" else "white")
        ax.set_title(f"Nonlinearity (A superposition + B residual + C amplitude-BLA) — "
                     f"{os.path.basename(self.path)}", fontsize=10)
        fig.tight_layout()
        if save:
            os.makedirs(self.figdir, exist_ok=True)
            out = os.path.join(self.figdir, "nonlinear_stats.png")
            fig.savefig(out, dpi=130, bbox_inches="tight")
            print(f"[plot] saved {out}")
        return fig
