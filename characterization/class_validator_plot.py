import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # headless (smaug/Orion/lips have no $DISPLAY)
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
        if "n4_field" not in self.results:
            self.harmonics()
        if not any(k.startswith("n5") for k in self.results):
            self.volterra()
        # n6 (Dambre IPC) is only shown if ALREADY computed — it's intractably slow
        # for many-input reservoirs (196-input MNIST nets), so don't force it here;
        # the IPC table cells fall back to n/a when absent. Compute it explicitly
        # (Validator.dambre / plot_characteristics --ipc) for few-input reservoirs.
        if not any(k.startswith("n7") for k in self.results):
            self.dimension_expansion()
        R = self.results

        def _cell(x, fmt="{:.3g}"):
            return "n/a" if x is None else fmt.format(x)

        # n2 may be split (complex ipc → n2_field/n2_intensity) or single (intensity ipc → n2)
        n2f = R.get("n2_field"); n2i = R.get("n2_intensity") or R.get("n2")
        n1f = R.get("n1_field"); n1i = R.get("n1_intensity")
        n3f = R.get("n3_field"); n3i = R.get("n3_intensity")
        n4f = R.get("n4_field"); n4i = R.get("n4_intensity")
        n5f = R.get("n5_field"); n5i = R.get("n5_intensity") or R.get("n5")
        n6 = R.get("n6")
        n7f = R.get("n7_field"); n7i = R.get("n7_intensity") or R.get("n7")

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
            ("D. harmonics  THD / distortion frac",
             _cell(n4f.get("thd") if n4f else None, "{:.3f}") + " / " +
             _cell(n4f.get("distortion_frac") if n4f else None, "{:.3f}"),
             _cell(n4i.get("thd") if n4i else None, "{:.3f}") + " / " +
             _cell(n4i.get("distortion_frac") if n4i else None, "{:.3f}")),
            ("E. Volterra  nonlinear frac (order≥2)",
             _cell(n5f.get("nonlinear_fraction") if n5f else None, "{:.3f}"),
             _cell(n5i.get("nonlinear_fraction") if n5i else None, "{:.3f}")),
            ("F. Dambre IPC  total / ceiling",
             "n/a",
             _cell(n6.get("ipc_total") if n6 else None, "{:.1f}") + " / " +
             _cell(n6.get("bound") if n6 else None, "{:.0f}")),
            ("F. IPC  nonlinear frac (deg≥2)",
             "n/a",
             _cell(n6.get("nonlinear_fraction") if n6 else None, "{:.3f}")),
            ("G. dim-expansion  PR / d99",
             _cell(n7f.get("pr") if n7f else None, "{:.1f}") + " / " +
             _cell(n7f.get("d99") if n7f else None, "{:.0f}"),
             _cell(n7i.get("pr") if n7i else None, "{:.1f}") + " / " +
             _cell(n7i.get("d99") if n7i else None, "{:.0f}")),
            ("G. dim-expansion  plateau R²",
             _cell(n7f.get("plateau_r2") if n7f else None, "{:.4f}"),
             _cell(n7i.get("plateau_r2") if n7i else None, "{:.4f}")),
            ("verdict",
             "LINEAR" if (n1f and n1f.get("linear")) else "—",
             "NONLINEAR" if (n1i and not n1i.get("linear")) else "—"),
        ]

        # 2×2 grid: stats table | D spectrum | E/F order bars | G expansion R²(k)
        fig, ((ax_tbl, ax_sp), (ax_ord, ax_exp)) = plt.subplots(2, 2, figsize=(16, 7),
                                                gridspec_kw={"width_ratios": [1.2, 1],
                                                             "height_ratios": [1, 1]})
        ax_tbl.axis("off")
        tbl = ax_tbl.table(cellText=rows, colLabels=["nonlinearity metric", "field (E)", "readout |E|²"],
                           colWidths=[0.55, 0.225, 0.225], loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.35)
        for c in (1, 2):                                   # colour the verdict row
            cell = tbl[len(rows), c]
            txt = cell.get_text().get_text()
            cell.set_facecolor("#d6f5d6" if txt == "LINEAR" else "#f8d6d6" if txt == "NONLINEAR" else "white")
        ax_tbl.set_title(f"Nonlinearity — {os.path.basename(self.path)}", fontsize=10,
                         fontweight="bold")

        self._plot_harmonic_spectrum(ax_sp)                 # D
        self._plot_order_spectrum(ax_ord)                   # E + F
        self._plot_expansion(ax_exp)                         # G

        fig.tight_layout()
        if save:
            os.makedirs(self.figdir, exist_ok=True)
            out = os.path.join(self.figdir, "nonlinear_stats.png")
            fig.savefig(out, dpi=130, bbox_inches="tight")
            print(f"[plot] saved {out}")
        return fig

    def _plot_harmonic_spectrum(self, ax):
        """D. Harmonic spectrum: power-by-order bars from n4.harmonic_specter() —
        field (only order 1) vs |E|² (DC + harmonics + intermod at higher orders)."""
        n4f = self.results.get("n4_field"); n4i = self.results.get("n4_intensity")
        if n4f is None and n4i is None:
            ax.text(0.5, 0.5, "no harmonics.npz\n(run n4 data gen)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_xticks([]); ax.set_yticks([]); return
        orders = sorted(set(list((n4f or {}).get("power_by_order", {}).keys()) +
                            list((n4i or {}).get("power_by_order", {}).keys())))
        orders = [o for o in orders if o >= 0] or list(range(6))
        x = np.arange(len(orders)); w = 0.35
        def _norm(d, key):
            po = d.get(key, {}) if d else {}
            tot = sum(po.values()) or 1.0
            return [po.get(o, 0.0) / tot for o in orders]
        if n4f:
            ax.bar(x - w/2, _norm(n4f, "power_by_order"), w, color="C0", label="field (E)")
        if n4i:
            ax.bar(x + w/2, _norm(n4i, "power_by_order"), w, color="C3", label="|E|²")
        ax.set_xticks(x); ax.set_xticklabels([str(o) for o in orders])
        ax.set_xlabel("harmonic order"); ax.set_ylabel("fraction of total power")
        tones = list(map(int, (n4i or n4f).get("tones", [])))
        thd_i = (n4i or {}).get("thd", 0); df_i = (n4i or {}).get("distortion_frac", 0)
        ax.set_title(f"D. harmonic_specter — tones {tones}  |E|² THD={thd_i:.3f} distort={df_i:.3f}",
                     fontsize=10)
        ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    def _plot_order_spectrum(self, ax):
        """E. Volterra variance-explained by polynomial order + F. Dambre IPC capacity
        by degree — grouped bars vs order/degree. Reads from self.results (output of
        n5.volterra_series + n6.dambre_ipc). A pure |E|² system puts all the nonlinear
        weight at order/degree 2."""
        R = self.results
        n5 = R.get("n5_intensity") or R.get("n5")
        n6 = R.get("n6")
        if n5 is None and n6 is None:
            ax.text(0.5, 0.5, "no ipc.npz\n(run n5/n6 data gen)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_xticks([]); ax.set_yticks([]); return
        degs = sorted(set(list((n5.get("gain_by_order", {}) if n5 else {}).keys()) +
                          list((n6.get("ipc_by_degree", {}) if n6 else {}).keys())))
        degs = [d for d in degs if d >= 1] or [1, 2]
        x = np.arange(len(degs)); w = 0.4
        if n5:
            g = n5.get("gain_by_order", {})
            v = [max(g.get(d, 0.0), 0.0) for d in degs]
            ax.bar(x - w/2, v, w, color="C0", label="E. Volterra variance frac")
        if n6:
            ipc = n6.get("ipc_by_degree", {})
            tot = n6.get("ipc_total", 1.0) or 1.0
            v = [ipc.get(d, 0.0) / tot for d in degs]        # normalized IPC share
            ax.bar(x + w/2, v, w, color="C3", label="F. IPC capacity frac")
        ax.set_xticks(x); ax.set_xticklabels([str(d) for d in degs])
        ax.set_xlabel("polynomial order / degree"); ax.set_ylabel("fraction")
        ax.set_title("E. Volterra order + F. IPC degree", fontsize=10)
        ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    def _plot_expansion(self, ax):
        """G. Dimension expansion: R²(k) — linear-fit held-out R² vs input dimension k
        (from n7.dimension_expansion). Field: R²→1 at k=K. |E|²: plateaus ≪1 + higher
        PCA effective rank."""
        R = self.results
        n7f = R.get("n7_field"); n7i = R.get("n7_intensity") or R.get("n7")
        if n7f is None and n7i is None:
            ax.text(0.5, 0.5, "no ipc.npz\n(run n7 data gen)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_xticks([]); ax.set_yticks([]); return
        ks = sorted((n7f or n7i)["r2_vs_k"].keys())
        if n7f:
            r2f = [n7f["r2_vs_k"][k] for k in ks]
            ax.plot(ks, r2f, "C0o-", lw=2, ms=6, label="field R²(k)")
            ax.axhline(1.0, color="C0", ls=":", lw=0.8)
        if n7i:
            r2i = [n7i["r2_vs_k"][k] for k in ks]
            ax.plot(ks, r2i, "C3s--", lw=2, ms=6, label="|E|² R²(k)")
        ax.set_xticks(ks); ax.set_xlabel("input dimension k"); ax.set_ylabel("R² (linear fit)")
        txt = ""
        if n7f: txt += f"field PR={n7f['pr']:.1f} d99={n7f['d99']}"
        if n7i: txt += f"\n|E|² PR={n7i['pr']:.1f} d99={n7i['d99']}"
        ax.text(0.95, 0.05, txt, transform=ax.transAxes, fontsize=7.5, va="bottom",
                ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))
        ax.set_title(f"G. dimension expansion  "
                     f"{'LINEAR' if (n7f or {}).get('linear') else ''}"
                     f"{' | NONLINEAR' if n7i and not n7i.get('linear') else ''}",
                     fontsize=10)
        ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.15)
