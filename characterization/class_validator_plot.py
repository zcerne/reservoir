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
    def _readout_complex(self):
        """Is the reservoir readout a complex field (optical) or a real state (NN)?"""
        d = self._load("ipc.npz")
        return bool(d is not None and np.iscomplexobj(d.get("outputs")))

    def _ro(self, n):
        """Return the result dict for metric `n` for the physical readout:
        complex reservoir → the |E|² (intensity) analysis; real reservoir → the
        plain/real-state analysis. Handles the Validator's mixed key naming."""
        R = self.results
        if self._complex:
            return R.get(f"{n}_intensity") or R.get(n)
        return R.get(n) or R.get(f"{n}_field") or R.get(f"{n}_intensity")

    def plot_nonlinear_stats(self, save=True):
        """NONLINEARITY of the reservoir readout (what comes out for what goes in).
        ONE column = the physical output: the real state for an NN reservoir, |E|²
        (the detector reading) for an optical one. A. superposition, B. linear
        residual, C–G. amplitude/harmonics/Volterra/IPC/dim-expansion."""
        self._complex = self._readout_complex()
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
        def _cell(x, fmt="{:.3g}"):
            return "n/a" if x is None else fmt.format(x)

        # SINGLE readout column = "what actually comes out of the reservoir".
        #   real reservoir (NN)  → the real state itself.
        #   complex reservoir (optical) → |E|² (what the photodetector measures).
        # _ro(n) returns the right result dict for metric n regardless of which
        # naming the Validator used (real: nX / nX_field ; complex: nX_intensity).
        col_label = "|E|² readout (detector)" if self._complex else "reservoir output"

        r1 = self._ro("n1"); r2 = self._ro("n2"); r3 = self._ro("n3")
        r4 = self._ro("n4"); r5 = self._ro("n5"); r6 = self.results.get("n6")
        r7 = self._ro("n7")

        rows = [
            ("A. superposition  R²", _cell(r1.get("r2") if r1 else None, "{:.4f}")),
            ("A. superposition  mean violation", _cell(r1.get("violation") if r1 else None, "{:.2e}")),
            ("B. linear residual  1−R²", _cell(r2.get("residual_fraction") if r2 else None, "{:.3g}")),
            ("C. amplitude-BLA  max drift", _cell(r3.get("max_drift") if r3 else None, "{:.3g}")),
            ("D. harmonics  THD / distortion frac",
             _cell(r4.get("thd") if r4 else None, "{:.3f}") + " / " +
             _cell(r4.get("distortion_frac") if r4 else None, "{:.3f}")),
            ("E. Volterra  nonlinear frac (order≥2)",
             _cell(r5.get("nonlinear_fraction") if r5 else None, "{:.3f}")),
            ("F. Dambre IPC  total / ceiling",
             _cell(r6.get("ipc_total") if r6 else None, "{:.1f}") + " / " +
             _cell(r6.get("bound") if r6 else None, "{:.0f}")),
            ("F. IPC  nonlinear frac (deg≥2)",
             _cell(r6.get("nonlinear_fraction") if r6 else None, "{:.3f}")),
            ("G. dim-expansion  PR / d99",
             _cell(r7.get("pr") if r7 else None, "{:.1f}") + " / " +
             _cell(r7.get("d99") if r7 else None, "{:.0f}")),
            ("G. dim-expansion  plateau R²", _cell(r7.get("plateau_r2") if r7 else None, "{:.4f}")),
            ("verdict", "NONLINEAR" if (r1 and not r1.get("linear")) else
                        "LINEAR" if (r1 and r1.get("linear")) else "—"),
        ]

        # 2×2 grid: stats table | D spectrum | E/F order bars | G expansion R²(k)
        fig, ((ax_tbl, ax_sp), (ax_ord, ax_exp)) = plt.subplots(2, 2, figsize=(15, 7),
                                                gridspec_kw={"width_ratios": [1.15, 1],
                                                             "height_ratios": [1, 1]})
        ax_tbl.axis("off")
        tbl = ax_tbl.table(cellText=rows, colLabels=["nonlinearity metric", col_label],
                           colWidths=[0.62, 0.38], loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
        cell = tbl[len(rows), 1]                            # colour the verdict cell
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
        """D. Harmonic spectrum of the reservoir readout: fraction of output power by
        harmonic order (0=DC, 1=fundamentals=linear, ≥2=nonlinear harmonics/intermod)."""
        n4 = self._ro("n4")
        if n4 is None:
            ax.text(0.5, 0.5, "no harmonics.npz\n(run n4 data gen)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_xticks([]); ax.set_yticks([]); return
        orders = [o for o in sorted(n4.get("power_by_order", {}).keys()) if o >= 0] or list(range(6))
        po = n4.get("power_by_order", {}); tot = sum(po.values()) or 1.0
        x = np.arange(len(orders))
        vals = [po.get(o, 0.0) / tot for o in orders]
        ax.bar(x, vals, 0.6, color="C0")
        ax.set_xticks(x); ax.set_xticklabels([str(o) for o in orders])
        ax.set_xlabel("harmonic order"); ax.set_ylabel("fraction of total power")
        tones = list(map(int, n4.get("tones", [])))
        ax.set_title(f"D. harmonic_specter — tones {tones}  "
                     f"THD={n4.get('thd', 0):.3f} distort={n4.get('distortion_frac', 0):.3f}",
                     fontsize=10)
        ax.set_ylim(0, 1.05)

    def _plot_order_spectrum(self, ax):
        """E. Volterra variance-explained by polynomial order + F. Dambre IPC capacity
        by degree — grouped bars vs order/degree. Reads from self.results (output of
        n5.volterra_series + n6.dambre_ipc). A pure |E|² system puts all the nonlinear
        weight at order/degree 2."""
        n5 = self._ro("n5")
        n6 = self.results.get("n6")
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
        n7 = self._ro("n7")
        if n7 is None:
            ax.text(0.5, 0.5, "no ipc.npz\n(run n7 data gen)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_xticks([]); ax.set_yticks([]); return
        ks = sorted(n7["r2_vs_k"].keys())
        r2 = [n7["r2_vs_k"][k] for k in ks]
        ax.plot(ks, r2, "C0o-", lw=2, ms=5, label="R²(k) linear fit")
        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        step = max(1, len(ks) // 12)                        # avoid tick overcrowding for many-input nets
        ax.set_xticks(ks[::step]); ax.set_xlabel("input dimension k"); ax.set_ylabel("R² (linear fit)")
        ax.text(0.95, 0.05, f"PR={n7['pr']:.1f}  d99={n7['d99']}", transform=ax.transAxes,
                fontsize=8, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))
        ax.set_title(f"G. dimension expansion  "
                     f"{'LINEAR' if n7.get('linear') else 'NONLINEAR'}", fontsize=10)
        ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.15)
