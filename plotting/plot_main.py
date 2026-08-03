"""Single-entry plotting driver — find available data, characterize, plot everything.

  python plotting/plot_main.py --path data/lasing_testing/02_adding_pump
  python plotting/plot_main.py --path data/lasing_testing/02_adding_pump --skip-cached
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from characterization.class_reservoir_validator import Validator
from plot_function_n1_superposition import plot_n1_superposition
from plot_function_n2_linear_residual import plot_n2_linear_residual
from plot_function_n3_amplitude_dependance import plot_n3_amplitude_dependance
from plot_function_n4_harmonics_distortion import plot_n4_harmonics_distortion
from plot_function_n5_volterra_series import plot_n5_volterra_series
from plot_function_n6_dambre_ipc import plot_n6_dambre_ipc
from plot_function_n7_dimension_expansion import plot_n7_dimension_expansion
from plot_single_source_sweep import plot_single_source_amplitude_sweep


class PlotMain:
    """Discover datasets (local → Orion fallback), run Validator, plot all results."""

    def __init__(self, reservoir_path: str, fig_dir: str | None = None,
                 skip_cached: bool = False, component: str | None = None,
                 max_order: int = 6, rel_thresh: float = 1e-9,
                 suffix: str = ""):
        self.path = reservoir_path
        self.fig_dir = Path(fig_dir) if fig_dir else Path(reservoir_path) / "figures"
        self.skip_cached = skip_cached
        # dataset-variant suffix ("_a10"): the validator resolves every
        # canonical name as <stem><suffix>.npz, and every figure/stats file
        # carries the same suffix, so variants never overwrite the originals.
        self.suffix = str(suffix or "")
        self.validator = Validator(reservoir_path, component=component,
                                   max_order=max_order, rel_thresh=rel_thresh,
                                   suffix=self.suffix)
        # polarization-slice tag ("_ExEy"): rides on every figure/stats name so
        # e.g. n4_harmonics_distortion_a10_ExEy_* sits next to the full-vector
        # figures instead of overwriting them.
        self.suffix += self.validator.comp_tag
        self.saved: list[Path] = []

    # ------------------------------------------------------------------- dispatch
    def run(self) -> list[Path]:
        """Run characterization (or load cache) then plot every available result."""
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.saved = []

        # --- characterization ---
        if self.skip_cached:
            import shutil
            shutil.rmtree(self.validator.stats_dir, ignore_errors=True)
        self.validator.run_all()
        R = self.validator.results
        if not R:
            print("[plot_main] no results — nothing to plot", flush=True)
            return []

        ds_tag = self._dataset_tag()

        # --- MODES plots (inline — no standalone plot_function_m*.py yet) ---
        self._plot_modes_svd(ds_tag)
        self._plot_modes_pca(ds_tag)

        # --- n1–n7 plots (one figure per method, field and intensity) ---
        plot_map = [
            ("n1_field",       plot_n1_superposition,        "n1_field"),
            ("n1_intensity",   plot_n1_superposition,        "n1_intensity"),
            ("n2_field",       plot_n2_linear_residual,      "n2_field"),
            ("n2_intensity",   plot_n2_linear_residual,      "n2_intensity"),
            ("n2",             plot_n2_linear_residual,      "n2"),
            ("n3_field",       plot_n3_amplitude_dependance, "n3_field"),
            ("n3_intensity",   plot_n3_amplitude_dependance, "n3_intensity"),
            ("n4_field",       plot_n4_harmonics_distortion, "n4_field"),
            ("n4_intensity",   plot_n4_harmonics_distortion, "n4_intensity"),
            ("n5_field",       plot_n5_volterra_series,      "n5_field"),
            ("n5_intensity",   plot_n5_volterra_series,      "n5_intensity"),
            ("n5",             plot_n5_volterra_series,      "n5"),
            ("n6_field",       plot_n6_dambre_ipc,           "n6_field"),
            ("n6_intensity",   plot_n6_dambre_ipc,           "n6_intensity"),
            ("n6",             plot_n6_dambre_ipc,           "n6"),
            ("n7_field",       plot_n7_dimension_expansion,  "n7_field"),
            ("n7_intensity",   plot_n7_dimension_expansion,  "n7_intensity"),
            ("n7",             plot_n7_dimension_expansion,  "n7"),
        ]
        # the n4 plot grows a raw-sweep panel when handed the dataset itself
        harm_data = self.validator._load("harmonics.npz")
        for key, fn, tag in plot_map:
            if key in R:
                try:
                    kw = ({"data": harm_data}
                          if key.startswith("n4") and harm_data is not None else {})
                    out = fn(R[key], self.fig_dir,
                             suffix=f"{self.suffix}_{tag}", **kw)
                    self.saved.append(out)
                    print(f"[plot_main] {out}", flush=True)
                except Exception as e:
                    print(f"[plot_main] {tag} FAILED: {e}", flush=True)

        # --- single-source amplitude sweep ---
        # Not a validator result (no R[...] key): single_source_sweep.py writes its
        # own dataset, so load and plot it directly. The _meep variant is the same
        # sweep run through MEEP, plotted alongside for cross-solver comparison.
        for name, tag in (("single_source_sweep.npz", "sweep"),
                          ("single_source_sweep_meep.npz", "sweep_meep")):
            sweep = self.validator._load(name)
            if sweep is None:
                continue
            try:
                out = plot_single_source_amplitude_sweep(
                    sweep, self.fig_dir, suffix=f"{self.suffix}_{tag}")
                self.saved.append(out)
                print(f"[plot_main] {out}", flush=True)
            except Exception as e:
                print(f"[plot_main] {tag} FAILED: {e}", flush=True)

        # --- combined summary ---
        try:
            out = self._plot_summary(ds_tag)
            self.saved.append(out)
            print(f"[plot_main] {out}", flush=True)
        except Exception as e:
            print(f"[plot_main] summary FAILED: {e}", flush=True)

        return self.saved

    # --------------------------------------------------------------- MODES plots
    def _plot_modes_svd(self, ds_tag: str):
        """SVD channel spectrum (capacity) from m1_bla (best linear approximation)."""
        R = self.validator.results
        bla = R.get("m1_bla")
        if bla is None:
            return
        s2 = bla.get("power")
        if s2 is None:
            s = bla.get("s")
            if s is None:
                return
            s2 = s ** 2
        s2 = np.asarray(s2)
        n_eff = bla.get("n_eff", len(s2))
        rank = bla.get("rank", len(s2))
        cond = bla.get("cond", float(s2[0]) / (float(s2[-1]) + 1e-30))

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.semilogy(np.arange(1, len(s2) + 1), s2 / (s2[0] + 1e-30) + 1e-30,
                    "o-", color="C0", markersize=3, lw=1.5)
        ax.axvline(n_eff, color="crimson", ls="--",
                   label=f"n_eff = {n_eff:.1f}  (rank={rank}, cond={cond:.1e})")
        ax.set_xlabel("channel index")
        ax.set_ylabel(r"$|s_i|^2 / |s_1|^2$")
        ax.set_title(f"SVD spectrum — linear coupling G  [{ds_tag}]")
        ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
        ax.set_ylim(1e-30, 2)
        fig.tight_layout()
        out = self.fig_dir / f"m1_svd_capacity_{ds_tag}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        self.saved.append(out)

    def _plot_modes_pca(self, ds_tag: str):
        """PCA spectrum of the output covariance (m2) — nonlinear dimension expansion."""
        R = self.validator.results
        pca = R.get("m2_pca")
        if pca is None:
            return
        evr = np.asarray(pca.get("explained_var_ratio", []))
        if len(evr) == 0:
            return
        n_eff = pca.get("n_eff", len(evr))
        expansion = pca.get("expansion_ratio", None)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.semilogy(np.arange(1, len(evr) + 1), evr + 1e-30,
                    "o-", color="C1", markersize=3, lw=1.5)
        ax.axvline(n_eff, color="crimson", ls="--",
                   label=f"n_eff = {n_eff:.1f}"
                   + (f"  expansion = {expansion:.2f}" if expansion is not None else ""))
        ax.set_xlabel("principal component")
        ax.set_ylabel("explained variance ratio")
        ax.set_title(f"PCA spectrum — output covariance  [{ds_tag}]")
        ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
        ax.set_ylim(1e-30, 1)
        fig.tight_layout()
        out = self.fig_dir / f"m2_pca_spectrum_{ds_tag}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        self.saved.append(out)

    # ------------------------------------------------------------ summary figure
    def _plot_summary(self, ds_tag: str) -> Path:
        """One-pager: verdict table + nonlinearity overview (IPC, Volterra, harmonics)."""
        R = self.validator.results
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.axis("off")

        lines = [f"Reservoir characterization — {self.path}",
                 f"datasets: {self.validator.datasets}", ""]
        # verdicts
        verdict_keys = [
            ("n1_field",     "superposition (field)"),
            ("n1_intensity", "superposition (|E|²)"),
            ("n2_field",     "linear residual (field)"),
            ("n2_intensity", "linear residual (|E|²)"),
            ("n2",           "linear residual"),
            ("n3_field",     "amplitude BLA (field)"),
            ("n3_intensity", "amplitude BLA (|E|²)"),
            ("n4_field",     "harmonics (field)"),
            ("n4_intensity", "harmonics (|E|²)"),
            ("n5_field",     "Volterra (field)"),
            ("n5_intensity", "Volterra (|E|²)"),
            ("n5",           "Volterra"),
            ("n6",           "Dambre IPC"),
            ("n7_field",     "dim. expansion (field)"),
            ("n7_intensity", "dim. expansion (|E|²)"),
            ("n7",           "dim. expansion"),
        ]
        for key, label in verdict_keys:
            v = R.get(key)
            if v is None:
                continue
            linear = v.get("linear", None)
            if linear is None:
                lines.append(f"  {label}: ran (no linear flag)")
                continue
            icon = "LINEAR" if linear else "NONLINEAR"
            detail = ""
            if "violation" in v:
                detail = f"  violation={v['violation']:.2e}"
            elif "residual_fraction" in v:
                detail = f"  1-R²={v['residual_fraction']:.2e}"
            elif "drift" in v and len(v.get("drift", [])) > 1:
                detail = f"  max drift={max(v['drift']):.2e}"
            elif "thd" in v:
                detail = f"  THD={v['thd']:.2e}  IMD={v['imd']:.2e}"
            elif "ipc_total" in v:
                detail = f"  IPC={v['ipc_total']:.3f}  nl frac={v.get('nonlinear_fraction', 0):.3f}"
            elif "expansion_ratio" in v:
                detail = f"  expansion={v.get('expansion_ratio', 0):.2f}"
            elif "plateau_r2" in v:
                detail = f"  plateau R²={v['plateau_r2']:.4f}"
            lines.append(f"  {icon:>10s}  {label}{detail}")

        # MODES summary
        bla = R.get("m1_bla")
        if bla:
            lines.append("")
            lines.append(f"MODES: n_eff={bla.get('n_eff', '?'):.1f}  "
                        f"rank={bla.get('rank', '?')}  "
                        f"cond={bla.get('cond', float('nan')):.1e}")
        pca = R.get("m2_pca")
        if pca:
            lines.append(f"PCA:   n_eff={pca.get('n_eff', '?'):.1f}  "
                        f"expansion={pca.get('expansion_ratio', '?'):.2f}")

        text = "\n".join(lines)
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontfamily="monospace",
                fontsize=8, va="top", linespacing=1.3)

        out = self.fig_dir / f"summary_{ds_tag}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return out

    # ------------------------------------------------------------------- helpers
    def _dataset_tag(self) -> str:
        """Short tag from the datasets path for filenames. The dataset-variant
        suffix rides along so e.g. summary_05_adding_mirror_a10.png sits next
        to the drive-1 summary instead of replacing it."""
        ds = str(self.validator.datasets)
        # e.g. ".../data/lasing_testing/02_adding_pump/datasets" → "02_adding_pump"
        parts = ds.replace("/datasets", "").rstrip("/").split("/")
        return (parts[-1] if parts else "unknown") + self.suffix


# ====================================================================== __main__
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Characterize + plot all available results")
    ap.add_argument("--path", required=True, help="reservoir dir (has simulation_data.json)")
    ap.add_argument("--fig-dir", default=None, help="output directory for figures (default: <path>/figures)")
    ap.add_argument("--component", "--components", default=None,
                    help="restrict analysis to a polarization subset (e.g. Ey, "
                         "or Ex,Ey to drop the pump-fed Ez channel); figures and "
                         "stats caches carry a _ExEy-style tag, so sliced runs "
                         "live next to the full-vector ones")
    ap.add_argument("--max-order", type=int, default=6,
                    help="harmonic/intermod attribution depth for n4: bins are "
                         "labelled as a*f1+b*f2 up to |a|+|b| <= this (default 6); "
                         "lines beyond it fall into the 'other' class")
    ap.add_argument("--rel-thresh", type=float, default=1e-9,
                    help="n4: spectral bins below this fraction of total power "
                         "count as numerical zero (default 1e-9)")
    ap.add_argument("--skip-cached", action="store_true",
                    help="re-run all analyses, ignore cached stats_data/")
    ap.add_argument("--suffix", default="",
                    help="dataset-variant suffix, e.g. _a10: every canonical "
                         "dataset resolves as <stem><suffix>.npz (harmonics_a10"
                         ".npz, ipc_a10.npz, ...) and figures/stats carry the "
                         "same suffix, so drive variants live side by side")
    args = ap.parse_args()
    pm = PlotMain(args.path, fig_dir=args.fig_dir, skip_cached=args.skip_cached,
                  component=args.component, max_order=args.max_order,
                  rel_thresh=args.rel_thresh, suffix=args.suffix)
    saved = pm.run()
    print(f"\n[done] {len(saved)} figures saved to {pm.fig_dir}", flush=True)
