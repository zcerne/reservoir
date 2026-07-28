"""Plot + save all reservoir-characterization figures (MODES + n1–n7).
Delegates to plotting.plot_main.PlotMain — kept as a convenience entry point.

  python plot_characteristics.py --path data/lasing_testing/02_adding_pump
"""
from __future__ import annotations
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Characterize + plot all results (delegates to PlotMain)")
    ap.add_argument("--path", required=True, help="reservoir design dir (repo-relative or absolute)")
    ap.add_argument("--out", default=None, help="output dir for figures (default: <path>/figures)")
    ap.add_argument("--skip-cached", action="store_true",
                    help="re-run all analyses, ignore cached stats_data/")
    args = ap.parse_args()

    from plotting.plot_main import PlotMain
    pm = PlotMain(args.path, fig_dir=args.out, skip_cached=args.skip_cached)
    saved = pm.run()
    R = pm.validator.results
    # brief scalar summary
    m1 = R.get("m1_bla")
    if m1 is not None:
        print(f"  [MODES] rank={m1['rank']} n_eff={m1['n_eff']:.3f} "
              f"f_in/f_out={m1['f_in']}/{m1['f_out']}", flush=True)
    n6 = R.get("n6")
    if isinstance(n6, dict):
        print(f"  [IPC] total={n6.get('ipc_total', float('nan')):.3f} "
              f"nonlinear_fraction={n6.get('nonlinear_fraction', float('nan')):.3f}", flush=True)
    print(f"\n[done] {len(saved)} figures saved to {pm.fig_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
