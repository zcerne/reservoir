"""Plot + save a reservoir's capacity (MODES) and nonlinear-stats figures from its
design PATH. Reads the datasets from the given path (e.g. an Orion-mount design dir)
and writes the two PNGs to a LOCAL (workbox) output dir so the figures survive an
Orion unmount.

  python plot_characteristics.py --path /home/ziga/Orion/resevoir/data/reservoir_clasifications/01_2D_director

  # custom output location
  python plot_characteristics.py --path <design> --out ~/reservoir_figs/01_director

Saves:  <out>/capacity.png   (SVD spectrum of G + scalar capacity table)
        <out>/nonlinear_stats.png  (n1–n7 nonlinearity: harmonics, order spectrum, expansion)
"""
from __future__ import annotations
import argparse, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="reservoir design dir (has datasets/)")
    ap.add_argument("--out", default=None,
                    help="LOCAL output dir for the figures (default: "
                         "~/reservoir_figs/<design-name> on workbox)")
    args = ap.parse_args()

    from class_validator_plot import PlotValidator

    path = os.path.abspath(os.path.expanduser(args.path))
    name = os.path.basename(path.rstrip("/"))
    out = args.out or os.path.join(os.path.expanduser("~"), "reservoir_figs", name)
    out = os.path.abspath(os.path.expanduser(out))
    os.makedirs(out, exist_ok=True)

    v = PlotValidator(path)
    v.figdir = out                          # redirect saves to the LOCAL dir (not the mount)

    print(f"[characteristics] design: {path}", flush=True)
    print(f"[characteristics] figures → {out}", flush=True)

    # Ensure the stats exist. run_all() computes every analysis (modes + n1–n7) from
    # the datasets and CACHES them to <path>/stats_data/; it's a fast no-op when the
    # cache is already there, and skips any missing dataset gracefully. So if the raw
    # data was generated but the stats weren't, this generates them before plotting.
    if "m1_bla" not in v.results or "n6" not in v.results:
        print("[characteristics] stats not loaded — running full analysis "
              "(computes + caches to stats_data/) …", flush=True)
        v.run_all()

    cap = v.plot_capacity(save=True)
    print(f"  capacity.png {'✓' if cap is not None else '— skipped (no field data)'}", flush=True)
    nl = v.plot_nonlinear_stats(save=True)
    print(f"  nonlinear_stats.png {'✓' if nl is not None else '— skipped (no data)'}", flush=True)

    # brief scalar summary to stdout
    m1 = v.results.get("m1_bla")
    if m1 is not None:
        print(f"  [MODES] rank={m1['rank']} n_eff={m1['n_eff']:.3f} "
              f"throughput={m1['throughput']:.4g} f_in/f_out={m1['f_in']}/{m1['f_out']}", flush=True)
    n6 = v.results.get("n6")
    if isinstance(n6, dict):
        print(f"  [IPC] total={n6.get('ipc_total', float('nan')):.3f} "
              f"nonlinear_fraction={n6.get('nonlinear_fraction', float('nan')):.3f} "
              f"max_degree={n6.get('max_degree_present')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
