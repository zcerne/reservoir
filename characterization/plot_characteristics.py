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


ORION_DATA = "/home/ziga/Orion/resevoir/data"     # datasets live here, not in the git repo


def _resolve_path(path, orion_root):
    """The heavy datasets live on the Orion mount, not in the git repo (they're
    gitignored). If the given path has no datasets, remap it onto the Orion data
    root by its `data/`-relative suffix so a convenient repo-relative path works."""
    path = os.path.abspath(os.path.expanduser(path))
    if _has_data(path):
        return path
    # find "…/data/<suffix>" and rebuild as <orion_root>/<suffix>
    parts = path.replace(os.sep, "/").split("/data/")
    if len(parts) >= 2:
        cand = os.path.join(orion_root, parts[-1])
        if _has_data(cand):
            print(f"[characteristics] local path has no datasets → using Orion: {cand}", flush=True)
            return cand
    return path                                    # fall through; caller reports missing


def _has_data(path):
    """True only if datasets/ holds a real .npz file or a NON-EMPTY .parts dir
    (ignore .gitkeep and the empty mirrored .parts dirs the repo carries)."""
    ds = os.path.join(path, "datasets")
    if not os.path.isdir(ds):
        return False
    for f in os.listdir(ds):
        if f.endswith(".npz"):
            return True
        if f.endswith(".parts"):
            pp = os.path.join(ds, f)
            if os.path.isdir(pp) and any(x.endswith(".npz") for x in os.listdir(pp)):
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="reservoir design dir (repo-relative ok — auto-resolves to Orion)")
    ap.add_argument("--out", default=None,
                    help="LOCAL output dir for the figures (default: "
                         "~/reservoir_figs/<design-name> on workbox)")
    ap.add_argument("--orion-root", default=ORION_DATA, help="Orion data root to resolve against")
    ap.add_argument("--ipc", action="store_true",
                    help="also compute the Dambre IPC (n6) — SLOW for many-input reservoirs")
    args = ap.parse_args()

    from class_validator_plot import PlotValidator

    # DATA is read from Orion (resolved); FIGURES are written next to the design in
    # the NEXTCLOUD repo path the user gave (that folder is Nextcloud-synced), so the
    # figures land at reservoir_clasifications/<design>/figures on Nextcloud, not Orion.
    repo_path = os.path.abspath(os.path.expanduser(args.path))
    data_path = _resolve_path(args.path, args.orion_root)
    out = os.path.abspath(os.path.expanduser(args.out)) if args.out \
        else os.path.join(repo_path, "figures")
    os.makedirs(out, exist_ok=True)

    v = PlotValidator(data_path)
    v.figdir = out                          # save figures to the Nextcloud repo dir

    print(f"[characteristics] data: {data_path}", flush=True)
    print(f"[characteristics] figures → {out}", flush=True)

    # Ensure the stats the FIGURES need exist (compute + cache if missing). The two
    # plots use MODES (capacity) + n1/n3/n4/n7 (nonlinearity spectra) — NOT the Dambre
    # IPC (n6), which is intractably slow when the reservoir has many inputs (e.g. the
    # 196-input MNIST nets: the polynomial target space explodes). So we skip n6 here
    # by default; add --ipc to also compute it (only feasible for few-input reservoirs).
    steps = [v.modes, v.superposition, v.linear_residual, v.amplitude,
             v.harmonics, v.volterra, v.dimension_expansion]
    if args.ipc:
        steps.append(v.dambre)
    if "m1_bla" not in v.results:
        print("[characteristics] computing stats (skipping Dambre IPC — pass --ipc to include it) …",
              flush=True)
        for step in steps:
            try:
                step()
            except Exception as e:
                print(f"[characteristics] {step.__name__} skipped: {e}", flush=True)

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
