"""Morning report for the 04 vs 03b nonlinearity/capacity campaign.

Analyses every dataset that has assembled so far, for both designs, using the
Ey component only (the signal polarization; Ez is the pump, Ex the other TE
projection). Missing datasets are reported as pending, never fatal.

  python scripts/campaign_report.py [--designs 04_adding_LC,03b_isotropic_ds]
"""
import os, sys, glob, argparse
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R, "characterization"), os.path.join(R, "plotting")]
import numpy as np

import n1_superposition as n1
import n3_amplitude_dependant as n3
import n4_harmonics_distortion as n4
import n6_dambre as n6
from plot_function_n4_harmonics_distortion import plot_n4_harmonics_distortion


def ey_only(arr, comps):
    """Slice the Ey block out of a (..., n_comp*n_pts) output array."""
    a = np.asarray(arr)
    n_pts = a.shape[-1] // len(comps)
    k = comps.index("Ey")
    return a[..., k * n_pts:(k + 1) * n_pts]


def comps_of(d):
    return [str(c) for c in np.asarray(d["components"]).reshape(-1)]


def load(path):
    return dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None


def do_harmonics(ds_dir, fig_dir, tag):
    out = []
    for f in sorted(glob.glob(os.path.join(ds_dir, "harmonics*.npz"))):
        if "analysis" in os.path.basename(f):
            continue
        d = load(f)
        if d is None or "outputs" not in d:
            continue
        c = comps_of(d)
        d = dict(d)
        d["outputs"] = ey_only(d["outputs"], c)
        res = n4.harmonic_specter(d)
        amp = float(np.asarray(d["amps"]).reshape(-1)[0])
        by_order = res["power_by_order"]
        odd = sum(p for o, p in by_order.items() if o % 2 == 1 and o > 1)
        even = sum(p for o, p in by_order.items() if o % 2 == 0 and o > 0)
        fund = by_order.get(1, 0.0)
        out.append(dict(amp=amp, fund=fund, order3=by_order.get(3, 0.0),
                        order5=by_order.get(5, 0.0), odd_nl=odd, even_nl=even,
                        thd=res["thd"], distortion=res["distortion_frac"],
                        max_order=res["max_order"], file=os.path.basename(f)))
        try:
            plot_n4_harmonics_distortion(res, fig_dir,
                                         suffix=f"_{tag}_Ey_amp{amp:g}", data=d)
        except Exception as e:
            print(f"    [plot failed: {e}]")
    return sorted(out, key=lambda r: r["amp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs",
                    default="04_adding_LC,03b_isotropic_ds")
    a = ap.parse_args()

    for des in [x.strip() for x in a.designs.split(",") if x.strip()]:
        base = os.path.join(R, "data/lasing_testing", des)
        ds, figs = os.path.join(base, "datasets"), os.path.join(base, "figures")
        print("=" * 72)
        print(des)
        print("=" * 72)
        if not os.path.isdir(ds):
            print("  no datasets dir yet")
            continue

        rows = do_harmonics(ds, figs, des.split("_")[0])
        if rows:
            print("\n-- harmonics (Method D, Ey): drive-amplitude series")
            print(f"  {'amp':>6} {'fund':>11} {'order3':>11} {'order5':>11} "
                  f"{'even NL':>11} {'ord3/fund':>10} {'THD':>9}")
            for r in rows:
                print(f"  {r['amp']:6g} {r['fund']:11.4g} {r['order3']:11.4g} "
                      f"{r['order5']:11.4g} {r['even_nl']:11.4g} "
                      f"{r['order3']/max(r['fund'],1e-300):10.3e} {r['thd']:9.3g}")
            print("  (even-order power should be ~0: odd-only = centrosymmetric "
                  "saturable nonlinearity)")
        else:
            print("\n-- harmonics: pending")

        d = load(os.path.join(ds, "amp_sweep.npz"))
        if d is not None:
            c = comps_of(d)
            d = dict(d); d["outputs"] = ey_only(d["outputs"], c)
            res = n3.amplitude_dependance(d)
            lv = np.asarray(d["levels"], float).reshape(-1)
            lid = np.asarray(d["level_id"]).reshape(-1)
            print("\n-- amplitude sweep (Method C, Ey)")
            print(f"  {'level':>7} {'n':>3} {'|out|/level':>13} {'rel':>8} "
                  f"{'BLA drift':>11}")
            b = None
            for i, L in enumerate(lv):
                m = lid == i
                if not m.any():
                    print(f"  {L:7g}   -  (missing)")
                    continue
                g = float(np.linalg.norm(np.abs(d["outputs"][m]))
                          / np.sqrt(m.sum()) / L)
                b = g if b is None else b
                print(f"  {L:7g} {m.sum():3d} {g:13.5g} {g/b:8.4f} "
                      f"{res['drift'][i]:11.4g}")
        else:
            print("\n-- amplitude sweep: pending")

        d = load(os.path.join(ds, "superposition.npz"))
        if d is not None:
            c = comps_of(d)
            d = dict(d)
            for k in ("out1", "out2", "out_combo"):
                d[k] = ey_only(d[k], c)
            print("\n-- superposition (Method A, Ey)")
            print("  " + n1.report(n1.super_position_test(d)).replace("\n", "\n  "))
        else:
            print("\n-- superposition: pending")

        d = load(os.path.join(ds, "ipc.npz"))
        if d is not None:
            c = comps_of(d)
            d = dict(d); d["outputs"] = ey_only(d["outputs"], c)
            print(f"\n-- IPC / capacity (Method F, Ey): "
                  f"{d['outputs'].shape[0]} samples x {d['outputs'].shape[1]} features")
            try:
                print("  " + n6.report(n6.dambre_ipc(d)).replace("\n", "\n  "))
            except Exception as e:
                print(f"  FAILED: {e}")
        else:
            n_parts = len(glob.glob(os.path.join(ds, "ipc.npz.parts", "part_*.npz")))
            print(f"\n-- IPC / capacity: pending ({n_parts}/400 parts)")
    print()


if __name__ == "__main__":
    main()
