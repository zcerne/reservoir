"""Pump-threshold sweep: one GPU forward run per thr_* design, saving the
output-spectrum energy at the emission line vs pump amplitude.

Usage:  python data_gen/run_threshold_sweep.py --designs thr_s006_p50,...
Writes <design>/simulation/threshold_point.npz {pump, sgma, out_energy,
out_peak, spec} from monitor_2 (n_lam frequencies over the emission band).
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import _gen_common as gc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", required=True)
    ap.add_argument("--base", default="data/reservoir_clasifications")
    args = ap.parse_args()
    for name in args.designs.split(","):
        path = os.path.join(args.base, name)
        out = os.path.join(path, "simulation", "threshold_point.npz")
        if os.path.exists(out):
            print(f"[thr] {name}: exists, skip", flush=True)
            continue
        j = json.load(open(os.path.join(path, "simulation_data.json")))
        forward, n_strips, _ = gc.open_reservoir(path, ["Ey"])
        amp = j["source_1"]["amplitude"]
        y = forward(np.asarray(amp, dtype=float))
        m2 = np.load(os.path.join(path, "simulation", "monitor_2.npz"))
        Ey = m2["Ey"]                       # (n_lam, ny)
        spec = np.sqrt((np.abs(Ey) ** 2).sum(axis=1))
        np.savez(out, pump=float(j["source_2"]["amplitude"][0]),
                 sgma=float(j["reservoir"]["sted"]["SGMA"]),
                 out_energy=float((np.abs(Ey) ** 2).sum()),
                 out_peak=float(spec.max()), spec=spec, freqs=m2["freqs"])
        print(f"[thr] {name}: pump={j['source_2']['amplitude'][0]} "
              f"energy={float((np.abs(Ey)**2).sum()):.6g}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
