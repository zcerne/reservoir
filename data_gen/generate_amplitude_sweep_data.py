"""Generate amplitude-swept probe data — Nonlinearity Method C (amplitude-dependent BLA).

Re-fitting the best-linear map G at several drive levels reveals nonlinearity: a
linear system's G is amplitude-independent, so any drift of G with drive level is
nonlinearity (and tells you at what amplitude it turns on).

We draw M random UNIT input directions ONCE, then at each amplitude level ℓ scale
them by `levels[ℓ]` and forward-run the reservoir — so the per-level BLA fits share
the same input directions (apples-to-apples). Total = L·M sims.

  python data_gen/generate_amplitude_sweep_data.py --path data/test2D \
      --levels 0.1,0.3,1,3,10 --n_probes 12 --out data/test2D/amp_sweep.npz

Then:  from n3_amplitude_dependant import amplitude_dependance, report
        d = dict(np.load("data/test2D/amp_sweep.npz", allow_pickle=True))
        print(report(amplitude_dependance(d)))                 # field: no drift
        d["outputs"] = np.abs(d["outputs"])**2                 # |E|² readout
        print(report(amplitude_dependance(d)))                 # G drifts with drive
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="reservoir design dir (simulation_data.json + relaxed LC)")
    ap.add_argument("--out", default=None, help="output npz (default <path>/amp_sweep.npz)")
    ap.add_argument("--levels", default="0.1,0.3,1,3,10", help="drive amplitude levels, comma-sep")
    ap.add_argument("--n_probes", type=int, default=12, help="M random input directions (shared across levels)")
    ap.add_argument("--components", default="Ey", help="sensor components to save (Ey[,Ex,Ez])")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from class_simulation_T import SimulationT
    try:
        import meep as mp
        is_master = bool(mp.am_master())
    except Exception:
        is_master = True
    comps = [c.strip() for c in args.components.split(",") if c.strip()]
    levels = np.array([float(x) for x in args.levels.split(",")], dtype=float)
    out_path = args.out or os.path.join(args.path, "amp_sweep.npz")

    sim = SimulationT(os.path.join(args.path, "simulation_data.json"))
    sim._set_data()
    src_key = sim._source_key(sim.args)
    amp0 = sim.args[src_key].get("amplitude", [1.0])
    n_strips = len(amp0) if isinstance(amp0, (list, tuple)) else 1
    print(f"[ampdata] reservoir={args.path}  n_strips={n_strips}  levels={list(levels)}  "
          f"n_probes={args.n_probes}  comps={comps}", flush=True)

    def forward(E):
        Ey, Ex, Ez = sim._run_basis(list(E))
        fields = {"Ey": Ey, "Ex": Ex, "Ez": Ez}
        return np.concatenate([np.asarray(fields[c]).ravel() for c in comps])

    rng = np.random.default_rng(args.seed)
    # M shared unit input directions
    dirs = rng.normal(size=(args.n_probes, n_strips)) + 1j * rng.normal(size=(args.n_probes, n_strips))
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30)

    inputs, outputs, level_id = [], [], []
    for li, lv in enumerate(levels):
        for p in range(args.n_probes):
            E = lv * dirs[p]
            inputs.append(E); outputs.append(forward(E)); level_id.append(li)
            print(f"[ampdata] level {li+1}/{len(levels)} (amp {lv:g})  probe {p+1}/{args.n_probes}", flush=True)

    if is_master:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez(out_path,
                 inputs=np.stack(inputs), outputs=np.stack(outputs),
                 level_id=np.asarray(level_id), levels=levels,
                 components=np.asarray(comps), n_strips=n_strips)
        print(f"[ampdata] DONE → {out_path}  ({len(levels)*args.n_probes} sims)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
