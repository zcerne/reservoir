"""Generate Dambre-IPC data — Nonlinearity Method F (gold standard).

The IPC target family is orthonormal Legendre polynomials of the input channels, so
inputs MUST be i.i.d. ~ Uniform[-1,1] per source strip. We draw M such real input
vectors, forward-run the reservoir once each, and save (inputs, outputs). The analysis
(`characterization/n6_dambre.dambre_ipc`) then measures the linear-readout capacity of
the reservoir output onto the whole polynomial family, bucketed by degree.

Use the |E|² intensity as the reservoir output (capacity is a property of the nonlinear
readout state) — the generator saves the complex field; square it before dambre_ipc,
or pass --readout intensity to save |E|² directly.

Need M ≫ F (output features) for clean capacities; the analysis thresholds at ~2F/M.

  python data_gen/generate_ipc_data.py --path data/test2D --n 400 \
      --readout intensity --out data/test2D/ipc.npz

Then:  from n6_dambre import dambre_ipc, report
        d = dict(np.load("data/test2D/ipc.npz"))
        print(report(dambre_ipc(d, max_degree=3)))
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="reservoir design dir (simulation_data.json + relaxed LC)")
    ap.add_argument("--out", default=None, help="output npz (default <path>/ipc.npz)")
    ap.add_argument("--n", type=int, default=400, help="#input probes (need ≫ #output features)")
    ap.add_argument("--readout", default="intensity", choices=["field", "intensity"],
                    help="save complex field or |E|² intensity (IPC uses the |E|² state)")
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
    out_path = args.out or os.path.join(args.path, "ipc.npz")

    sim = SimulationT(args.path)
    sim._set_data()
    src_key = sim._source_key(sim.args)
    amp0 = sim.args[src_key].get("amplitude", [1.0])
    n_strips = len(amp0) if isinstance(amp0, (list, tuple)) else 1
    print(f"[ipcdata] reservoir={args.path}  n_strips={n_strips}  n={args.n}  "
          f"readout={args.readout}  comps={comps}", flush=True)

    def forward(E):
        Ey, Ex, Ez = sim._run_basis(list(E))
        fields = {"Ey": Ey, "Ex": Ex, "Ez": Ez}
        v = np.concatenate([np.asarray(fields[c]).ravel() for c in comps])
        return (np.abs(v) ** 2) if args.readout == "intensity" else v

    rng = np.random.default_rng(args.seed)
    # i.i.d. Uniform[-1,1] per channel — required for Legendre orthonormality
    U = rng.uniform(-1.0, 1.0, size=(args.n, n_strips))

    outputs = []
    for m in range(args.n):
        outputs.append(forward(U[m].astype(complex)))
        if m % 25 == 0:
            print(f"[ipcdata] {m+1}/{args.n}", flush=True)

    if is_master:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez(out_path, inputs=U, outputs=np.stack(outputs),
                 readout=np.asarray(args.readout), components=np.asarray(comps), n_strips=n_strips)
        print(f"[ipcdata] DONE → {out_path}  ({args.n} sims, readout={args.readout})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
