"""Dambre-IPC data — Nonlinearity Method F (gold standard). Index/assemble + incremental.

Inputs MUST be i.i.d. ~ Uniform[-1,1] per source strip (Legendre orthonormality). M
such real input vectors, one forward run each. Work item m runs U[m]; each → part
(incremental). --assemble → <out>.npz {inputs, outputs} for n6_dambre.dambre_ipc.
Also serves n2 (residual) and n5 (Volterra). Need M ≫ #output features.

Parts ALWAYS store the raw complex field (lossless); `--readout intensity` applies
|E|² at ASSEMBLE time only — regenerate nothing to switch readouts.

  N=$(python data_gen/generate_ipc_data.py --path data/reservoir_clasifications/01_2D_director --n 400 --count)
  sbatch --array=0-$((N-1)) slurm_char_array.sh ipc data/reservoir_clasifications/01_2D_director --n 400 --readout intensity
  python data_gen/generate_ipc_data.py --path data/reservoir_clasifications/01_2D_director --n 400 --readout intensity --assemble
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import _gen_common as gc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--readout", default="field", choices=["field", "intensity"],
                    help="applied at ASSEMBLE only: raw complex field (default) or |E|²; "
                         "parts always store the field")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="physical amplitude = scale*u; u itself (Uniform[-1,1], what the "
                         "Legendre polynomial basis is evaluated on) is unaffected and is "
                         "what gets saved as 'inputs' -- lets the device operate at any real "
                         "drive level without breaking the [-1,1] orthonormality the capacity "
                         "decomposition (n6_dambre) assumes.")
    gc.add_extras_args(ap)
    gc.add_common_args(ap)
    args = ap.parse_args()

    comps = [c.strip() for c in args.components.split(",") if c.strip()]
    out_path = args.out or os.path.join(args.path, "datasets", "ipc.npz")
    n_items = args.n

    forward = n_strips = is_master = U = None
    if args.count or args.assemble:
        is_master = True
    else:
        forward, n_strips, is_master = gc.open_reservoir(
            args.path, comps, out_sensor=args.out_sensor,
            n_sources=args.n_sources, with_extras=args.extras,
            n2f_lam=args.n2f_lam)
        rng = np.random.default_rng(args.seed)
        U = rng.uniform(-1.0, 1.0, size=(args.n, n_strips))   # i.i.d. Uniform[-1,1] (real)

    def run_one(m):
        if getattr(args, "skip_existing", False):
            part = os.path.join(gc._parts_dir(out_path), f"part_{int(m):06d}.npz")
            if os.path.exists(part):
                return
        v = forward((args.scale * U[m]).astype(complex))
        gc.save_part(out_path, m, is_master, output=v, inp=U[m],
                     **getattr(forward, "extras", {}))

    def assemble():
        parts = gc.load_parts(out_path)
        inputs = np.stack([p["inp"] for p in parts])
        outputs = np.stack([p["output"] for p in parts])
        if args.readout == "intensity":
            outputs = np.abs(outputs) ** 2
        extra = gc.collect_extras(parts, reserved=("output", "inp"))
        np.savez(out_path, inputs=inputs, outputs=outputs, scale=args.scale,
                 readout=np.asarray(args.readout), components=np.asarray(comps),
                 out_sensor=np.asarray(args.out_sensor or "monitor_2"), **extra)
        gc.report_extras("ipcdata", extra, len(parts))
        print(f"[ipcdata] assembled → {out_path}  ({len(parts)} probes, readout={args.readout})", flush=True)

    return gc.run_mode(args, n_items, run_one, assemble, is_master,
                       out_path=out_path)


if __name__ == "__main__":
    raise SystemExit(main())
