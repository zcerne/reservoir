"""Dambre-IPC data — Nonlinearity Method F (gold standard). Index/assemble + incremental.

Inputs MUST be i.i.d. ~ Uniform[-1,1] per source strip (Legendre orthonormality). M
such real input vectors, one forward run each. Work item m runs U[m]; each → part
(incremental). --assemble → <out>.npz {inputs, outputs} for n6_dambre.dambre_ipc.
Also serves n2 (residual) and n5 (Volterra). Need M ≫ #output features.

`--readout intensity` (default) saves |E|² (IPC uses the nonlinear readout state);
`--readout field` saves the complex field.

  N=$(python data_gen/generate_ipc_data.py --path data/test2D --n 400 --count)
  sbatch --array=0-$((N-1)) slurm_char_array.sh ipc data/test2D --n 400 --readout intensity
  python data_gen/generate_ipc_data.py --path data/test2D --n 400 --readout intensity --assemble
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
    ap.add_argument("--readout", default="intensity", choices=["field", "intensity"])
    gc.add_common_args(ap)
    args = ap.parse_args()

    comps = [c.strip() for c in args.components.split(",") if c.strip()]
    out_path = args.out or os.path.join(args.path, "ipc.npz")
    n_items = args.n

    forward = n_strips = is_master = U = None
    if args.count or args.assemble:
        is_master = True
    else:
        forward, n_strips, is_master = gc.open_reservoir(args.path, comps)
        rng = np.random.default_rng(args.seed)
        U = rng.uniform(-1.0, 1.0, size=(args.n, n_strips))   # i.i.d. Uniform[-1,1] (real)

    def run_one(m):
        v = forward(U[m].astype(complex))
        out = (np.abs(v) ** 2) if args.readout == "intensity" else v
        gc.save_part(out_path, m, is_master, output=out, inp=U[m])

    def assemble():
        parts = gc.load_parts(out_path)
        inputs = np.stack([p["inp"] for p in parts])
        outputs = np.stack([p["output"] for p in parts])
        np.savez(out_path, inputs=inputs, outputs=outputs,
                 readout=np.asarray(args.readout), components=np.asarray(comps))
        print(f"[ipcdata] assembled → {out_path}  ({len(parts)} probes, readout={args.readout})", flush=True)

    return gc.run_mode(args, n_items, run_one, assemble, is_master)


if __name__ == "__main__":
    raise SystemExit(main())
