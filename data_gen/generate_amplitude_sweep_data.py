"""Amplitude-sweep data — Nonlinearity Method C (amplitude-dependent BLA). Index/assemble.

M random UNIT input directions (shared) × L drive levels → L·M forward runs. Work item
k=(li·M + p) runs E = levels[li]·dirs[p]. Each item → part (incremental). --assemble →
final <out>.npz {inputs, outputs, level_id, levels} for n3_amplitude_dependant.

REAL amplitudes (source casts to float). Deterministic from --seed.

  N=$(python data_gen/generate_amplitude_sweep_data.py --path data/test2D --levels 0.1,0.3,1,3,10 --n_probes 12 --count)
  sbatch --array=0-$((N-1)) slurm_char_array.sh ampsweep data/test2D --levels 0.1,0.3,1,3,10 --n_probes 12
  python data_gen/generate_amplitude_sweep_data.py --path data/test2D --levels 0.1,0.3,1,3,10 --n_probes 12 --assemble
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
    ap.add_argument("--levels", default="0.1,0.3,1,3,10")
    ap.add_argument("--n_probes", type=int, default=12)
    gc.add_common_args(ap)
    args = ap.parse_args()

    comps = [c.strip() for c in args.components.split(",") if c.strip()]
    levels = np.array([float(x) for x in args.levels.split(",")], dtype=float)
    out_path = args.out or os.path.join(args.path, "datasets", "amp_sweep.npz")
    M = args.n_probes
    n_items = len(levels) * M

    forward = n_strips = is_master = dirs = None
    if args.count or args.assemble:
        is_master = True
    else:
        forward, n_strips, is_master = gc.open_reservoir(args.path, comps)
        rng = np.random.default_rng(args.seed)
        dirs = rng.normal(size=(M, n_strips))                 # REAL directions
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30)

    def run_one(k):
        li, p = divmod(k, M)
        E = levels[li] * dirs[p]
        gc.save_part(out_path, k, is_master, output=forward(E), inp=E, level_id=int(li))

    def assemble():
        parts = gc.load_parts(out_path)
        inputs = np.stack([p["inp"] for p in parts])
        outputs = np.stack([p["output"] for p in parts])
        level_id = np.asarray([int(p["level_id"]) for p in parts])
        np.savez(out_path, inputs=inputs, outputs=outputs, level_id=level_id,
                 levels=levels, components=np.asarray(comps))
        print(f"[ampdata] assembled → {out_path}  ({len(parts)} probes, {len(levels)} levels)", flush=True)

    return gc.run_mode(args, n_items, run_one, assemble, is_master)


if __name__ == "__main__":
    raise SystemExit(main())
