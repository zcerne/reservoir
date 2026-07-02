"""Harmonic/intermodulation-distortion data — Nonlinearity Method D. Index/assemble.

Drive with 1–2 tones via a phase-sweep parameter t; N_t samples over one period, one
forward run each. The reservoir source casts amplitude to real, so `e^{iωt}` becomes a
REAL cosine `cos(ωt)` — i.e. real-cosine multi-tone drive (gives 2ω + sum + diff under
|E|²). Work item j runs E(t_j); each → part (incremental). --assemble → <out>.npz
{outputs, inputs, t, tones, ...} for n4_harmonics_distortion.harmonic_specter.

Use well-separated integer tones (default 3,5) on distinct channels; N_t > 2·max_order·max_tone.

  N=$(python data_gen/generate_harmonics_data.py --path data/reservoir_clasifications/01_2D_director --tones 3,5 --channels 0,1 --n_t 64 --count)
  sbatch --array=0-$((N-1)) slurm_char_array.sh harmonics data/reservoir_clasifications/01_2D_director --tones 3,5 --channels 0,1 --n_t 64
  python data_gen/generate_harmonics_data.py --path data/reservoir_clasifications/01_2D_director --tones 3,5 --channels 0,1 --n_t 64 --assemble
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
    ap.add_argument("--tones", default="3,5")
    ap.add_argument("--amps", default=None)
    ap.add_argument("--channels", default=None)
    ap.add_argument("--n_t", type=int, default=64)
    gc.add_common_args(ap)
    args = ap.parse_args()

    comps = [c.strip() for c in args.components.split(",") if c.strip()]
    tones = [int(t) for t in args.tones.split(",") if t.strip()]
    amps = ([float(a) for a in args.amps.split(",")] if args.amps else [1.0] * len(tones))
    if len(amps) != len(tones):
        raise SystemExit("--amps must have one value per tone")
    out_path = args.out or os.path.join(args.path, "datasets", "harmonics.npz")
    t_grid = 2.0 * np.pi * np.arange(args.n_t) / args.n_t
    n_items = args.n_t

    forward = n_strips = is_master = U = chans = None
    if args.count or args.assemble:
        is_master = True
    else:
        forward, n_strips, is_master = gc.open_reservoir(args.path, comps)
        chans = ([int(c) for c in args.channels.split(",")] if args.channels
                 else list(range(len(tones))))
        if len(chans) != len(tones) or max(chans) >= n_strips:
            raise SystemExit(f"--channels needs one valid strip index (<{n_strips}) per tone")
        U = np.zeros((len(tones), n_strips))                  # REAL unit patterns
        for k, s in enumerate(chans):
            U[k, s] = 1.0

    def run_one(j):
        t = t_grid[j]
        E = np.zeros(n_strips, dtype=complex)
        for k in range(len(tones)):
            E += amps[k] * np.exp(1j * tones[k] * t) * U[k]   # Re() taken by the source
        gc.save_part(out_path, j, is_master, output=forward(E), inp=E, t=float(t))

    def assemble():
        parts = gc.load_parts(out_path)
        outputs = np.stack([p["output"] for p in parts])
        inputs = np.stack([p["inp"] for p in parts])
        t = np.asarray([float(p["t"]) for p in parts])
        np.savez(out_path, outputs=outputs, inputs=inputs, t=t,
                 tones=np.asarray(tones), amps=np.asarray(amps), components=np.asarray(comps))
        print(f"[harmdata] assembled → {out_path}  ({len(parts)} samples, tones={tones})", flush=True)

    return gc.run_mode(args, n_items, run_one, assemble, is_master)


if __name__ == "__main__":
    raise SystemExit(main())
