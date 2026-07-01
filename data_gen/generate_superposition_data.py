"""Superposition (linearity) data — Nonlinearity Method A. Index/assemble + incremental.

Per trial the analysis needs f(E₁), f(E₂), f(αE₁+βE₂). Base POOL: n_base base inputs
(run once each) + n_trials combinations (each references two base members via α,β).
Work items (deterministic from --seed):
    items 0 .. n_base-1                  : base run  E_base[i]
    items n_base .. n_base+n_trials-1    : combo run αE_base[i]+βE_base[j]
Each item → one forward run → part file (incremental save). --assemble expands
out1/out2 from the base parts → final <out>.npz with the keys n1 expects.

REAL amplitudes (the reservoir source casts amplitude to float; complex phase is lost —
real inputs fully test linearity). Consumed by n1_superposition.super_position_test.

  N=$(python data_gen/generate_superposition_data.py --path data/test2D --n_base 8 --n_trials 40 --count)
  sbatch --array=0-$((N-1)) slurm_char_array.sh superposition data/test2D --n_base 8 --n_trials 40
  python data_gen/generate_superposition_data.py --path data/test2D --n_base 8 --n_trials 40 --assemble
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import _gen_common as gc


def _plan(seed, n_base, n_trials, n_strips, scale):
    rng = np.random.default_rng(seed)
    E_base = rng.normal(size=(n_base, n_strips)) * scale
    combos = []
    for _ in range(n_trials):
        i, j = rng.choice(n_base, size=2, replace=False)
        a = rng.normal() * (scale / max(scale, 1.0)); b = rng.normal()
        combos.append((int(i), int(j), float(a), float(b)))
    return E_base, combos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_base", type=int, default=8)
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--scale", type=float, default=1.0)
    gc.add_common_args(ap)
    args = ap.parse_args()

    comps = [c.strip() for c in args.components.split(",") if c.strip()]
    out_path = args.out or os.path.join(args.path, "superposition.npz")
    n_items = args.n_base + args.n_trials

    forward = n_strips = is_master = None
    E_base = combos = None
    if args.count:
        is_master = True
    else:
        if args.assemble:                                     # assemble: no reservoir needed
            is_master = True
        else:
            forward, n_strips, is_master = gc.open_reservoir(args.path, comps)
            E_base, combos = _plan(args.seed, args.n_base, args.n_trials, n_strips, args.scale)

    def run_one(k):
        if k < args.n_base:
            gc.save_part(out_path, k, is_master, kind="base",
                         output=forward(E_base[k]), inp=E_base[k])
        else:
            i, j, a, b = combos[k - args.n_base]
            gc.save_part(out_path, k, is_master, kind="combo",
                         output=forward(a * E_base[i] + b * E_base[j]),
                         i=i, j=j, alpha=a, beta=b)

    def assemble():
        parts = gc.load_parts(out_path)
        base = {int(p["idx"]): p for p in parts if str(p["kind"]) == "base"}
        E1, E2, alpha, beta, out1, out2, out_combo = [], [], [], [], [], [], []
        for p in parts:
            if str(p["kind"]) != "combo":
                continue
            i, j = int(p["i"]), int(p["j"])
            E1.append(base[i]["inp"]); E2.append(base[j]["inp"])
            alpha.append(float(p["alpha"])); beta.append(float(p["beta"]))
            out1.append(base[i]["output"]); out2.append(base[j]["output"]); out_combo.append(p["output"])
        np.savez(out_path, E1=np.stack(E1), E2=np.stack(E2),
                 alpha=np.asarray(alpha), beta=np.asarray(beta),
                 out1=np.stack(out1), out2=np.stack(out2), out_combo=np.stack(out_combo),
                 components=np.asarray(comps))
        print(f"[supdata] assembled → {out_path}  ({len(out_combo)} trials from {len(parts)} parts)", flush=True)

    return gc.run_mode(args, n_items, run_one, assemble, is_master)


if __name__ == "__main__":
    raise SystemExit(main())
