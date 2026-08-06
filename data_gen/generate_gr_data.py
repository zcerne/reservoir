"""Generalisation-rank probe set — Legenstein & Maass 2007 §5. Index/assemble + incremental.

The companion of the IPC/kernel set. Where `generate_ipc_data.py` draws M *distinct*
inputs (kernel quality: how many independent states the reservoir can produce), this
draws a few BASE inputs and many NOISY REPLICAS of each (generalisation: how much the
state moves when the *same* input is repeated with noise).

Legenstein & Maass estimate the readout's VC dimension by the rank of the state matrix
over S_univ = noisy variations of one signal (their Thm 5.1: rank ≤ VC-dim ≤ rank+1),
and predict computational performance by **kernel rank − generalisation rank**. A low
GR means similar inputs map to similar states; a high GR means the reservoir amplifies
noise into new state directions the readout will happily fit.

Their exact protocol, which the defaults here reproduce: 4 templates × 125 jittered
versions = 500 probes for GR, against 500 distinct probes for the kernel measure.

WHY THIS CANNOT BE FAKED FROM EXISTING DATA (tried, 2026-08-04): adding i.i.d. noise to
the *recorded* fields of one sample gives a matrix whose rank is the number of replicas
— that measures the noise, not the reservoir. The noise has to enter BEFORE the medium
and propagate, i.e. new FDTD runs. FDTD itself is deterministic, so the noise source is
the drive: `--sigma` jitters each strip's amplitude, which physically is laser
amplitude noise.

Replica r=0 of every base is the UNJITTERED base itself, so the clean state is always
in the set (useful as the reference each replica is compared against).

Inputs are stored on the same u ∈ [-1,1] scale as the IPC set (physical amplitude is
`scale*u`), so the two sets are directly comparable and KR−GR is well defined.

  # 4 bases × 125 replicas = 500 items, 2% amplitude jitter
  N=$(python data_gen/generate_gr_data.py --path <design> --n_base 4 --n_rep 125 --count)
  sbatch --array=0-$((N-1))%8 --export=ALL,BATCH_SIZE=50 scripts/slurm_lips_array.sh \
      gr <design_on_project> --n_base 4 --n_rep 125 --sigma 0.02 --scale 50 \
      --out_sensor n2f_map --components Ex,Ey,Ez
  python data_gen/generate_gr_data.py --path <design> --n_base 4 --n_rep 125 --assemble
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
    ap.add_argument("--n_base", type=int, default=4,
                    help="number of distinct base inputs (L&M used 4 templates)")
    ap.add_argument("--n_rep", type=int, default=125,
                    help="noisy replicas per base, replica 0 being the clean base "
                         "(L&M used 125 per template → 500 total)")
    ap.add_argument("--sigma", type=float, default=0.02,
                    help="std of the per-strip amplitude jitter, in units of the "
                         "u ∈ [-1,1] input scale (0.02 = 2%% of full drive range). "
                         "This is the ONLY noise in the pipeline: FDTD is "
                         "deterministic, so GR is entirely set by how the medium "
                         "propagates this input jitter.")
    ap.add_argument("--readout", default="field", choices=["field", "intensity"],
                    help="applied at ASSEMBLE only; parts always store the raw field")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="physical amplitude = scale*u, exactly as in "
                         "generate_ipc_data.py — keep it EQUAL to the IPC set's "
                         "scale or KR and GR are measured at different drive levels "
                         "and their difference is meaningless")
    gc.add_extras_args(ap)
    gc.add_common_args(ap)
    args = ap.parse_args()

    comps = [c.strip() for c in args.components.split(",") if c.strip()]
    out_path = args.out or os.path.join(args.path, "datasets", "gr.npz")
    n_items = args.n_base * args.n_rep

    forward = n_strips = is_master = None
    U = base_id = rep_id = None
    if args.count or args.assemble:
        is_master = True
    else:
        forward, n_strips, is_master = gc.open_reservoir(
            args.path, comps, out_sensor=args.out_sensor,
            n_sources=args.n_sources, with_extras=args.extras,
            n2f_lam=args.n2f_lam)
        # Deterministic from --seed alone: every array task rebuilds the identical
        # probe table, so parts from different tasks belong to one coherent set.
        rng = np.random.default_rng(args.seed)
        base = rng.uniform(-1.0, 1.0, size=(args.n_base, n_strips))
        jit = rng.normal(0.0, args.sigma, size=(args.n_base, args.n_rep, n_strips))
        jit[:, 0, :] = 0.0                                  # replica 0 = clean base
        U = np.clip(base[:, None, :] + jit, -1.0, 1.0).reshape(n_items, n_strips)
        base_id = np.repeat(np.arange(args.n_base), args.n_rep)
        rep_id = np.tile(np.arange(args.n_rep), args.n_base)

    def run_one(m):
        v = forward((args.scale * U[m]).astype(complex))
        gc.save_part(out_path, m, is_master, output=v, inp=U[m],
                     base_id=np.asarray(base_id[m]), rep_id=np.asarray(rep_id[m]),
                     **getattr(forward, "extras", {}))

    def assemble():
        parts = gc.load_parts(out_path)
        inputs = np.stack([p["inp"] for p in parts])
        outputs = np.stack([p["output"] for p in parts])
        bids = np.stack([p["base_id"] for p in parts]).ravel()
        rids = np.stack([p["rep_id"] for p in parts]).ravel()
        if args.readout == "intensity":
            outputs = np.abs(outputs) ** 2
        extra = gc.collect_extras(parts, reserved=("output", "inp", "base_id", "rep_id"))
        np.savez(out_path, inputs=inputs, outputs=outputs, base_id=bids, rep_id=rids,
                 sigma=args.sigma, n_base=args.n_base, n_rep=args.n_rep,
                 scale=args.scale, readout=np.asarray(args.readout),
                 components=np.asarray(comps),
                 out_sensor=np.asarray(args.out_sensor or "monitor_2"), **extra)
        gc.report_extras("grdata", extra, len(parts))
        per = [int((bids == b).sum()) for b in np.unique(bids)]
        print(f"[grdata] assembled → {out_path}  ({len(parts)} probes, "
              f"{len(per)} bases, per-base {per}, sigma={args.sigma}, "
              f"readout={args.readout})", flush=True)

    return gc.run_mode(args, n_items, run_one, assemble, is_master,
                       out_path=out_path)


if __name__ == "__main__":
    raise SystemExit(main())
