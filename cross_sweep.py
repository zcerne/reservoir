"""Cross-saturation probe: sweep ONE strip's amplitude with the others held fixed.

The equal-drive amp sweep only probes the common mode (every channel gets the
same saturation factor). This drives an ASYMMETRIC input — one strip at
--level, the rest at --fixed (the knee) — so the swept strip rewrites the
shared gain landscape and the FIXED strips' output lobes respond iff the
channels are truly coupled (shared inversion / population grating). That
response vs. swept level is the input-pattern-dependent mixing a space
reservoir computes with.

    python cross_sweep.py --path data/signal_modulation/05b_ampsweep \
        --strip 2 --level 90 --fixed 70

One FDTD per call (array-friendly). The summary npz stores the drive vector and
output norm; the y-profiles live in the design's per-run monitor npz
(monitor_2_<SIMPLESIM_SCRATCH_TAG>.npz etc.), so the analysis reads those.
"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--path", required=True)
ap.add_argument("--strip", type=int, default=2,
                help="1-based index of the swept source strip (default 2, an "
                     "inner strip — two neighbours, strongest coupling)")
ap.add_argument("--level", type=float, required=True,
                help="swept strip amplitude")
ap.add_argument("--fixed", type=float, default=70.0,
                help="amplitude of every other strip (default 70 ~ the knee)")
ap.add_argument("--out", default=None,
                help="summary npz (default <path>/datasets/cross_s<strip>_L<level>.npz)")
args = ap.parse_args()

out_path = args.out or os.path.join(
    args.path, "datasets", f"cross_s{args.strip}_L{args.level:g}.npz")

import data_gen._gen_common as gc
forward, n_strips, is_master = gc.open_reservoir(args.path, ["Ey"])
if not (1 <= args.strip <= n_strips):
    sys.exit(f"--strip {args.strip} out of range (design has {n_strips} strips)")

E = np.full(n_strips, float(args.fixed))
E[args.strip - 1] = float(args.level)
out = forward(E)                       # MPI-collective; None off master
if is_master:
    out_norm = float(np.linalg.norm(out))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, level=args.level, fixed=args.fixed, strip=args.strip,
             E=E, out_norm=out_norm)
    print(f"[cross] strip {args.strip} at {args.level:g}, others {args.fixed:g}"
          f": |out|={out_norm:.6g} -> {out_path}", flush=True)
