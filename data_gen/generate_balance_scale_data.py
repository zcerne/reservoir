"""UCI Balance Scale through the optical reservoir — the task training set.

The 4 attributes (left weight, left distance, right weight, right distance) map
onto the 4 source strips, one feature per strip. The label is decided by
comparing the PRODUCTS LW*LD vs RW*RD, so it is a degree-2 function of the
inputs that no linear readout on the raw features can express — see
scripts/train_balance_scale_nn.py for the reference numbers (linear ~0.89,
sigmoid ~0.96 test accuracy). A reservoir readout landing near the linear
figure is contributing nothing; approaching the sigmoid figure means it is
doing real degree-2 work.

Drive level matters. Design 04's own amplitude sweep shows response already
11% compressed at amplitude 10 and roughly halved by 35, so features are mapped
into [--amp_min, --amp_max] = [10, 50] by default: the reservoir has to sit in
the saturating regime for gain nonlinearity to mix the inputs at all. Driving
it at amplitude ~1 would make the device effectively linear and the task
unlearnable beyond the linear baseline.

Saves the WHOLE sensor array for every requested polarization (`--full_sensor`,
on by default here): the far-field map is 200x200 per component, so a sample is
~1.9 MB over Ex/Ey/Ez and the full 625-sample set is ~1.2 GB. `sensor_shape` in
the assembled npz says how to reshape the flat vector. Keeping everything means
the readout — which spatial region, which polarization, field or intensity — is
chosen at analysis time rather than being baked in here.

  N=$(python data_gen/generate_balance_scale_data.py --path data/lasing_testing/04_LC_4src --count)
  python data_gen/generate_balance_scale_data.py --path data/lasing_testing/04_LC_4src \
      --out_sensor n2f_map --components Ex,Ey,Ez --batch 0 --batch_size 50 --skip_existing
  python data_gen/generate_balance_scale_data.py --path ... --assemble
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import _gen_common as gc

CLASSES = ["L", "B", "R"]
UCI = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "balance-scale/balance-scale.data")


def load_balance_scale(repo_root):
    cache = os.path.join(repo_root, "data", "balance_scale", "balance-scale.data")
    if not os.path.exists(cache):
        import urllib.request
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        req = urllib.request.Request(UCI, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            open(cache, "wb").write(r.read())
    rows = [l.split(",") for l in open(cache).read().split() if l.strip()]
    y = np.array([CLASSES.index(r[0]) for r in rows], dtype=np.int64)
    X = np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float64)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--amp_min", type=float, default=10.0)
    ap.add_argument("--amp_max", type=float, default=50.0)
    ap.add_argument("--full_sensor", action="store_true", default=True)
    ap.add_argument("--extras", action="store_true", default=True,
                    help="also store monitor_2 fields and the near2far "
                         "equivalence currents in each part (default on)")
    ap.add_argument("--no_extras", dest="extras", action="store_false")
    ap.add_argument("--last_column_only", dest="full_sensor", action="store_false",
                    help="store only the far screen's last column instead of the whole map")
    gc.add_common_args(ap)
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    comps = [c.strip() for c in args.components.split(",") if c.strip()]
    out_path = args.out or os.path.join(args.path, "datasets", "balance_scale.npz")
    X, y = load_balance_scale(repo_root)
    n_items = len(X)

    # per-feature min-max -> [amp_min, amp_max]; each feature drives one strip
    lo, hi = X.min(0), X.max(0)
    A = args.amp_min + (X - lo) / np.where(hi > lo, hi - lo, 1.0) * (args.amp_max - args.amp_min)

    forward = n_strips = is_master = None
    if args.count or args.assemble:
        is_master = True
    else:
        forward, n_strips, is_master = gc.open_reservoir(
            args.path, comps, out_sensor=args.out_sensor,
            full_sensor=args.full_sensor, with_extras=args.extras,
            n_sources=args.n_sources)
        if n_strips != X.shape[1]:
            raise SystemExit(
                f"design has {n_strips} source strips but Balance Scale has "
                f"{X.shape[1]} features — use a 4-strip design (e.g. 04_LC_4src)")

    def run_one(k):
        out = forward(A[k].astype(complex))
        gc.save_part(out_path, k, is_master, output=out, inp=X[k], amp=A[k],
                     label=int(y[k]), **getattr(forward, "extras", {}))

    def assemble():
        parts = gc.load_parts(out_path)
        outputs = np.stack([p["output"] for p in parts])
        n_per = outputs.shape[1] // len(comps)
        side = int(round(np.sqrt(n_per)))
        shape = ([len(comps), side, side] if args.full_sensor and side * side == n_per
                 else [len(comps), n_per])
        # monitor_2 fields and near2far equivalence currents. Geometry keys
        # (ys, wy, freqs, x_line, dx) are identical in every run, so store one
        # copy; only the per-sample arrays are stacked.
        extra = {}
        for key in sorted(set(parts[0]) - {"idx", "output", "inp", "amp", "label"}):
            vals = [np.asarray(p[key]) for p in parts if key in p]
            if len(vals) != len(parts):
                continue
            same = all(v.shape == vals[0].shape and np.array_equal(v, vals[0])
                       for v in vals[1:])
            extra[key] = vals[0] if same else np.stack(vals)

        np.savez(out_path,
                 inputs=np.stack([p["inp"] for p in parts]),
                 amplitudes=np.stack([p["amp"] for p in parts]),
                 labels=np.asarray([int(p["label"]) for p in parts]),
                 outputs=outputs, components=np.asarray(comps),
                 class_names=np.asarray(CLASSES),
                 sensor_shape=np.asarray(shape),
                 amp_range=np.asarray([args.amp_min, args.amp_max]),
                 **extra)
        if extra:
            per = [k for k, v in extra.items() if np.asarray(v).shape[:1] == (len(parts),)]
            print(f"[balance] extras: {len(per)} per-sample arrays "
                  f"({', '.join(per[:6])}{'…' if len(per) > 6 else ''}), "
                  f"{len(extra) - len(per)} shared", flush=True)
        print(f"[balance] assembled → {out_path}  ({len(parts)} samples, "
              f"outputs {outputs.shape}, per-sample layout {shape})", flush=True)

    return gc.run_mode(args, n_items, run_one, assemble, is_master,
                       out_path=out_path)


if __name__ == "__main__":
    raise SystemExit(main())
