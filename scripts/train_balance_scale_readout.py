"""Train a linear readout on the reservoir's far-field sensor — Balance Scale.

Reads the task set produced by data_gen/generate_balance_scale_data.py (whole
200x200 far-field map per polarization) and trains a ridge-regularised softmax
readout on N sensor points sampled at random — physically, N discrete
photodetectors scattered over the screen.

Why subsample rather than use everything: 120,000 features against 625 samples
fits any labelling perfectly and measures nothing. Sweeping N also answers a
question worth knowing on its own — how many detectors the device actually
needs.

Readout modes (`--mode`):
  intensity   |E|^2 at each point  — what a real photodetector measures (default)
  field       Re and Im separately — assumes coherent detection, upper bound
  both        intensity + field

The numbers to beat, from scripts/train_balance_scale_nn.py on the same split
protocol: multinomial logistic regression on the RAW 4 features ~0.89 test
accuracy, sigmoid MLP ~0.96. A readout near 0.89 means the reservoir added
nothing a linear map could not; approaching 0.96 means it is doing the degree-2
work the labels require.

  python scripts/train_balance_scale_readout.py --path data/lasing_testing/04_LC_4src
  python scripts/train_balance_scale_readout.py --path ... --n_points 10,25,50,100,200 --repeats 5
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


def softmax_ridge(Xtr, ytr, Xte, yte, n_out=3, epochs=3000, lr=0.5, ridge=1e-3):
    """Multinomial logistic regression with L2, full batch. Features are
    standardised on TRAIN statistics only (test must not inform the scaling)."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    W = np.zeros((n_out, A.shape[1])); b = np.zeros(n_out)
    Y = np.eye(n_out)[ytr]

    def P(X):
        lg = X @ W.T + b
        e = np.exp(lg - lg.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    for _ in range(epochs):
        d = (P(A) - Y) / len(A)
        W -= lr * (d.T @ A + ridge * W)
        b -= lr * d.sum(0)
    return (float((P(A).argmax(1) == ytr).mean()),
            float((P(B).argmax(1) == yte).mean()))


def features(out, shape, comps, pts, mode, keep=None):
    """out: (n_samples, n_comp*npix) complex → features at the chosen points.

    `keep`: component names to read out, default all. Worth restricting: on the
    pumped cavity the far field is 42% Ez, which the Ey (TE) signal cannot
    excite in 2D at all — it is the Ez pump at 0.45 re-radiated at 0.55 by the
    dye, whose emission couples every component (sigma_diag = 1,1,1). Reading
    Ez therefore measures the gain medium's state rather than the signal path,
    and it is worth ~0.04 of Balance accuracy on its own.
    """
    n = out.shape[0]
    cube = out.reshape(n, len(comps), -1)
    if keep:
        ci = [comps.index(c) for c in keep]
        cube = cube[:, ci, :]
    sel = cube[:, :, pts]                      # (n, n_comp, n_pts)
    if mode == "intensity":
        return (np.abs(sel) ** 2).reshape(n, -1)
    fld = np.concatenate([sel.real, sel.imag], axis=1).reshape(n, -1)
    if mode == "field":
        return fld
    return np.concatenate([(np.abs(sel) ** 2).reshape(n, -1), fld], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--dataset", default="balance_scale.npz")
    ap.add_argument("--n_points", default="10,25,50,100,200")
    ap.add_argument("--mode", default="intensity",
                    choices=["intensity", "field", "both"])
    ap.add_argument("--repeats", type=int, default=5,
                    help="independent random detector layouts per N")
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--components", default=None,
                    help="restrict the readout to these polarizations, e.g. "
                         "Ex,Ey to exclude the pump-fed Ez channel (default: all)")
    a = ap.parse_args()

    f = os.path.join(a.path, "datasets", a.dataset)
    d = dict(np.load(f, allow_pickle=True))
    out, y = np.asarray(d["outputs"]), np.asarray(d["labels"])
    comps = [str(c) for c in np.asarray(d["components"]).reshape(-1)]
    keep = ([c.strip() for c in a.components.split(",") if c.strip()]
            if a.components else comps)
    missing = [c for c in keep if c not in comps]
    if missing:
        raise SystemExit(f"dataset has {comps}, asked for {missing}")
    npix = out.shape[1] // len(comps)
    print(f"{os.path.basename(f)}: {out.shape[0]} samples, {len(comps)} comps "
          f"x {npix} px, readout={','.join(keep)}, mode={a.mode}")

    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(y))
    n_te = int(len(y) * a.test_frac)
    te, tr = idx[:n_te], idx[n_te:]

    rows = []
    for N in [int(v) for v in a.n_points.split(",") if v.strip()]:
        accs = []
        for r in range(a.repeats):
            pts = np.random.default_rng(1000 + r).choice(npix, size=min(N, npix),
                                                         replace=False)
            X = features(out, d.get("sensor_shape"), comps, pts, a.mode, keep)
            accs.append(softmax_ridge(X[tr], y[tr], X[te], y[te],
                                      ridge=a.ridge)[1])
        accs = np.array(accs)
        n_feat = len(keep) * len(pts) * (1 if a.mode == "intensity"
                                         else 2 if a.mode == "field" else 3)
        rows.append(dict(n_points=N, n_features=n_feat,
                         test_acc_mean=float(accs.mean()),
                         test_acc_std=float(accs.std())))
        print(f"  N={N:4d} detectors ({n_feat:5d} features): "
              f"test {accs.mean():.3f} ± {accs.std():.3f}")

    print("\n  reference on raw 4 features: logistic ~0.89, sigmoid MLP ~0.96")
    tag = "" if set(keep) == set(comps) else "_" + "".join(keep)
    stem = os.path.splitext(a.dataset)[0]
    js = os.path.join(a.path, "datasets",
                      f"{stem}_readout_{a.mode}{tag}.json")
    json.dump({"mode": a.mode, "components": keep, "ridge": a.ridge,
               "repeats": a.repeats, "test_frac": a.test_frac, "rows": rows},
              open(js, "w"), indent=2)
    print(f"  -> {js}")


if __name__ == "__main__":
    main()
