"""Linear vs sigmoid MLP on UCI Balance Scale — the nonlinearity yardstick.

Balance Scale is the right benchmark for a 4-input reservoir: 4 attributes
(left weight, left distance, right weight, right distance), 625 samples, 3
classes, and the label is defined by comparing the PRODUCTS
(left weight x left distance) vs (right weight x right distance). The target
is therefore a degree-2 function of the inputs, which no linear readout on the
raw features can express — unlike Iris, which is already linearly separable and
so cannot tell a linear device from a nonlinear one.

Trains the same two reference networks the characterization suite uses:
  identity  hidden = W1.x            (no nonlinearity anywhere -> the null control)
  sigmoid   hidden = sig(W1.x + b1)  (known nonlinearity)
plus a multinomial logistic regression directly on the raw features, which is
the honest "linear readout" floor.

The gap between the identity net and the sigmoid net is the accuracy that
nonlinearity buys on this task — the number a reservoir has to beat to be
earning its keep.

  python scripts/train_balance_scale_nn.py
  python scripts/train_balance_scale_nn.py --hidden 32 --epochs 4000 --seed 0
"""
from __future__ import annotations
import argparse, json, os, urllib.request
import numpy as np

UCI = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "balance-scale/balance-scale.data")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "balance_scale", "balance-scale.data")
CLASSES = ["L", "B", "R"]


def load_balance_scale():
    """(X, y, product_margin). Cached under data/balance_scale/ after one fetch."""
    if not os.path.exists(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        req = urllib.request.Request(UCI, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            open(CACHE, "wb").write(r.read())
        print(f"[data] downloaded UCI Balance Scale -> {CACHE}")
    rows = [l.split(",") for l in open(CACHE).read().split() if l.strip()]
    y = np.array([CLASSES.index(r[0]) for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows])
    # the generating rule, for reference: LW*LD - RW*RD
    margin = X[:, 0] * X[:, 1] - X[:, 2] * X[:, 3]
    return X, y, margin


def train_mlp(Xtr, ytr, Xte, yte, activation, hidden=32, epochs=4000,
              lr=0.5, seed=0, n_out=3):
    """Full-batch softmax-regression MLP, same shape as generate_nn_reference's."""
    rng = np.random.default_rng(seed)
    act = (lambda z: z) if activation == "identity" else \
          (lambda z: 1.0 / (1.0 + np.exp(-z)))
    # d(activation)/dz expressed in terms of the ACTIVATION h, so it never needs
    # the pre-activation z (identity: 1; sigmoid: h(1-h))
    grad = ((lambda h: np.ones_like(h)) if activation == "identity"
            else (lambda h: h * (1.0 - h)))
    n_in = Xtr.shape[1]
    W1 = rng.normal(0, 0.5, (hidden, n_in)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.5, (n_out, hidden)); b2 = np.zeros(n_out)
    Y = np.eye(n_out)[ytr]

    def fwd(X):
        H = act(X @ W1.T + b1)
        lg = H @ W2.T + b2
        P = np.exp(lg - lg.max(1, keepdims=True))
        return H, P / P.sum(1, keepdims=True)

    for _ in range(epochs):
        H, P = fwd(Xtr)
        dL = (P - Y) / len(Xtr)
        dW2 = dL.T @ H; db2 = dL.sum(0)
        dZ1 = (dL @ W2) * grad(H)
        dW1 = dZ1.T @ Xtr; db1 = dZ1.sum(0)
        W2 -= lr * dW2; b2 -= lr * db2; W1 -= lr * dW1; b1 -= lr * db1

    tr = float((fwd(Xtr)[1].argmax(1) == ytr).mean())
    te = float((fwd(Xte)[1].argmax(1) == yte).mean())
    return dict(activation=activation, hidden=hidden, epochs=epochs,
                train_acc=tr, test_acc=te), (W1, b1, W2, b2)


def logistic_baseline(Xtr, ytr, Xte, yte, epochs=4000, lr=0.5, n_out=3):
    """Multinomial logistic regression on the raw features: the linear floor."""
    W = np.zeros((n_out, Xtr.shape[1])); b = np.zeros(n_out)
    Y = np.eye(n_out)[ytr]

    def P(X):
        lg = X @ W.T + b
        e = np.exp(lg - lg.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    for _ in range(epochs):
        d = (P(Xtr) - Y) / len(Xtr)
        W -= lr * (d.T @ Xtr); b -= lr * d.sum(0)
    return dict(model="logistic_regression_raw",
                train_acc=float((P(Xtr).argmax(1) == ytr).mean()),
                test_acc=float((P(Xte).argmax(1) == yte).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--root", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "reservoir_clasifications"))
    a = ap.parse_args()

    X, y, margin = load_balance_scale()
    mu, sd = X.mean(0), X.std(0)
    Xs = (X - mu) / sd
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(Xs))
    n_te = int(len(Xs) * a.test_frac)
    te, tr = idx[:n_te], idx[n_te:]
    print(f"Balance Scale: {len(Xs)} samples, {Xs.shape[1]} features, "
          f"classes {dict(zip(CLASSES, np.bincount(y)))}; "
          f"train {len(tr)} / test {len(te)}")

    lin = logistic_baseline(Xs[tr], y[tr], Xs[te], y[te], a.epochs, a.lr)
    print(f"  logistic on raw features : train {lin['train_acc']:.3f}  "
          f"test {lin['test_acc']:.3f}")

    out = {"dataset": "uci_balance_scale", "n": int(len(Xs)),
           "note": ("label = sign(LW*LD - RW*RD): a degree-2 function of the "
                    "inputs, so a linear model cannot express it"),
           "logistic_raw": lin, "models": {}}
    for act, folder in (("identity", "05_linearNN"), ("sigmoid", "06_nonlinearNN")):
        res, (W1, b1, W2, b2) = train_mlp(Xs[tr], y[tr], Xs[te], y[te], act,
                                          a.hidden, a.epochs, a.lr, a.seed)
        print(f"  MLP {act:8s} hidden={a.hidden} : train {res['train_acc']:.3f}  "
              f"test {res['test_acc']:.3f}")
        d = os.path.join(a.root, folder)
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "balance_scale_model.npz"),
                 W1=W1, b1=b1, W2=W2, b2=b2, mu=mu, sd=sd,
                 activation=np.asarray(act), classes=np.asarray(CLASSES),
                 train_idx=tr, test_idx=te, **{f"acc_{k}": v for k, v in
                                               res.items() if "acc" in k})
        json.dump(res, open(os.path.join(d, "balance_scale_metrics.json"), "w"),
                  indent=2)
        out["models"][act] = {**res, "folder": folder}

    gap = out["models"]["sigmoid"]["test_acc"] - out["models"]["identity"]["test_acc"]
    out["nonlinearity_gain_test_acc"] = gap
    print(f"\n  => nonlinearity buys {gap*100:+.1f} points of test accuracy "
          f"({out['models']['identity']['test_acc']:.3f} -> "
          f"{out['models']['sigmoid']['test_acc']:.3f})")
    summ = os.path.join(a.root, "balance_scale_nn_summary.json")
    json.dump(out, open(summ, "w"), indent=2)
    print(f"  summary -> {summ}")


if __name__ == "__main__":
    main()
