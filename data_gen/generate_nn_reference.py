"""Reference-NN characterization datasets — calibrate the suite against KNOWN systems.

Trains a small MLP on Iris (4 features → hidden → 3 classes) and treats the HIDDEN
LAYER as the "reservoir state" (what a linear readout taps). Generates the same four
characterization datasets (ipc / superposition / amp_sweep / harmonics) by pushing
inputs through forward(x) = hidden activations. No MEEP, no cluster — runs in seconds.

Two variants:
  --activation linear   → hidden = W1·x            (PURELY LINEAR — suite must say LINEAR)
  --activation sigmoid  → hidden = σ(W1·x + b1)    (KNOWN nonlinearity — odd-degree spectrum)

The linear net is the null control (validates the suite reports zero nonlinearity);
the sigmoid net is a known-nonlinear reference to compare the LC reservoir against.

  python data_gen/generate_nn_reference.py --activation linear  --out_dir data/reservoir_clasifications/05_linearNN
  python data_gen/generate_nn_reference.py --activation sigmoid --out_dir data/reservoir_clasifications/06_nonlinearNN
"""
from __future__ import annotations
import argparse, os
import numpy as np


def _act_fn(activation):
    if activation == "linear":
        return lambda z: z
    if activation == "tanh":
        return np.tanh
    return lambda z: 1.0 / (1.0 + np.exp(-z))                  # sigmoid


def build_forward(activation, hidden=32, seed=0, epochs=400, lr=0.1,
                  n_in=4, gain=1.0, depth=1):
    """Return forward(x)->state, the input dimension n_in, and a quality score.

    n_in=4 & depth=1 → the original Iris-TRAINED 4→hidden→3 MLP (hidden = state).
    A `gain` multiplies the pre-activation (gain>1 pushes the sigmoid harder into
    its nonlinear regime → stronger nonlinear MIXING at fixed rank).
    n_in>4 → an UNTRAINED random nonlinear network with `n_in` independent inputs
    (rank ≤ n_in, so more inputs = higher achievable rank) and `depth` hidden
    layers (deeper = stronger high-degree mixing). Random reference system, no dataset."""
    act = _act_fn(activation)
    rng = np.random.default_rng(seed)

    if n_in == 4 and depth == 1:
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        mu, sd = X.mean(0), X.std(0)
        Xs = (X - mu) / sd
        Y = np.eye(3)[y]                                       # one-hot
        n_out = 3
        W1 = rng.normal(0, 0.5, (hidden, 4)); b1 = np.zeros(hidden)
        W2 = rng.normal(0, 0.5, (n_out, hidden)); b2 = np.zeros(n_out)
        act_grad = ((lambda h, z: np.ones_like(z)) if activation == "linear"
                    else (lambda h, z: h * (1.0 - h)))
        for _ in range(epochs):                               # full-batch SGD (150 pts)
            Z1 = gain * (Xs @ W1.T) + b1; H = act(Z1)
            logits = H @ W2.T + b2
            P = np.exp(logits - logits.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            dL = (P - Y) / len(Xs)
            dW2 = dL.T @ H; db2 = dL.sum(0)
            dZ1 = (dL @ W2) * act_grad(H, Z1)
            dW1 = gain * (dZ1.T @ Xs); db1 = dZ1.sum(0)
            W2 -= lr * dW2; b2 -= lr * db2; W1 -= lr * dW1; b1 -= lr * db1
        acc = float((P.argmax(1) == y).mean())

        def forward(x):
            x = np.real(np.asarray(x)).ravel()
            return act(gain * (x @ W1.T) + b1)
        return forward, 4, acc

    # untrained random nonlinear reservoir with n_in inputs, `depth` hidden layers
    Ws, bs, d_prev = [], [], n_in
    for _ in range(depth):
        Ws.append(rng.normal(0, 1.0 / np.sqrt(d_prev), (hidden, d_prev)))
        bs.append(rng.normal(0, 0.1, hidden)); d_prev = hidden

    def forward(x):
        h = np.real(np.asarray(x)).ravel()
        for W, b in zip(Ws, bs):
            h = act(gain * (h @ W.T) + b)
        return h
    return forward, n_in, float("nan")


def gen_ipc(forward, n_strips, out, n=200, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.uniform(-1, 1, size=(n, n_strips))
    outs = np.stack([forward(U[m]) for m in range(n)])
    np.savez(out, inputs=U, outputs=outs.astype(np.float64),
             readout=np.asarray("state"), components=np.asarray("hidden"))
    print(f"[nn-ipc] {n} probes → {out}")


def gen_superposition(forward, n_strips, out, n_base=10, n_trials=40, seed=1):
    rng = np.random.default_rng(seed)
    E1 = rng.normal(size=(n_trials, n_strips)); E2 = rng.normal(size=(n_trials, n_strips))
    alpha = rng.normal(size=n_trials); beta = rng.normal(size=n_trials)
    o1 = np.stack([forward(E1[i]) for i in range(n_trials)])
    o2 = np.stack([forward(E2[i]) for i in range(n_trials)])
    oc = np.stack([forward(alpha[i] * E1[i] + beta[i] * E2[i]) for i in range(n_trials)])
    np.savez(out, E1=E1, E2=E2, alpha=alpha, beta=beta,
             out1=o1, out2=o2, out_combo=oc, components=np.asarray("hidden"))
    print(f"[nn-super] {n_trials} triples → {out}")


def gen_amp_sweep(forward, n_strips, out, levels=(0.1, 0.3, 1, 3, 10), n_probes=12, seed=2):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_probes, n_strips))
    ins, outs, lid = [], [], []
    lv = np.asarray(levels, float)
    for li, level in enumerate(lv):
        for p in range(n_probes):
            E = level * dirs[p]; ins.append(E); outs.append(forward(E)); lid.append(li)
    np.savez(out, inputs=np.stack(ins), outputs=np.stack(outs).astype(np.float64),
             level_id=np.asarray(lid), levels=lv, components=np.asarray("hidden"))
    print(f"[nn-amp] {len(lv)}×{n_probes} → {out}")


def gen_harmonics(forward, n_strips, out, tones=(3, 5), n_t=64, seed=3):
    chans = list(range(len(tones)))
    U = np.zeros((len(tones), n_strips))
    for k, s in enumerate(chans):
        U[k, s] = 1.0
    t = 2.0 * np.pi * np.arange(n_t) / n_t
    outs = []
    for j in range(n_t):
        E = np.zeros(n_strips)
        for k, tone in enumerate(tones):
            E += np.cos(tone * t[j]) * U[k]                   # real cosine multi-tone
        outs.append(forward(E))
    np.savez(out, outputs=np.stack(outs).astype(np.float64), inputs=np.zeros((n_t, n_strips)),
             t=t, tones=np.asarray(tones), amps=np.ones(len(tones)),
             components=np.asarray("hidden"))
    print(f"[nn-harm] tones={list(tones)} n_t={n_t} → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", required=True, choices=["linear", "sigmoid", "tanh"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n_ipc", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_in", type=int, default=4, help="input channels (rank ≤ n_in)")
    ap.add_argument("--gain", type=float, default=1.0, help="pre-activation scale (stronger nonlinear mixing)")
    ap.add_argument("--depth", type=int, default=1, help="hidden layers (n_in>4 random net)")
    args = ap.parse_args()

    forward, n_strips, acc = build_forward(args.activation, hidden=args.hidden, seed=args.seed,
                                           n_in=args.n_in, gain=args.gain, depth=args.depth)
    print(f"[nn] activation={args.activation} hidden={args.hidden} n_in={args.n_in} gain={args.gain} "
          f"depth={args.depth} train-acc={acc:.3f} state_dim={args.hidden}", flush=True)

    ds = os.path.join(args.out_dir, "datasets"); os.makedirs(ds, exist_ok=True)
    gen_ipc(forward, n_strips, os.path.join(ds, "ipc.npz"), n=args.n_ipc)
    gen_superposition(forward, n_strips, os.path.join(ds, "superposition.npz"))
    gen_amp_sweep(forward, n_strips, os.path.join(ds, "amp_sweep.npz"))
    gen_harmonics(forward, n_strips, os.path.join(ds, "harmonics.npz"))
    print(f"[nn] all 4 datasets → {ds}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
