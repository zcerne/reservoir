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


def build_forward(activation, hidden=32, seed=0, epochs=400, lr=0.1):
    """Train a 4→hidden→3 MLP on standardized Iris; return forward(x)->hidden state
    (the readout layer), the standardizer, and the trained test accuracy."""
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    mu, sd = X.mean(0), X.std(0)
    Xs = (X - mu) / sd
    Y = np.eye(3)[y]                                           # one-hot
    rng = np.random.default_rng(seed)
    n_in, n_out = 4, 3
    W1 = rng.normal(0, 0.5, (hidden, n_in)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.5, (n_out, hidden)); b2 = np.zeros(n_out)

    def act(z):
        return z if activation == "linear" else 1.0 / (1.0 + np.exp(-z))

    def act_grad(h, z):
        return np.ones_like(z) if activation == "linear" else h * (1.0 - h)

    # plain SGD full-batch (150 pts — trivial)
    for _ in range(epochs):
        Z1 = Xs @ W1.T + b1; H = act(Z1)
        logits = H @ W2.T + b2
        P = np.exp(logits - logits.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        dL = (P - Y) / len(Xs)                                 # softmax+CE grad
        dW2 = dL.T @ H; db2 = dL.sum(0)
        dH = dL @ W2
        dZ1 = dH * act_grad(H, Z1)
        dW1 = dZ1.T @ Xs; db1 = dZ1.sum(0)
        W2 -= lr * dW2; b2 -= lr * db2; W1 -= lr * dW1; b1 -= lr * db1

    acc = float((P.argmax(1) == y).mean())

    def forward(x):
        """x: (4,) standardized-feature-space input → (hidden,) real state."""
        x = np.real(np.asarray(x)).ravel()                    # NN takes real input
        return act(x @ W1.T + b1)

    return forward, 4, acc


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
    ap.add_argument("--activation", required=True, choices=["linear", "sigmoid"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n_ipc", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    forward, n_strips, acc = build_forward(args.activation, hidden=args.hidden, seed=args.seed)
    print(f"[nn] activation={args.activation} hidden={args.hidden} iris train-acc={acc:.3f} "
          f"state_dim={args.hidden}", flush=True)

    ds = os.path.join(args.out_dir, "datasets"); os.makedirs(ds, exist_ok=True)
    gen_ipc(forward, n_strips, os.path.join(ds, "ipc.npz"), n=args.n_ipc)
    gen_superposition(forward, n_strips, os.path.join(ds, "superposition.npz"))
    gen_amp_sweep(forward, n_strips, os.path.join(ds, "amp_sweep.npz"))
    gen_harmonics(forward, n_strips, os.path.join(ds, "harmonics.npz"))
    print(f"[nn] all 4 datasets → {ds}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
