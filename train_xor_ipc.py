"""XOR readout training on the IPC dataset — is the reservoir's cross-strip
nonlinearity USABLE for computation?

Task: from the IPC probes (inputs u ~ U(-1,1)^4, outputs = complex exit fields)
build the XOR-of-signs dataset y = [u_a * u_b < 0] for a strip pair (a,b) and
train readouts on the RESERVOIR OUTPUT:

  * linear probe (logistic regression)  — succeeds only if the reservoir itself
    computed the cross product u_a*u_b (XOR is not linearly separable in the
    inputs; this is the reservoir-computing verdict);
  * simple nonlinear net (MLP tanh 64-32) — the upper bound with a nonlinear
    readout on top;
  * control: the same two models on the RAW INPUTS (linear must sit at ~50%,
    MLP near 100% — sanity that the task is XOR and the models train);
  * shuffle control: labels permuted -> everything must drop to ~50%.

Features from the reservoir: |E|^2 of the exit line at the signal bin
(lam=1.064) by default — 252 real features; --features field uses Re/Im instead;
--features all uses every lambda bin. PCA is deliberately avoided: plain ridge-
style standardisation only, split BEFORE any fitting.

  python train_xor_ipc.py --npz <path>/datasets/ipc.npz --strips 0 1
"""
from __future__ import annotations
import argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="/home/ziga/Lips/resevoir/data/signal_modulation/05d_ampsweep/datasets/ipc.npz")
ap.add_argument("--strips", type=int, nargs=2, default=[0, 1],
                help="strip pair (a b) whose sign-XOR is the label")
ap.add_argument("--features", choices=["intensity", "field", "all"], default="intensity")
ap.add_argument("--test", type=int, default=200)
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None, help="npz to store metrics/curves")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
d = np.load(args.npz)
U = np.asarray(d["inputs"])                # (M,4)
Y = (U[:, args.strips[0]] * U[:, args.strips[1]] < 0).astype(np.float32)
out = np.asarray(d["outputs"])             # (M, 61*ny) complex
M = len(U)
freqs = np.asarray(d["m2_freqs"]) if "m2_freqs" in d.files else None
nlam = len(freqs) if freqs is not None else 61
ny = out.shape[1] // nlam
E = out.reshape(M, nlam, ny)
if args.features == "all":
    X = np.abs(E).reshape(M, -1) ** 2
else:
    k = int(np.argmin(np.abs(freqs - 1/1.064))) if freqs is not None else nlam // 2
    Ek = E[:, k, :]
    X = (np.abs(Ek) ** 2) if args.features == "intensity" else \
        np.concatenate([Ek.real, Ek.imag], axis=1)
X = X.astype(np.float64)

idx = rng.permutation(M)
tr, te = idx[:-args.test], idx[-args.test:]
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
Xs = (X - mu) / sd

# --- dependency-free training: manual Adam, full batch (tiny problem) -------
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

class Linear:
    def __init__(self, nf, rng):
        self.W = rng.normal(0, 1/np.sqrt(nf), (nf, 1)); self.b = np.zeros(1)
        self.params = [self.W, self.b]
    def forward(self, X):
        self._X = X
        return X @ self.W + self.b
    def backward(self, dz):
        return [self._X.T @ dz / len(dz), dz.mean(0)]

class MLP:
    def __init__(self, nf, rng, h1=64, h2=32):
        s = lambda a, b: rng.normal(0, np.sqrt(2.0/a), (a, b))
        self.W1, self.b1 = s(nf, h1), np.zeros(h1)
        self.W2, self.b2 = s(h1, h2), np.zeros(h2)
        self.W3, self.b3 = s(h2, 1),  np.zeros(1)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
    def forward(self, X):
        self.X = X
        self.a1 = np.tanh(X @ self.W1 + self.b1)
        self.a2 = np.tanh(self.a1 @ self.W2 + self.b2)
        return self.a2 @ self.W3 + self.b3
    def backward(self, dz):
        n = len(dz)
        dW3 = self.a2.T @ dz / n; db3 = dz.mean(0)
        d2 = (dz @ self.W3.T) * (1 - self.a2**2)
        dW2 = self.a1.T @ d2 / n; db2 = d2.mean(0)
        d1 = (d2 @ self.W2.T) * (1 - self.a1**2)
        dW1 = self.X.T @ d1 / n; db1 = d1.mean(0)
        return [dW1, db1, dW2, db2, dW3, db3]

def train(Xtr, ytr, Xte, yte, model, epochs=args.epochs, lr=1e-3, wd=1e-4):
    m = [np.zeros_like(p) for p in model.params]
    v = [np.zeros_like(p) for p in model.params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    ytr2 = ytr[:, None]
    for ep in range(1, epochs + 1):
        z = model.forward(Xtr)
        dz = sigmoid(z) - ytr2                       # dBCE/dz
        grads = model.backward(dz)
        for i, (p, g) in enumerate(zip(model.params, grads)):
            g = g + wd * p
            m[i] = b1*m[i] + (1-b1)*g
            v[i] = b2*v[i] + (1-b2)*g*g
            p -= lr * (m[i]/(1-b1**ep)) / (np.sqrt(v[i]/(1-b2**ep)) + eps)
    acc = lambda X, y: float((((model.forward(X)[:, 0]) > 0) == (y > 0.5)).mean())
    return acc(Xtr, ytr), acc(Xte, yte)

def linear(nf): return Linear(nf, rng)
def mlp(nf):    return MLP(nf, rng)

results = {}
# reservoir features
for name, make in (("reservoir/linear", linear), ("reservoir/MLP", mlp)):
    a_tr, a_te = train(Xs[tr], Y[tr], Xs[te], Y[te], make(Xs.shape[1]))
    results[name] = (a_tr, a_te)
# raw-input controls
Ur = (U - U[tr].mean(0)) / (U[tr].std(0) + 1e-12)
for name, make in (("raw-inputs/linear", linear), ("raw-inputs/MLP", mlp)):
    a_tr, a_te = train(Ur[tr], Y[tr], Ur[te], Y[te], make(4))
    results[name] = (a_tr, a_te)
# shuffle control on the best reservoir features
Ysh = Y.copy(); rng.shuffle(Ysh)
a_tr, a_te = train(Xs[tr], Ysh[tr], Xs[te], Ysh[te], linear(Xs.shape[1]))
results["reservoir/linear SHUFFLED"] = (a_tr, a_te)

print(f"XOR(sign u{args.strips[0]}, sign u{args.strips[1]}), features={args.features} "
      f"({Xs.shape[1]}), train {M-args.test}/test {args.test}")
for k, (a, b) in results.items():
    print(f"  {k:28s} train {a*100:5.1f}%  TEST {b*100:5.1f}%")
if args.out:
    np.savez(args.out, **{k.replace("/", "_").replace(" ", "_"): v for k, v in results.items()})
