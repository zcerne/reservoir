"""Control: does |E| of the BEST LINEAR field model already solve XOR?
Fit E_lin(u) = e0 + sum_i u_i e_i (least squares, complex, per pixel) on the
TRAIN split, then train the same linear readout on |E_lin| features.
If this matches the real-reservoir 91%, the XOR comes from the detector
modulus acting on interference, not from medium mixing."""
import numpy as np, sys
sys.argv = ["x"]
S = "/tmp/claude-1000/-home-ziga/618a2443-12aa-4145-9641-203874d25f7f/scratchpad"
d = np.load(f"{S}/ipc_05d.npz")
U = np.asarray(d["inputs"]); out = np.asarray(d["outputs"])
freqs = np.asarray(d["m2_freqs"]); nlam = len(freqs)
M = len(U); ny = out.shape[1] // nlam
E = out.reshape(M, nlam, ny)
k = int(np.argmin(np.abs(freqs - 1/1.064)))
Ek = E[:, k, :]                                   # (M, 252) complex
Y = (U[:, 0] * U[:, 1] < 0).astype(np.float64)

rng = np.random.default_rng(0)
idx = rng.permutation(M); tr, te = idx[:-200], idx[-200:]

# linear field model fit on TRAIN only
A = np.concatenate([np.ones((M, 1)), U], axis=1)  # (M,5)
coef, *_ = np.linalg.lstsq(A[tr], Ek[tr], rcond=None)
Elin = A @ coef
res = np.linalg.norm(Ek - Elin, axis=1) / np.linalg.norm(Ek, axis=1)
print(f"linear-field model residual: median {np.median(res)*100:.2f}% of |E| (train fit)")

def run(X, name):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    Xs = (X - mu) / sd
    W = rng.normal(0, 1/np.sqrt(X.shape[1]), (X.shape[1], 1)); b = np.zeros(1)
    m = [np.zeros_like(W), np.zeros_like(b)]; v = [np.zeros_like(W), np.zeros_like(b)]
    y2 = Y[tr][:, None]
    for ep in range(1, 401):
        z = Xs[tr] @ W + b
        dz = 1/(1+np.exp(-z)) - y2
        g = [Xs[tr].T @ dz / len(dz) + 1e-4*W, dz.mean(0)]
        for i, (p, gr) in enumerate(zip([W, b], g)):
            m[i] = .9*m[i] + .1*gr; v[i] = .999*v[i] + .001*gr*gr
            p -= 1e-3 * (m[i]/(1-.9**ep)) / (np.sqrt(v[i]/(1-.999**ep)) + 1e-8)
    acc = lambda ii: float((((Xs[ii] @ W + b)[:, 0] > 0) == (Y[ii] > .5)).mean())
    print(f"{name:42s} train {acc(tr)*100:5.1f}%  TEST {acc(te)*100:5.1f}%")

run(np.abs(Ek), "REAL reservoir |E| / linear")
run(np.abs(Elin), "LINEAR-FIELD surrogate |E| / linear")
run(np.abs(Ek) - np.abs(Elin), "residual |E|-|E_lin| / linear")
