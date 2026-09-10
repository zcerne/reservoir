"""User decode: signed amplitude sign(phase)*|E| per pixel.
Phase reference per pixel = phase of E0 (intercept of the linear field fit),
so sign = sign(cos(phi - phi_ref)) = sign(Re(E * conj(E0))). Compare real
reservoir vs linear-field surrogate vs residual, plus the fully-linear
quadrature projection Re(E e^{-i phi_ref}) for reference."""
import numpy as np
S = "/tmp/claude-1000/-home-ziga/618a2443-12aa-4145-9641-203874d25f7f/scratchpad"
d = np.load(f"{S}/ipc_05d.npz")
U = np.asarray(d["inputs"]); out = np.asarray(d["outputs"])
freqs = np.asarray(d["m2_freqs"]); nlam = len(freqs)
M = len(U); ny = out.shape[1] // nlam
Ek = out.reshape(M, nlam, ny)[:, int(np.argmin(np.abs(freqs - 1/1.064))), :]
Y = (U[:, 0] * U[:, 1] < 0).astype(np.float64)
rng = np.random.default_rng(0)
idx = rng.permutation(M); tr, te = idx[:-200], idx[-200:]
A = np.concatenate([np.ones((M, 1)), U], axis=1)
coef, *_ = np.linalg.lstsq(A[tr], Ek[tr], rcond=None)
Elin = A @ coef
E0 = coef[0]                                  # per-pixel reference field
ref = E0 / np.abs(E0)

def signed(E): return np.sign((E * np.conj(ref)).real) * np.abs(E)
def quad(E):   return (E * np.conj(ref)).real   # linear projection

def run(X, name):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    Xs = (X - mu) / sd
    W = rng.normal(0, 1/np.sqrt(X.shape[1]), (X.shape[1], 1)); b = np.zeros(1)
    m = [0*W, 0*b]; v = [0*W, 0*b]; y2 = Y[tr][:, None]
    for ep in range(1, 401):
        dz = 1/(1+np.exp(-(Xs[tr] @ W + b))) - y2
        g = [Xs[tr].T @ dz / len(dz) + 1e-4*W, dz.mean(0)]
        for i, (p, gr) in enumerate(zip([W, b], g)):
            m[i] = .9*m[i]+.1*gr; v[i] = .999*v[i]+.001*gr*gr
            p -= 1e-3*(m[i]/(1-.9**ep))/(np.sqrt(v[i]/(1-.999**ep))+1e-8)
    acc = lambda ii: float((((Xs[ii] @ W + b)[:, 0] > 0) == (Y[ii] > .5)).mean())
    print(f"{name:46s} train {acc(tr)*100:5.1f}%  TEST {acc(te)*100:5.1f}%")

print("signed amplitude sign(phase)*|E|:")
run(signed(Ek),   "  REAL reservoir")
run(signed(Elin), "  LINEAR-FIELD surrogate")
run(signed(Ek) - signed(Elin), "  residual (medium only)")
print("linear quadrature Re(E e^-iphiref) [reference]:")
run(quad(Ek),     "  REAL reservoir")
# how binary is the phase actually? sanity
dphi = np.angle(Ek * np.conj(ref))
frac = np.mean(np.minimum(np.abs(dphi), np.pi - np.abs(dphi)) < np.pi/4)
print(f"[phase near 0 or pi within 45 deg: {frac*100:.1f}% of pixels]")
