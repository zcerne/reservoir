"""Parity of THREE strips (odd function) with the sign-faithful decodes:
signed amplitude sign(phase)*|E| and strictly-linear Re/Im. If saturable gain
writes odd field products (u_i u_j u_k), these decodes should beat chance here
even though they sit at chance for pair-XOR (even function)."""
import numpy as np
S = "/tmp/claude-1000/-home-ziga/618a2443-12aa-4145-9641-203874d25f7f/scratchpad"
d = np.load(f"{S}/ipc_05d.npz")
U = np.asarray(d["inputs"]); out = np.asarray(d["outputs"])
freqs = np.asarray(d["m2_freqs"]); nlam = len(freqs)
M = len(U); ny = out.shape[1] // nlam
Ek = out.reshape(M, nlam, ny)[:, int(np.argmin(np.abs(freqs - 1/1.064))), :]
rng = np.random.default_rng(0)
idx = rng.permutation(M); tr, te = idx[:-200], idx[-200:]
A = np.concatenate([np.ones((M, 1)), U], axis=1)
coef, *_ = np.linalg.lstsq(A[tr], Ek[tr], rcond=None)
Elin = A @ coef; E0 = coef[0]; ref = E0/np.abs(E0)
def signed(E): return np.sign((E*np.conj(ref)).real)*np.abs(E)

def run(X, Y, name):
    mu, sd = X[tr].mean(0), X[tr].std(0)+1e-12
    Xs = (X-mu)/sd
    W = rng.normal(0,1/np.sqrt(X.shape[1]),(X.shape[1],1)); b=np.zeros(1)
    m=[0*W,0*b]; v=[0*W,0*b]; y2=Y[tr][:,None]
    for ep in range(1,401):
        dz = 1/(1+np.exp(-(Xs[tr]@W+b))) - y2
        g=[Xs[tr].T@dz/len(dz)+1e-4*W, dz.mean(0)]
        for i,(p,gr) in enumerate(zip([W,b],g)):
            m[i]=.9*m[i]+.1*gr; v[i]=.999*v[i]+.001*gr*gr
            p -= 1e-3*(m[i]/(1-.9**ep))/(np.sqrt(v[i]/(1-.999**ep))+1e-8)
    acc=lambda ii: float((((Xs[ii]@W+b)[:,0]>0)==(Y[ii]>.5)).mean())
    print(f"{name:46s} train {acc(tr)*100:5.1f}%  TEST {acc(te)*100:5.1f}%")

for label, Y in [("XOR3 = parity(u0,u1,u2) [ODD]", (U[:,0]*U[:,1]*U[:,2] < 0).astype(float)),
                 ("XOR2 = sign(u0 u1) [EVEN, reference]", (U[:,0]*U[:,1] < 0).astype(float))]:
    print(label)
    run(signed(Ek), Y, "  signed amp, REAL reservoir")
    run(signed(Elin), Y, "  signed amp, LINEAR surrogate")
    run(np.concatenate([Ek.real,Ek.imag],1), Y, "  Re/Im (strictly linear), REAL")
    run(np.abs(Ek), Y, "  |E| , REAL")
