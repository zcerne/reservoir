"""Dump MEEP's actual PML arrays for config 4 — sigma, kappa, siginv,
conductivity, condinv. These are the target values to replicate in gpu's UPML."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ladder import build_json
from class_simulation import Simulation

path = build_json(4)
sm = Simulation(path); sm._set_everything(); sm.simulation.init_sim()

for ch in sm.simulation.chunks:
    if not ch.is_mine():
        continue
    s = ch.s
    gv = ch.gv
    Nx, Ny = gv.nx(), gv.ny()
    res = int(1.0 / gv.inva)
    dx = 1.0 / res
    print(f"Grid: Nx={Nx} Ny={Ny} dx={dx:.4f} inva={gv.inva:.4f}")
    pmld = int(round(1.5 / dx))
    print(f"PML thickness = {pmld} cells")
    for d_idx, d_name in enumerate(["X", "Y", "Z"]):
        sig = s.sig[d_idx]
        kap = s.kap[d_idx]
        sinv = s.siginv[d_idx]
        cnd = s.conductivity[0][d_idx] if d_idx < 2 else None  # Ex component
        if sig is not None:
            N = s.sigsize[d_idx]
            print(f"\nPML {d_name}: sigsize={N}")
            print(f"  sig[0..4]  = {[sig[i] for i in range(min(5,N))]}")
            print(f"  sig[-5:]   = {[sig[i] for i in range(N-5,N)]}")
            print(f"  kap[0..4]  = {[kap[i] for i in range(min(5,N))]}")
            print(f"  kap[-5:]   = {[kap[i] for i in range(N-5,N)]}")
            print(f"  siginv[0..4]={[sinv[i] for i in range(min(5,N))]}")
            print(f"  siginv[-5:] ={[sinv[i] for i in range(N-5,N)]}")
    # Dump conductivity for D components
    for cc, cn in [(0, "Ex"), (1, "Ey"), (2, "Ez")]:
        for d_idx, d_name in enumerate(["X", "Y", "Z"]):
            if d_idx >= 2:
                break
            cnd = s.conductivity[cc][d_idx] if s.conductivity[cc] else None
            cdinv = s.condinv[cc][d_idx] if s.condinv[cc] else None
            if cnd is not None:
                nz = int(np.sum(np.abs(cnd) > 1e-10))
                print(f"  conductivity[{cn}][{d_name}]: nonzeros={nz}")
                if nz > 0:
                    idxs = np.nonzero(np.abs(cnd) > 1e-10)[0]
                    print(f"    at idx={idxs[:5]}: cnd={[cnd[i] for i in idxs[:5]]}")
    # Dump chi1inv for Ey component (should be 1 outside PML, attenuated in PML)
    c1 = s.chi1inv[1]  # Ey
    c1x = c1[1] if c1[1] else None  # chi1inv[Ey][X]  — cross-term
    c1y = c1[2] if c1[2] else None  # chi1inv[Ey][Y]  — along-component
    if c1y is not None:
        print(f"\nchi1inv[Ey][Y]: first 5 = {[c1y[i] for i in range(5)]}")
        print(f"  last 5 = {[c1y[i] for i in range(len(c1y)-5, len(c1y))]}")

print("DONE_PMLDUMP")
