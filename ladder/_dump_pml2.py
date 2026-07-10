import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ladder import build_json
from class_simulation import Simulation

path = build_json(4)
sm = Simulation(path); sm._set_everything(); sm.simulation.init_sim()

for ch in sm.simulation.fields.chunks:
    if not ch.is_mine():
        continue
    s = ch.s
    gv = ch.gv
    print("Nx=%d Ny=%d" % (gv.nx(), gv.ny()))
    for di, dn in [(0, "X"), (1, "Y")]:
        sig = s.sig[di]
        if sig is not None:
            n = s.sigsize[di]
            kap = s.kap[di]; sinv = s.siginv[di]
            print("PML_%s size=%d sig0=%.8f sigN=%.8f kap0=%.8f kapN=%.8f sinv0=%.8f sinvN=%.8f"
                  % (dn, n, sig[0], sig[-1], kap[0], kap[-1], sinv[0], sinv[-1]))
        for cc in [0, 1]:
            cs = s.conductivity[cc]
            if cs is not None and cs[di] is not None:
                cnd = cs[di]
                print("  cond[cc=%d][%s]: len=%d max=%.6f" % (cc, dn, len(cnd), max(cnd)))
    break
print("DONE")
