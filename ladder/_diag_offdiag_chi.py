"""Directly compare gpu M.ixy_Ex (off-diagonal chi_xy at Ex face) against MEEP's
chi1inv[Ex][Y] array, index-by-index along a horizontal cut, to detect any
half-cell / one-index layout shift between the two engines' tensors.

If they align at shift 0, the OFFDIAG stencil premise is right and the mismatch
is in the material build; if best alignment is at shift +/-1, THAT is the bug
making the copied OFFDIAG stencil diverge.
"""
import os, sys, importlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ladder import build_json
import meep as mp
from meep.simulation import py_v3_to_vec
from class_simulation import Simulation

p = build_json(2)
sm = Simulation(p); sm._set_everything(); sm.simulation.init_sim()
f = sm.simulation.fields; DIM = sm.simulation.dimensions
def vec(x, y): return py_v3_to_vec(DIM, mp.Vector3(x, y), False)

import jax; jax.config.update("jax_enable_x64", True)
cpu = os.environ.get("GPUMEEP_PATH", "")
sys.path.insert(0, cpu)
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
g = csg.SimulationGPU(folder_path=p); g.force_fullvector = True
g._set_data(); g._update_all_args(); g._build_material()

ixy_g = np.array(g.material.ixy_Ex)      # gpu chi_xy at Ex face (i+1/2, j)
jmid = g.Ny // 2
dx = g.dx

# Sample MEEP chi1inv[Ex][Y] along x at the Ex-face y-level (node j = jmid)
yn = jmid * dx - g.cy
xs = np.arange(g.Nx)
ixy_m = np.array([f.get_chi1inv(mp.Ex, mp.Y, vec((ii + 0.5) * dx - g.cx, yn), 0.0, True).real
                  for ii in xs])

# restrict to the LC reservoir x-window where chi_xy != 0
mask = np.abs(ixy_m) > 1e-9
i0 = np.argmax(mask); i1 = len(mask) - np.argmax(mask[::-1])
sl = slice(i0 + 2, i1 - 2)
a = ixy_m[sl]; b = ixy_g[sl, jmid]
print(f"LC x-window i=[{i0},{i1}) N={i1-i0}")
print(f"MEEP chi_xy[Ex][Y] range [{a.min():+.4f},{a.max():+.4f}]  gpu ixy_Ex range [{b.min():+.4f},{b.max():+.4f}]")
for sh in (-2, -1, 0, 1, 2):
    bb = np.roll(ixy_g[:, jmid], sh)[sl]
    rms = np.sqrt(np.mean((a - bb) ** 2))
    cor = np.corrcoef(a, bb)[0, 1]
    print(f"  shift {sh:+d}: rms(MEEP-gpu)={rms:.5f}  corr={cor:.5f}")
# show first mismatched samples at shift 0
print("first 8 in-window (x-idx: MEEP  gpu  diff):")
for k in range(i0 + 2, i0 + 10):
    print(f"  i={k}: {ixy_m[k]:+.5f}  {ixy_g[k,jmid]:+.5f}  {ixy_m[k]-ixy_g[k,jmid]:+.5f}")
print("DONE_OFFDIAG_CHI")
