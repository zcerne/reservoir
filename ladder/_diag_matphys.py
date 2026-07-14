"""Validate gpumeep Lorentzian/Drude/conductivity/chi3 vs MEEP.

Periodic 2D cell (no PML — proven bit-exact regime), a material slab
|x|<=1.25, Ey plane pulse; point probe Ey(t) at slab center compared
sample-by-sample for four material cases.
"""
import os, sys
import numpy as np

CASE = os.environ.get("CASE", "lorentzian")
N_STEPS = int(os.environ.get("DIAG_STEPS", "3000"))
res = 40; dx = 1.0 / res; dt = 0.5 * dx
CX, CY = 8.0, 4.0
Nx, Ny = int(CX * res), int(CY * res)

import meep as mp  # noqa: E402

if CASE == "lorentzian":
    med = mp.Medium(epsilon=2.25, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=1.1, gamma=1e-5, sigma=0.5)])
elif CASE == "drude":
    med = mp.Medium(epsilon=2.25, E_susceptibilities=[
        mp.DrudeSusceptibility(frequency=1.1, gamma=1e-5, sigma=0.5)])
elif CASE == "cond":
    med = mp.Medium(epsilon=2.25, D_conductivity=2.0)
elif CASE == "chi3":
    med = mp.Medium(epsilon=2.25, chi3=0.2)
else:
    raise SystemExit(f"unknown CASE {CASE}")

sim = mp.Simulation(
    cell_size=mp.Vector3(CX, CY, 0), resolution=res, boundary_layers=[],
    k_point=mp.Vector3(0, 0, 0),
    geometry=[mp.Block(center=mp.Vector3(0, 0),
                       size=mp.Vector3(2.5, mp.inf, mp.inf), material=med)],
    sources=[mp.Source(mp.GaussianSource(2.0, fwidth=0.5), component=mp.Ey,
                       center=mp.Vector3(-3.0, 0), size=mp.Vector3(0, CY),
                       amplitude=1.0)])
sim.init_sim()
assert abs(sim.fields.dt - dt) < 1e-15
tm = []


def _rec(sim_):
    tm.append(np.real(complex(sim_.get_field_point(mp.Ey, mp.Vector3(0.0, 0.0125)))))


sim.run(_rec, until=N_STEPS * dt - 1e-9)
tm = np.array(tm)
print(f"MEEP [{CASE}]: {len(tm)} steps, max |Ey| = {np.abs(tm).max():.6f}")

# ---------------- gpumeep ----------------
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src"))
import gpumeep as gm  # noqa: E402

if CASE == "lorentzian":
    gmed = gm.Medium(epsilon=2.25, E_susceptibilities=[
        gm.LorentzianSusceptibility(frequency=1.1, gamma=1e-5, sigma=0.5)])
elif CASE == "drude":
    gmed = gm.Medium(epsilon=2.25, E_susceptibilities=[
        gm.DrudeSusceptibility(frequency=1.1, gamma=1e-5, sigma=0.5)])
elif CASE == "cond":
    gmed = gm.Medium(epsilon=2.25, D_conductivity=2.0)
else:
    gmed = gm.Medium(epsilon=2.25, chi3=0.2)

gsim = gm.Simulation(
    cell_size=gm.Vector3(CX, CY), resolution=res, boundary_layers=[],
    k_point=gm.Vector3(0, 0, 0),
    geometry=[gm.Block(center=gm.Vector3(0, 0),
                       size=gm.Vector3(2.5, gm.inf), material=gmed)],
    sources=[gm.Source(gm.GaussianSource(2.0, fwidth=0.5), component=gm.Ey,
                       center=gm.Vector3(-3.0, 0), size=gm.Vector3(0, CY))],
    verbose=False)
gsim.init_sim()
ip = int(round((0.0 + gsim.cx) / dx))
jp = int(round((0.0125 + gsim.cy) / dx - 0.5))
tg = gsim._run_steps(N_STEPS, record_pt=("Ey", (ip, jp, 0)))
tg = np.asarray(tg)
print(f"GPU  [{CASE}]: max |Ey| = {np.abs(tg).max():.6f}")

best = None
for sh in (0, 1):
    a = tm[sh:N_STEPS]; b = tg[:len(a)]
    k = min(len(a), len(b)); a, b = a[:k], b[:k]
    rel = np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-300)
    if best is None or rel < best[0]:
        best = (rel, sh, a, b)
rel, sh, a, b = best
print(f"{CASE}: shift {sh:+d}  rel-L2 = {rel:.6e}")
bad = np.nonzero(np.abs(b - a) > 1e-9 * np.abs(a).max())[0]
if len(bad):
    i0 = bad[0]
    print(f"   first divergence step {i0}: meep={a[i0]:.9e} gpu={b[i0]:.9e}")
else:
    print("   bit-exact within 1e-9 of peak")
