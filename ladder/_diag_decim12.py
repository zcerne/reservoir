"""Verify DFT-decimation hypothesis for config 1.2.

The time-domain fields are bit-exact between engines, so compute BOTH DFT
quadratures from the SAME GPU time trace at the monitor point:
  full : sum_{m=1..N}  E^m e^{i w m dt} * dt/sqrt(2pi)      (GPU monitor)
  decim: sum_{m=q,2q..} E^m e^{i w m dt} * q*dt/sqrt(2pi)   (MEEP, q=18)
and compare their ratio with the observed GPU/MEEP = 1.0371 * e^{-i 3.445 deg}.
"""
import os, sys
import numpy as np

RESEVOIR = os.environ.get("RESEVOIR", "/home/cernez/resevoir")
sys.path.insert(0, RESEVOIR)
sys.path.insert(0, os.path.join(RESEVOIR, "ladder"))
import ladder  # noqa: E402

path = ladder.build_json(1.2)

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
gpu_src = os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src")
sys.path.insert(0, gpu_src)
import importlib  # noqa: E402
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
import fdtd_2d as f2  # noqa: E402
import jax.numpy as jnp  # noqa: E402

g = csg.SimulationGPU(folder_path=path)
g.force_fullvector = True
g._set_data(); g._update_all_args(); g._build_material()
g.dt = float(g.args.get("courant", 0.5)) * g.dx
g._build_pml_full(); g._build_sources_sted()

i_mon = int(round((2.75 + g.cx) / g.dx))   # monitor x (guide_2 center)
j_pr = 120
grid, dt, material, sources, pml = g.grid, g.dt, g.material, g.sources, g.pml
run_until, decay = 120.0, 50.0
N = int((run_until + decay) / dt)
print(f"N = {N} steps, monitor i = {i_mon}")


def body(state, i):
    D, f, p = state
    t = i * dt
    for src in sources:
        D = src.apply_D(D, t)
    D, f, p = f2.step_2d_full_dform(D, f, grid, dt, p, material)
    return (D, f, p), f.Ey[i_mon, j_pr]


state0 = (f2.zero_D_full(g.grid), f2.zero_fields_full(g.grid), pml)
_, trace = jax.lax.scan(body, state0, jnp.arange(N))
E = np.asarray(trace)              # E[m-1] = Ey^m at time m*dt, m = 1..N
print("trace max |Ey| =", np.abs(E).max())

omega = 2.0 * np.pi * 2.0
m = np.arange(1, N + 1)
ph = np.exp(1j * omega * m * dt)
full = np.sum(E * ph) * dt / np.sqrt(2 * np.pi)

for q in (16, 17, 18, 19, 20):
    sel = (m % q) == 0
    decim = np.sum(E[sel] * ph[sel]) * q * dt / np.sqrt(2 * np.pi)
    r = full / decim
    print(f"q={q:2d}: full/decim = {np.abs(r):.5f} * e^(i {np.degrees(np.angle(r)):+7.3f} deg)")

print("\nobserved GPU/MEEP  = 1.03710 * e^(i  -3.445 deg)")

# cross-check against the actual saved sensor values
sim_dir = os.path.join(path, "simulation")
try:
    a = np.load(os.path.join(sim_dir, "monitor_2_meep.npz"))["Ey"][0]
    b = np.load(os.path.join(sim_dir, "monitor_2_gpumeep.npz"))["Ey"][0]
    print(f"saved MEEP Ey[mid] = {a[len(a)//2]:.6g}")
    print(f"saved GPU  Ey[mid] = {b[len(b)//2]:.6g}")
    print(f"this-run full DFT  = {full:.6g}  (raw Yee, no collocation)")
except FileNotFoundError:
    pass
