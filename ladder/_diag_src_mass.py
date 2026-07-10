"""Compare the total integrated source current mass (sum of amp_map weights
over the source extent) between gpu and MEEP, for config 1 (simpler: no mirrors).
If gpu's fractional-cell weights don't sum to exactly MEEP's, a small amplitude
shift results. Also compare the src_scale (gv.a factor) dimensionality."""
import os, sys, importlib
import numpy as np
_RES = 40
os.environ["LADDER_RES"] = str(_RES)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json  # noqa: E402

path = build_json(1)  # vacuum, simpler

# --- gpu ---
import jax; jax.config.update("jax_enable_x64", True)
gpu_src = os.environ.get("GPUMEEP_PATH")
sys.path.insert(0, gpu_src); sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
g = csg.SimulationGPU(folder_path=path); g.force_fullvector = True
g._set_data(); g._update_all_args()
x_meep = g.objects_args[5]["center_x_meep"]    # source_1 (index 5 in order)
sy = g.objects_args[5]["size_y_meep"]
comp = "Ey"; xoff, yoff = 0.0, 0.5
ycen = (np.arange(g.Ny) + yoff) * g.dx - g.cy
wy = csg._src_overlap_weights(-sy/2, sy/2, g.Ny, g.dx, yoff, g.cy)
wx = csg._src_delta_weights(x_meep, g.Nx, g.dx, xoff, g.cx)
src_mass_gpu = float(wx.sum()) * float(wy.sum())  # total weight
# gv.a = res for Ey line source (1 zero-size dir)
src_scale = g.resolution
print(f"MEEP gv.a (res) = {g.resolution}")
print(f"gpu: Σwx={float(wx.sum()):.6f}  Σwy={float(wy.sum()):.6f}  total_mass={src_mass_gpu:.6f}")
wxy = wx[:, None] * wy[None, :]
n_nonzero = int(np.sum(wxy > 1e-12))
print(f"gpu: n_nonzero_source_cells = {n_nonzero}")

# --- MEEP ---
import meep as mp  # noqa: E402
from meep.simulation import py_v3_to_vec  # noqa: E402
from class_simulation import Simulation  # noqa: E402
sm = Simulation(path); sm._set_everything(); sm.simulation.init_sim()
f = sm.simulation.fields
# MEEP source amplitude: each source[] has .amp field which is the full complex amplitude
# for a point source, or per-point for volume. Our Ey is a line, each point gets the same amp.
for sv in sm.simulation.sources:
    c = sv.c
    if c == mp.Ey:
        # For a 0-x-size plane source, MEEP multiplies by gv.a (res) for the delta dir
        # and weights each point by amp * IVEC_LOOP_WEIGHT
        print(f"MEEP: component={component_name(c)} is_integrated={sv.t().is_integrated}")
        # count points
        npts = len(sv.index)
        print(f"MEEP: npts={npts}  amp[0]={sv.amp[0]:.6f}")
# MEEP's dipole normalization: 1/(-2*pi*i*freq) ≈ 1/(-2*pi*i*2.0)
f0 = 2.0
m_dip_amp = 1.0 / (-2.0j * np.pi * f0)
print(f"MEEP dipole amp 1/(-2πif) = {m_dip_amp:.12f}  |amp|={abs(m_dip_amp):.12f}")
# MEEP dt_factor
mdt = sm.simulation.fields.dt
print(f"MEEP dt = {mdt:.12f}  dt/sqrt(2π) = {mdt/np.sqrt(2*np.pi):.12f}")
print(f"gpu  dt = {g.dt:.12f}  dt/sqrt(2π) = {g.dt/np.sqrt(2*np.pi):.12f}")
# gpu DFT scale
print(f"DFT scale ratio gpu/meep: (gpu_dt/sqrt(2π)) / (meep_dt/sqrt(2π)) = "
      f"{(g.dt/np.sqrt(2*np.pi))/(mdt/np.sqrt(2*np.pi)):.6f}")

# --- Source amplitude at peak ---
# gpu: src_scale * _J(peak) * amp weight, then * dt at injection
# MEEP: dipole(peak) * gv.a, finite-diff'd by current(), then * dt, injected into D
# The product gv.a * dt * (dipole(t+dt)-dipole(t))/dt = gv.a * (dipole(t+dt)-dipole(t))
# gpu: res * dt * (dipole(t+dt)-dipole(t))/dt = res * (dipole(t+dt)-dipole(t))
# RESULT: gpu = res * dipole_diff, MEEP = res * dipole_diff → IDENTICAL. OK.
print("\nSource injection per-point: gv.a * (dipole(t+dt)-dipole(t)) — identical.")
print("DONE_SRCMASS")
