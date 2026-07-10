"""Compare the field after 1 step (source injection only, no propagation) between
gpu and MEEP for config 4. If the injected D/Ey differ, the source coupling is off.
If identical, the propagation (D-form vs EH-form) is the difference."""
import os, sys, importlib
import numpy as np
os.environ["LADDER_RES"] = "40"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ladder import build_json

path = build_json(4)

# --- MEEP: run 1 step, read Ey at source ---
import meep as mp
from class_simulation import Simulation
sm = Simulation(path); sm._set_everything(); sm.simulation.init_sim()
f = sm.simulation.fields
# Get the Ey component at the source x-line after 1 step
# Run 1 step manually
from meep.simulation import py_v3_to_vec
DIM = sm.simulation.dimensions
def _vec(x, y): return py_v3_to_vec(DIM, mp.Vector3(x, y), False)

# get source center
src_x = float(sm.simulation.sources[0].center.x)
mdt = f.dt
print(f"MEEP dt={mdt:.6f}")
# Step 1: MEEP's step() does the whole timestep
f.step()
# Read Ey along the source x-line
Ny = int(sm.simulation.fields.gv.ny())
# Read Ey after step 1 at source x, all y
ys = np.zeros(Ny, dtype=complex)
yc = np.zeros(Ny)
for j in range(Ny):
    y = (j + 0.5) * (1.0/40.0) - 4.5  # Ey y-location (j+0.5)*dy - cell_y/2
    yc[j] = y
    ys[j] = f.get_field(mp.Ey, _vec(src_x, y))
max_ey_meep = np.abs(ys).max()
print(f"MEEP step1: max|Ey| at src = {max_ey_meep:.6g}")

# --- gpu: same, run 1 step ---
import jax; jax.config.update("jax_enable_x64", True)
gpu_src = os.environ.get("GPUMEEP_PATH")
sys.path.insert(0, gpu_src); sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
g = csg.SimulationGPU(folder_path=path); g.force_fullvector = True
g._set_data(); g._update_all_args()
# Build sources + material, then step once
g._build_material()
g._build_sources_sted()
g._build_pml_full()

from fdtd_2d import zero_fields_full, zero_D_full, step_2d_full_dform

D = zero_D_full(g.grid)
fields = zero_fields_full(g.grid)
# Create DFT states (won't be used for 1 step, but needed for the loop)
# Just step once manually: apply source, step
t = 0.0
src = g.sources[0]
D = src.apply_D(D, t)
D, fields, pml = step_2d_full_dform(D, fields, g.grid, g.dt, g.pml, g.material)

# Read Ey at the source plane (index i_src)
xmeep = g.objects_args[5]["center_x_meep"]
isrc = csg._meep_to_grid_x(xmeep, g.cx, g.dx)
ey_gpu = np.asarray(fields.Ey[isrc, :])
max_ey_gpu = np.abs(ey_gpu).max()
print(f"gpu step1: max|Ey| at src = {max_ey_gpu:.6g}")
print(f"ratio gpu/meep = {max_ey_gpu/max_ey_meep:.6f}")
print("DONE_FIRSTSTEP")
