"""Extract MEEP's EXACT Yee x-coordinates for Ey (and Dielectric), and print
gpumeep's Ey x-grid, so we can align gpu's grid to MEEP without guessing."""
import os, sys, importlib
import numpy as np
RESV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RESV); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json  # noqa: E402

path = build_json(4)

import meep as mp
from class_simulation import Simulation
sim = Simulation(path); sim._set_everything(); sim.simulation.init_sim()
# Ey component coordinates
sim.simulation.get_array(component=mp.Ey)
md_ey = sim.simulation.get_array_metadata()
xey = np.asarray(md_ey[0])
sim.simulation.get_array(component=mp.Dielectric)
md_d = sim.simulation.get_array_metadata()
xd = np.asarray(md_d[0])
print(f"MEEP Ey   x[0:4]={xey[:4]}  n={len(xey)}  dx={xey[1]-xey[0]:.6f}")
print(f"MEEP Diel x[0:4]={xd[:4]}   n={len(xd)}")

# gpu grid (current code, with gx fix)
import jax; jax.config.update("jax_enable_x64", True)
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
g = csg.SimulationGPU(folder_path=path); g.force_fullvector = True
g._set_data(); g._update_all_args()
xg_new = np.arange(g.Nx) * g.dx - g.gx / 2          # Ey x with rounded-centered grid
xg_old = np.arange(g.Nx) * g.dx - g.cell_x / 2      # Ey x with old (unrounded) origin
print(f"gpu Ey x[0:4] (gx-centered)={xg_new[:4]}")
print(f"gpu Ey x[0:4] (old -cell/2)={xg_old[:4]}")
print(f"offset MEEP-Ey − gpu(gx)  = {xey[0]-xg_new[0]:+.6f}  ({(xey[0]-xg_new[0])/g.dx:+.3f} px)")
print(f"offset MEEP-Ey − gpu(old) = {xey[0]-xg_old[0]:+.6f}  ({(xey[0]-xg_old[0])/g.dx:+.3f} px)")
