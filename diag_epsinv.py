import os, sys, numpy as np
sys.path.insert(0, "/home/cernez/resevoir")
os.environ.pop("MEEP_NO_SUBPIXEL",None)
from ladder.ladder import build_json
p = build_json(2)
import importlib.util
def load_gpu():
    sys.modules.pop("class_simulation_gpu",None)
    s=importlib.util.spec_from_file_location("class_simulation_gpu","/home/cernez/resevoir/class_simulation_gpu.py")
    m=importlib.util.module_from_spec(s); sys.modules["class_simulation_gpu"]=m; s.loader.exec_module(m); return m
import jax; jax.config.update("jax_enable_x64", True)
csg = load_gpu()
sg = csg.SimulationGPU(folder_path=p, force_fullvector=True)
sg._set_data(); sg._update_all_args(); sg._build_material()
Nx,Ny,dx,cx,cy=sg.Nx,sg.Ny,sg.dx,sg.cell_x,sg.cell_y
# gpumeep epinv at Ey-face
iyy = np.array(sg.material.iyy_Ey)  # (eps^-1)_yy at Ey-face

import meep as mp
from class_simulation import Simulation
sm = Simulation(p); sm._set_everything(); sm.simulation.init_sim()

# Sample at a few Ey-face points in the reservoir
res = next(o for o in sg.objects_args if o.get("class") in ("reservoir","voltage_reservoir"))
rx0=res["edge_x_meep"]; rx1=rx0+float(res["size_x"])
print("Reservoir Ey-face points:")
for ix in range(int(rx0+cx/2)/1, int(rx1+cx/2)/1, 5):
    if ix>=Nx: break
    x = ix*dx - cx/2
    jm = Ny//2
    y = (jm+0.5)*dx - cy/2
    em = complex(sm.simulation.get_epsilon_point(mp.Vector3(x,y,0),mp.Ey)).real
    eg = 1.0/iyy[ix,jm]  # forward epsilon from inverse
    print(f"  x={x:.3f}: MEEP_eyy={em:.6f}  gpu_eyy={eg:.6f}  diff={eg-em:.6f}")
# also check chi1inv
em_inv = complex(sm.simulation.get_epsilon_inv_point(mp.Vector3(x,y,0),mp.Ey)).real
print(f"  MEEP epinv at last pt: {em_inv:.6f}  gpu epinv: {iyy[ix,jm]:.6f}")
print("DONE")
