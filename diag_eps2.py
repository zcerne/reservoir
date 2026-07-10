"""Compare MEEP vs gpumeep eps_yy at the EXACT gpu Ey Yee points (config 4)."""
import os, sys, numpy as np
sys.path.insert(0, "/home/cernez/resevoir")
os.environ.pop("GPUMEEP_KOTTKE", None); os.environ.pop("MEEP_NO_SUBPIXEL", None)
from ladder.ladder import build_json
p = build_json(4)
import importlib.util
def load_gpu():
    sys.modules.pop("class_simulation_gpu", None)
    s = importlib.util.spec_from_file_location("class_simulation_gpu", "/home/cernez/resevoir/class_simulation_gpu.py")
    m = importlib.util.module_from_spec(s); sys.modules["class_simulation_gpu"]=m; s.loader.exec_module(m); return m
csg = load_gpu()
sg = csg.SimulationGPU(folder_path=p, force_fullvector=True)
sg._set_data(); sg._update_all_args(); sg._build_material()
jm = sg.Ny // 2
xg = np.arange(sg.Nx) * sg.dx - sg.cell_x / 2       # Ey face x-positions (x_off=0)
yg = (jm + 0.5) * sg.dx - sg.cell_y / 2             # Ey face y (j+1/2)
eps_g = 1.0 / np.array(sg.material.iyy_Ey)[:, jm]

import meep as mp
from class_simulation import Simulation
sm = Simulation(p); sm._set_everything(); sm.simulation.init_sim()
eps_m = np.array([complex(sm.simulation.get_epsilon_point(mp.Vector3(x, yg, 0), mp.Ey)).real for x in xg])

# mirror-1 span
mir = next(o for o in sg.objects_args if o.get("class")=="mirror")
lay = sg._mirror_layers(mir); mx0=min(l[0] for l in lay); mx1=max(l[1] for l in lay)
w = (xg>=mx0-0.05)&(xg<=mx1+0.05)
print("At IDENTICAL Yee points, mirror-1 region:")
print("  MEEP<eps_yy>=%.4f  gpu<eps_yy>=%.4f  (ratio %.4f)" % (eps_m[w].mean(), eps_g[w].mean(), eps_g[w].mean()/eps_m[w].mean()))
print("  max|gpu-MEEP|=%.3f  rms=%.4f" % (np.abs(eps_g[w]-eps_m[w]).max(), np.sqrt(((eps_g[w]-eps_m[w])**2).mean())))
np.savez("/home/cernez/resevoir/diag_eps2.npz", xg=xg, eps_m=eps_m, eps_g=eps_g, mx0=mx0, mx1=mx1)
print("DONE")
