"""Compare MEEP get_epsilon vs gpumeep per-cell eps along DBR stack axis (config 4, y=0)."""
import os, sys, numpy as np
sys.path.insert(0, "/home/cernez/resevoir")
os.environ.pop("GPUMEEP_KOTTKE", None); os.environ.pop("MEEP_NO_SUBPIXEL", None)
from ladder.ladder import build_json
p = build_json(4)

import meep as mp
from class_simulation import Simulation
sm = Simulation(p); sm._set_everything(); sm.simulation.init_sim()
cx = sm._cell_x; res = sm.resolution
xs = np.linspace(-cx/2+0.02, cx/2-0.02, int(cx*res*4))   # fine sampling
eps_m_yy = np.array([sm.simulation.get_epsilon_point(mp.Vector3(x,0,0), mp.Ey) for x in xs])
eps_m_xx = np.array([sm.simulation.get_epsilon_point(mp.Vector3(x,0,0), mp.Ex) for x in xs])
print("MEEP  eps_yy range %.4f..%.4f" % (eps_m_yy.min(), eps_m_yy.max()))

import importlib.util
sys.modules.pop("class_simulation_gpu", None)
_spec = importlib.util.spec_from_file_location("class_simulation_gpu", "/home/cernez/resevoir/class_simulation_gpu.py")
csg = importlib.util.module_from_spec(_spec); sys.modules["class_simulation_gpu"] = csg; _spec.loader.exec_module(csg)
sg = csg.SimulationGPU(folder_path=p, force_fullvector=True)
sg._set_data(); sg._update_all_args(); sg._build_material()
iyy = np.array(sg.material.iyy_Ey); ixx = np.array(sg.material.ixx_Ex)
jm = sg.Ny // 2
xg = np.arange(sg.Nx) * sg.dx - sg.cell_x / 2      # Ey face x = node i (x_off=0)
eps_g_yy = 1.0 / iyy[:, jm]; eps_g_xx = 1.0 / ixx[:, jm]
print("gpu   eps_yy range %.4f..%.4f" % (eps_g_yy.min(), eps_g_yy.max()))

# effective average eps over the mirror span (this sets reflectivity)
mir = next(o for o in sg.objects_args if o.get("class")=="mirror")
lay = sg._mirror_layers(mir)
mx0 = min(l[0] for l in lay); mx1 = max(l[1] for l in lay)
mm = (xs>=mx0)&(xs<=mx1); mg = (xg>=mx0)&(xg<=mx1)
print("mirror span x=[%.4f,%.4f]  MEEP<eps_yy>=%.4f  gpu<eps_yy>=%.4f" %
      (mx0, mx1, eps_m_yy[mm].mean(), eps_g_yy[mg].mean()))
print("MEEP<1/eps_yy>=%.4f gpu<1/eps_yy>=%.4f (harmonic-relevant)" %
      ((1/eps_m_yy[mm]).mean(), (1/eps_g_yy[mg]).mean()))
np.savez("/home/cernez/resevoir/diag_eps.npz", xs=xs, m_yy=eps_m_yy, m_xx=eps_m_xx,
         xg=xg, g_yy=eps_g_yy, g_xx=eps_g_xx, mx0=mx0, mx1=mx1)
print("DONE")
