"""Compare MEEP get_epsilon vs gpumeep material ε at identical Yee points across the LC reservoir (config 2)."""
import os, sys, numpy as np
sys.path.insert(0, "/home/cernez/resevoir")
os.environ.pop("GPUMEEP_KOTTKE",None); os.environ.pop("GPUMEEP_AREAFRAC",None); os.environ.pop("MEEP_NO_SUBPIXEL",None)
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
Nx,Ny,dx,cx,cy = sg.Nx,sg.Ny,sg.dx,sg.cell_x,sg.cell_y
# gpu eps at Ey-face (i, j+1/2): 1/iyy_Ey
iyy = np.array(sg.material.iyy_Ey); ixx = np.array(sg.material.ixx_Ex)
eps_g_yy = 1.0/iyy; eps_g_xx = 1.0/ixx

import meep as mp
from class_simulation import Simulation
sm = Simulation(p); sm._set_everything(); sm.simulation.init_sim()
# reservoir region
res = next(o for o in sg.objects_args if o.get("class") in ("reservoir","voltage_reservoir"))
rx0 = res["edge_x_meep"]; rx1 = rx0 + float(res["size_x"])
# sample a horizontal line at y=0 (j=Ny//2) and vertical line at reservoir center
jm = Ny//2
xs = np.arange(Nx)*dx - cx/2
ax_meep = np.array([complex(sm.simulation.get_epsilon_point(mp.Vector3(x, (jm+0.5)*dx-cy/2, 0), mp.Ey)).real for x in xs])
inres = (xs>=rx0)&(xs<rx1)
print("Ey-face eps at y=0, reservoir x-span:")
print("  MEEP  mean=%.5f  gpu mean=%.5f  ratio=%.5f" % (ax_meep[inres].mean(), eps_g_yy[inres,jm].mean(), eps_g_yy[inres,jm].mean()/ax_meep[inres].mean()))
print("  max|gpu-MEEP|=%.5f  rms=%.5f" % (np.abs(eps_g_yy[inres,jm]-ax_meep[inres]).max(), np.sqrt(((eps_g_yy[inres,jm]-ax_meep[inres])**2).mean())))
# full-reservoir 2D mean over a grid of points
xr = np.linspace(rx0+0.05, rx1-0.05, 25); yr = np.linspace(-2.5,2.5,25)
dm=[]; dg=[]
for x in xr:
    i = int(round((x+cx/2)/dx))
    for y in yr:
        j = int(round((y+cy/2)/dx))
        em = complex(sm.simulation.get_epsilon_point(mp.Vector3(x,(j+0.5)*dx-cy/2,0), mp.Ey)).real
        dm.append(em); dg.append(eps_g_yy[i,j])
dm=np.array(dm); dg=np.array(dg)
print("Reservoir 2D grid Ey eps: MEEP mean=%.5f gpu mean=%.5f ratio=%.5f rms-diff=%.5f maxdiff=%.5f" % (
  dm.mean(), dg.mean(), dg.mean()/dm.mean(), np.sqrt(((dg-dm)**2).mean()), np.abs(dg-dm).max()))
print("DONE")
