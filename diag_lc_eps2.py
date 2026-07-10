"""CORRECT ε comparison: gpu FORWARD ε_yy at Ey-face vs MEEP ε_yy, config 2."""
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
csg = load_gpu()
sg = csg.SimulationGPU(folder_path=p, force_fullvector=True)
sg._set_data(); sg._update_all_args(); sg._setup_lc_interp()
# gpu FORWARD eps at Ey-face (i, j+1/2)
exx,eyy,ezz,exy,exz,eyz = sg._kottke_eps_at(0.0, 0.5)   # forward 3x3, 6 comps
Nx,Ny,dx,cx,cy = sg.Nx,sg.Ny,sg.dx,sg.cell_x,sg.cell_y

import meep as mp
from class_simulation import Simulation
sm = Simulation(p); sm._set_everything(); sm.simulation.init_sim()
res = next(o for o in sg.objects_args if o.get("class") in ("reservoir","voltage_reservoir"))
rx0=res["edge_x_meep"]; rx1=rx0+float(res["size_x"])
xr=np.linspace(rx0+0.1,rx1-0.1,25); yr=np.linspace(-2.5,2.5,25)
dm_yy=[];dg_yy=[];dm_xy=[];dg_xy=[]
for x in xr:
    i=int(round((x+cx/2)/dx))
    for y in yr:
        j=int(round((y+cy/2)/dx))
        yy_m=complex(sm.simulation.get_epsilon_point(mp.Vector3(x,(j+0.5)*dx-cy/2,0), mp.Ey)).real
        dm_yy.append(yy_m); dg_yy.append(eyy[i,j])
        dm_xy.append(0); dg_xy.append(exy[i,j])
dm_yy=np.array(dm_yy);dg_yy=np.array(dg_yy)
print("FORWARD eps_yy at Ey-face over reservoir:")
print("  MEEP mean=%.5f  gpu mean=%.5f  ratio=%.5f"%(dm_yy.mean(),dg_yy.mean(),dg_yy.mean()/dm_yy.mean()))
print("  rms-diff=%.5f maxdiff=%.5f"%(np.sqrt(((dg_yy-dm_yy)**2).mean()),np.abs(dg_yy-dm_yy).max()))
print("  gpu <exy>=%.4f (off-diag present: %s)"%(np.mean(dg_xy), np.abs(dg_xy).max()>0.01))
print("DONE")
