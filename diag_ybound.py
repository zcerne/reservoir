import os, sys, numpy as np
sys.path.insert(0, "/home/cernez/resevoir")
os.environ.pop("GPUMEEP_KOTTKE",None); os.environ.pop("MEEP_NO_SUBPIXEL",None)
from ladder.ladder import build_json
p = build_json(2)
import importlib.util
def load_gpu():
    sys.modules.pop("class_simulation_gpu",None)
    s=importlib.util.spec_from_file_location("class_simulation_gpu","/home/cernez/resevoir/class_simulation_gpu.py")
    m=importlib.util.module_from_spec(s); sys.modules["class_simulation_gpu"]=m; s.loader.exec_module(m); return m
csg=load_gpu(); sg=csg.SimulationGPU(folder_path=p,force_fullvector=True)
sg._set_data(); sg._update_all_args(); sg._setup_lc_interp()
Nx,Ny,dx,cx,cy=sg.Nx,sg.Ny,sg.dx,sg.cell_x,sg.cell_y
res=next(o for o in sg.objects_args if o.get("class") in ("reservoir","voltage_reservoir"))
rx0=res["edge_x_meep"]; sizes=res.get("sizes"); ry=(float(sizes[1]) if isinstance(sizes,list) and len(sizes)>1 else cy)
print("reservoir: x0=%.3f size_x=%.3f  size_y(ry)=%.3f  cell_y=%.3f"%(rx0,float(res["size_x"]),ry,cy))
# pointwise gpu eyy at Ey-face along y at x = reservoir center
xc = rx0 + float(res["size_x"])/2
i = int(round((xc+cx/2)/dx))
import meep as mp
from class_simulation import Simulation
sm=Simulation(p); sm._set_everything(); sm.simulation.init_sim()
# gpu pointwise eyy at Ey-face
o=np.ones((Nx,Ny)); z=np.zeros((Nx,Ny))
Xg=((np.arange(Nx))*dx-cx/2)[:,None]*np.ones((1,Ny)); Yg=((np.arange(Ny)+0.5)*dx-cy/2)[None,:]*np.ones((Nx,1))
e=sg._eps_sharp_at(Xg,Yg); eyy_pt=e[1]
print("\n y      MEEP_eyy  gpu_pt_eyy")
for j in range(Ny):
    y=(j+0.5)*dx-cy/2
    if abs(abs(y)-ry/2) < 0.15:  # near y=±ry/2 edge
        em=complex(sm.simulation.get_epsilon_point(mp.Vector3(xc,y,0),mp.Ey)).real
        print("%+.3f  %.4f    %.4f"%(y,em,eyy_pt[i,j]))
print("DONE")
