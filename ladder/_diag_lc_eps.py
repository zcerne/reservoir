"""Compare LC epsilon (config 2) between MEEP and gpu at Yee faces."""
import os,sys,importlib,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ladder import build_json
import meep as mp
from meep.simulation import py_v3_to_vec
from class_simulation import Simulation
p=build_json(2)
sm=Simulation(p);sm._set_everything();sm.simulation.init_sim()
f=sm.simulation.fields;DIM=sm.simulation.dimensions

def vec(x,y):return py_v3_to_vec(DIM,mp.Vector3(x,y),False)

# Sample at 5 x-positions across the LC reservoir
import jax;jax.config.update("jax_enable_x64",True)
cpu=os.environ.get("GPUMEEP_PATH","")
sys.path.insert(0,cpu)
sys.modules.pop("class_simulation_gpu",None)
csg=importlib.import_module("class_simulation_gpu")
g=csg.SimulationGPU(folder_path=p);g.force_fullvector=True
g._set_data();g._update_all_args();g._build_material()
eyy_g=np.array(1.0/g.material.iyy_Ey);exx_g=np.array(1.0/g.material.ixx_Ex)
jmid=g.Ny//2

for xi,x_meep in enumerate([-4.0,-2.0,0.0,2.0,4.0]):
    i=csg._meep_to_grid_x(x_meep,g.cx,g.dx)
    # MEEP chi1inv at Ey face (i, j+0.5)
    yc=(jmid+0.5)*g.dx-g.cy;xEy=i*g.dx-g.cx
    iy_m=f.get_chi1inv(mp.Ey,mp.Y,vec(xEy,yc),0.0,True).real
    ey_m=1.0/iy_m if abs(iy_m)>1e-12 else 0
    ey_g=eyy_g[i,jmid]
    # Ex face
    yn=jmid*g.dx-g.cy;xEx=(i+0.5)*g.dx-g.cx
    ix_m=f.get_chi1inv(mp.Ex,mp.X,vec(xEx,yn),0.0,True).real
    ex_m=1.0/ix_m if abs(ix_m)>1e-12 else 0
    ex_g=exx_g[i,jmid]
    print("x=%+5.1f i=%d: Ey eps meep=%.4f gpu=%.4f d=%.4f | Ex eps meep=%.4f gpu=%.4f d=%.4f"
          % (x_meep,i,ey_m,ey_g,ey_m-ey_g,ex_m,ex_g,ex_m-ex_g))
print("DONE_LC")
