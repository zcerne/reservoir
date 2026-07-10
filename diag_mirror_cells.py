"""Cell-by-cell: gpu analytic-Kottke mirror εyy vs MEEP get_epsilon εyy across DBR (config 4)."""
import os, sys, numpy as np
sys.path.insert(0, "/home/cernez/resevoir")
os.environ.pop("GPUMEEP_SHARP",None); os.environ.pop("MEEP_NO_SUBPIXEL",None)  # MEEP subpixel ON
from ladder.ladder import build_json
p = build_json(4)
import importlib.util
def load_gpu():
    sys.modules.pop("class_simulation_gpu",None)
    s=importlib.util.spec_from_file_location("class_simulation_gpu","/home/cernez/resevoir/class_simulation_gpu.py")
    m=importlib.util.module_from_spec(s); sys.modules["class_simulation_gpu"]=m; s.loader.exec_module(m); return m
csg = load_gpu()
sg = csg.SimulationGPU(folder_path=p, force_fullvector=True)
sg._set_data(); sg._update_all_args()
Nx,Ny,dx,cx,cy = sg.Nx,sg.Ny,sg.dx,sg.cell_x,sg.cell_y
# gpu forward eps at Ey-face via analytic mirror overlay on vacuum
o=np.ones((Nx,Ny)); z=np.zeros((Nx,Ny))
base=(o.copy(),o.copy(),o.copy(),z.copy(),z.copy(),z.copy())
exx,eyy,ezz,exy,exz,eyz = sg._overlay_iso_full(base, 0.0, 0.5)   # Ey-face
jm=Ny//2
xs = np.arange(Nx)*dx - cx/2
# MEEP
import meep as mp
from class_simulation import Simulation
sm = Simulation(p); sm._set_everything(); sm.simulation.init_sim()
mir = next(o for o in sg.objects_args if o.get("class")=="mirror")
lay = sg._mirror_layers(mir); mx0=min(l[0] for l in lay); mx1=max(l[1] for l in lay)
print("mirror-1 layers (x_lo,x_hi,n):")
for L in lay[:6]: print("   [%.4f, %.4f] n=%.3f (%.3f cells)"%(L[0],L[1],L[2],(L[1]-L[0])/dx))
print("\ncell   x       gpu_eyy   MEEP_eyy   diff")
for i in range(Nx):
    x=xs[i]
    if mx0-0.05 <= x <= mx0+0.7:  # first ~1um of DBR
        em=complex(sm.simulation.get_epsilon_point(mp.Vector3(x,(jm+0.5)*dx-cy/2,0),mp.Ey)).real
        print("%4d  %.4f   %.4f    %.4f    %+.4f"%(i,x,eyy[i,jm],em,eyy[i,jm]-em))
print("DONE")
