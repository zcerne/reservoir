"""gpu uses MEEP's get_epsilon as its ε (config 4). Isolates FDTD solver from ε-build."""
import os, sys, numpy as np, shutil
sys.path.insert(0, "/home/cernez/resevoir")
os.environ["LADDER_RUN_UNTIL"]="200"
os.environ.pop("GPUMEEP_SHARP",None); os.environ.pop("MEEP_NO_SUBPIXEL",None)  # MEEP subpixel ON
from ladder.ladder import build_json
p = build_json(4)
import meep as mp
from class_simulation import Simulation
sm = Simulation(p); sm.run_simulation()
mm = np.load(p+"/simulation/monitor_2.npz")["Ey"].flatten()
print("MEEP |Ey| max=%.5f" % np.abs(mm).max())
# MEEP structure for get_epsilon
sm2 = Simulation(p); sm2._set_everything(); sm2.simulation.init_sim()
import jax; jax.config.update("jax_enable_x64", True)
import importlib.util
sys.modules.pop("class_simulation_gpu",None)
s=importlib.util.spec_from_file_location("class_simulation_gpu","/home/cernez/resevoir/class_simulation_gpu.py")
csg=importlib.util.module_from_spec(s); sys.modules["class_simulation_gpu"]=csg; s.loader.exec_module(csg)
sg = csg.SimulationGPU(folder_path=p, force_fullvector=True)
sg._set_data(); sg._update_all_args()
Nx,Ny,dx,cx,cy = sg.Nx,sg.Ny,sg.dx,sg.cell_x,sg.cell_y
import jax.numpy as jnp
# sample MEEP diagonal eps at gpu Yee points (config 4 isotropic → offdiag 0)
def geteps(comp, xoff, yoff):
    arr=np.ones((Nx,Ny))
    for i in range(Nx):
        xx=i*dx-cx/2+xoff
        for j in range(Ny):
            yy=j*dx-cy/2+yoff
            arr[i,j]=complex(sm2.simulation.get_epsilon_point(mp.Vector3(xx,yy,0),comp)).real
    return arr
exx=geteps(mp.Ex,0.5*dx,0.0); eyy=geteps(mp.Ey,0.0,0.5*dx); ezz=geteps(mp.Ez,0.0,0.0)
z=jnp.zeros((Nx,Ny))
sg.material=csg.f2.AnisoFull2D(ixx_Ex=jnp.asarray(1/exx),ixy_Ex=z,ixz_Ex=z,
    ixy_Ey=z,iyy_Ey=jnp.asarray(1/eyy),iyz_Ey=z,ixz_nd=z,iyz_nd=z,izz_nd=jnp.asarray(1/ezz))
sg.eps_inv_zz=jnp.asarray(1/ezz); sg._n_max=float(np.sqrt(max(exx.max(),eyy.max(),ezz.max())))
sg._build_material=lambda: None   # keep injected material
sg.run()
gg=np.load(p+"/simulation/monitor_2.npz")["Ey"].flatten()
print("gpu(MEEPeps) |Ey| max=%.5f"%np.abs(gg).max())
m=mm.ravel();g=gg.ravel()
ym=np.linspace(-3,3,len(m));yg=np.linspace(-3,3,len(g));gi=np.interp(ym,yg,g.real)+1j*np.interp(ym,yg,g.imag)
tr=int(0.05*len(m));a=m[tr:-tr];b=gi[tr:-tr];phi=np.angle(np.vdot(b,a))
print("cfg4-MEEPeps |corr|=%.4f phase=%.2fdeg max-ratio=%.4f rel-L2(derot)=%.3f"%(
  np.abs(np.vdot(b,a))/(np.linalg.norm(b)*np.linalg.norm(a)),np.degrees(phi),
  np.abs(b).max()/np.abs(a).max(),np.linalg.norm(b*np.exp(1j*phi)-a)/np.linalg.norm(a)))
print("DONE")
