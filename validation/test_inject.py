"""Test: overwrite gpu mirror-region eps with MEEP's get_epsilon, rerun config 4."""
import os, sys, numpy as np, shutil
sys.path.insert(0, "/home/cernez/resevoir")
os.environ["LADDER_RUN_UNTIL"]="200"; os.environ.pop("GPUMEEP_KOTTKE",None); os.environ.pop("MEEP_NO_SUBPIXEL",None)
from ladder.ladder import build_json
p = build_json(4)
import meep as mp
from class_simulation import Simulation
sm = Simulation(p); sm.run_simulation()
mm = np.load(p+"/simulation/monitor_2.npz")["Ey"].flatten()
print("MEEP |Ey| max=%.5f" % np.abs(mm).max())
# build MEEP eps grid on gpu Yee points for mirror rows
sm2 = Simulation(p); sm2._set_everything(); sm2.simulation.init_sim()
import jax; jax.config.update("jax_enable_x64", True)
import importlib.util
sys.modules.pop("class_simulation_gpu",None)
s=importlib.util.spec_from_file_location("class_simulation_gpu","/home/cernez/resevoir/class_simulation_gpu.py")
csg=importlib.util.module_from_spec(s); sys.modules["class_simulation_gpu"]=csg; s.loader.exec_module(csg)
sg = csg.SimulationGPU(folder_path=p, force_fullvector=True)
sg._set_data(); sg._update_all_args(); sg._build_material()
Nx,Ny,dx,cx,cy = sg.Nx,sg.Ny,sg.dx,sg.cell_x,sg.cell_y
xg = np.arange(Nx)*dx
# mirror x-mask (union of all mirror layer spans, in MEEP coords)
mirs=[o for o in sg.objects_args if o.get("class")=="mirror"]
xmeep = xg - cx/2
mask_x = np.zeros(Nx, bool)
for mo in mirs:
    for (a,b,n) in sg._mirror_layers(mo): mask_x |= (xmeep>=a-dx)&(xmeep<=b+dx)
cols = np.where(mask_x)[0]
print("mirror cols:", len(cols))
# sample MEEP eps at Ex, Ey, node points for those columns, all rows
yE = (np.arange(Ny))*dx - cy/2
def geteps(comp, xoff, yoff):
    out=np.array(sg.material.iyy_Ey)*0.0 + 0.0
    arr = np.ones((Nx,Ny))
    for i in cols:
        xx = i*dx - cx/2 + xoff
        for j in range(Ny):
            yy = j*dx - cy/2 + yoff
            arr[i,j] = complex(sm2.simulation.get_epsilon_point(mp.Vector3(xx,yy,0), comp)).real
    return arr
eyy = geteps(mp.Ey, 0.0, 0.5*dx)   # Ey at (i, j+1/2)
exx = geteps(mp.Ex, 0.5*dx, 0.0)   # Ex at (i+1/2, j)
ezz = geteps(mp.Ez, 0.0, 0.0)      # Ez at node (i,j)
import jax.numpy as jnp
iyy=np.array(sg.material.iyy_Ey); ixx=np.array(sg.material.ixx_Ex); izz=np.array(sg.eps_inv_zz)
for i in cols:
    iyy[i,:]=1.0/eyy[i,:]; ixx[i,:]=1.0/exx[i,:]; izz[i,:]=1.0/ezz[i,:]
sg.material = sg.material._replace(iyy_Ey=jnp.asarray(iyy), ixx_Ex=jnp.asarray(ixx))
sg.eps_inv_zz = jnp.asarray(izz)
# monkeypatch _build_material to no-op so run() keeps our injected material
sg._build_material = lambda : None
sg.run()
gg = np.load(p+"/simulation/monitor_2.npz")["Ey"].flatten()
print("gpu(inj) |Ey| max=%.5f" % np.abs(gg).max())
m=mm.ravel(); g=gg.ravel()
ym=np.linspace(-3,3,len(m)); yg2=np.linspace(-3,3,len(g)); gi=np.interp(ym,yg2,g.real)+1j*np.interp(ym,yg2,g.imag)
tr=int(0.05*len(m)); a=m[tr:-tr]; b=gi[tr:-tr]
print("cfg4-INJECT complex-corr=%.4f max-ratio=%.4f rel-L2=%.3f" % (
  np.abs(np.vdot(b,a))/(np.linalg.norm(b)*np.linalg.norm(a)), np.abs(b).max()/np.abs(a).max(), np.linalg.norm(b-a)/np.linalg.norm(a)))
print("DONE")
