"""Brutal comparison: config 1, MeepPML with siginv=1 everywhere vs CPML.
Both should be identical (standard curl, no PML effect). If they differ,
the MeepPML dispatch/step function has a bug. If they match, the siginv
profile formula is wrong."""
import os,sys,importlib,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ladder import build_json

gpu_src=os.environ.get("GPUMEEP_PATH")
sys.path.insert(0,gpu_src)
import jax;jax.config.update("jax_enable_x64",True)

path=build_json(1)

# --- CPML run ---
sys.modules.pop("class_simulation_gpu",None)
csg=importlib.import_module("class_simulation_gpu")
g_cpml=csg.SimulationGPU(folder_path=path);g_cpml.force_fullvector=True
g_cpml._set_data();g_cpml._update_all_args()
g_cpml._build_material()
# Force CPML
import fdtd_2d as f2
n=int(round(float(g_cpml.args.get("pml_size",2.0))/g_cpml.dx))
g_cpml.pml = f2.make_cpml_full_2d(g_cpml.grid,g_cpml.dt,n_pml=(n,n))
g_cpml._build_sources_sted()
D_cpml=f2.zero_D_full(g_cpml.grid)
f_cpml=f2.zero_fields_full(g_cpml.grid)
p_cpml=g_cpml.pml

# --- MeepPML with siginv=1 ---
import pml_meep
all_ones_x=np.ones(g_cpml.Nx);all_ones_y=np.ones(g_cpml.Ny)
fake_pml=pml_meep.MeepPML(
    siginv_x=jax.numpy.asarray(all_ones_x),
    siginv_y=jax.numpy.asarray(all_ones_y))
D_meep=f2.zero_D_full(g_cpml.grid)
f_meep=f2.zero_fields_full(g_cpml.grid)
p_meep=fake_pml

# Step once (source injection the same for both)
t=0.0
src=g_cpml.sources[0]
D_cpml=src.apply_D(D_cpml,t)
D_meep=src.apply_D(D_meep,t)

# One step with each PML
D_cpml,f_cpml,p_cpml=f2.step_2d_full_dform(D_cpml,f_cpml,g_cpml.grid,g_cpml.dt,p_cpml,g_cpml.material)
D_meep,f_meep,p_meep=f2.step_2d_full_dform(D_meep,f_meep,g_cpml.grid,g_cpml.dt,p_meep,g_cpml.material)

Ey_cpml=np.array(f_cpml.Ey);Ey_meep=np.array(f_meep.Ey)
diff=Ey_cpml-Ey_meep
print(f"Step 1: CPML max|Ey|={np.abs(Ey_cpml).max():.6g}  MeepPML max|Ey|={np.abs(Ey_meep).max():.6g}")
print(f"  max|diff|={np.abs(diff).max():.6g}  rms={np.sqrt(np.mean(diff**2)):.6g}")
if np.abs(diff).max()<1e-12:
    print("  IDENTICAL — curl computation matches")
else:
    idx=np.unravel_index(np.argmax(np.abs(diff)),Ey_cpml.shape)
    print(f"  DIFFER at max-diff pos ({idx[0]},{idx[1]}): CPML={Ey_cpml[idx]:.6g} Meep={Ey_meep[idx]:.6g}")

# Step 2
t=g_cpml.dt
D_cpml=src.apply_D(D_cpml,t)
D_meep=src.apply_D(D_meep,t)
D_cpml,f_cpml,p_cpml=f2.step_2d_full_dform(D_cpml,f_cpml,g_cpml.grid,g_cpml.dt,p_cpml,g_cpml.material)
D_meep,f_meep,p_meep=f2.step_2d_full_dform(D_meep,f_meep,g_cpml.grid,g_cpml.dt,p_meep,g_cpml.material)
Ey_cpml2=np.array(f_cpml.Ey);Ey_meep2=np.array(f_meep.Ey)
diff2=Ey_cpml2-Ey_meep2
print(f"Step 2: CPML max|Ey|={np.abs(Ey_cpml2).max():.6g}  MeepPML max|Ey|={np.abs(Ey_meep2).max():.6g}")
print(f"  max|diff|={np.abs(diff2).max():.6g}  rms={np.sqrt(np.mean(diff2**2)):.6g}")
print("DONE_CMP")
