"""First 5 steps of config 1 with both CPML and MeepPML, dump max|Ey| each step."""
import os,sys,importlib
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ladder import build_json

gpu_src=os.environ.get("GPUMEEP_PATH")
sys.path.insert(0,gpu_src)
import jax;jax.config.update("jax_enable_x64",True)

path=build_json(1)  # config 1 vacuum

import pml_meep
sys.modules.pop("class_simulation_gpu",None)
csg=importlib.import_module("class_simulation_gpu")
g=csg.SimulationGPU(folder_path=path);g.force_fullvector=True
g._set_data();g._update_all_args()
g._build_material()
g._build_pml_full()
g._build_sources_sted()

# MeepPML test
import fdtd_2d as f2
D=f2.zero_D_full(g.grid)
fields=f2.zero_fields_full(g.grid)
p=g.pml

print(f"pml type={type(p).__name__}")
for step in range(5):
    t=step*g.dt
    src=g.sources[0]
    D=src.apply_D(D,t)
    # Apply PML step
    D,fields,p = f2.step_2d_full_dform(D,fields,g.grid,g.dt,p,g.material)
    ey=np.asarray(fields.Ey)
    print(f"step {step}: max|Ey|={np.abs(ey).max():.6g}")
print("DONE")
