import os, sys, numpy as np
sys.path.insert(0, "/home/cernez/resevoir")
os.environ["LADDER_RUN_UNTIL"] = "100"
os.environ["GPUMEEP_PATH"] = "/home/cernez/GPUmeep/src"
sys.path.insert(0, "/home/cernez/GPUmeep/src")
from ladder.ladder import build_json
import importlib.util, json, jax
def load_gpu():
    sys.modules.pop("class_simulation_gpu",None)
    s=importlib.util.spec_from_file_location("class_simulation_gpu","/home/cernez/resevoir/class_simulation_gpu.py")
    m=importlib.util.module_from_spec(s); sys.modules["class_simulation_gpu"]=m; s.loader.exec_module(m); return m
jax.config.update("jax_enable_x64", True)

p = build_json(3)
jf = p + "/simulation_data.json"; cfg = json.load(open(jf)); cfg["run_until"] = 100; json.dump(cfg, open(jf,"w"))
from class_simulation import Simulation
sm = Simulation(p); sm.run_simulation()
mm = np.load(p+"/simulation/monitor_2.npz")["Ey"].flatten()

csg = load_gpu(); sg = csg.SimulationGPU(folder_path=p, force_fullvector=True); sg.run()
gg = np.load(p+"/simulation/monitor_2.npz")["Ey"].flatten()
m=mm.ravel(); g=gg.ravel()
ym=np.linspace(-3,3,len(m)); yg=np.linspace(-3,3,len(g)); gi=np.interp(ym,yg,g.real)+1j*np.interp(ym,yg,g.imag)
tr=int(0.05*len(m)); a=m[tr:-tr]; b=gi[tr:-tr]
print("t=100: MEEP max=%.5f gpu max=%.5f ratio=%.4f" % (np.abs(mm).max(), np.abs(gg).max(), np.abs(b).max()/np.abs(a).max()))
print("DONE")
