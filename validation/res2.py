import os, sys, numpy as np, json
sys.path.insert(0, "/home/cernez/resevoir")
os.environ["LADDER_RUN_UNTIL"]="200"; os.environ.pop("GPUMEEP_KOTTKE",None)
os.environ.pop("GPUMEEP_SHARP",None); os.environ.pop("MEEP_NO_SUBPIXEL",None)
from ladder.ladder import build_json
import importlib.util
def load_gpu():
    sys.modules.pop("class_simulation_gpu",None)
    s=importlib.util.spec_from_file_location("class_simulation_gpu","/home/cernez/resevoir/class_simulation_gpu.py")
    m=importlib.util.module_from_spec(s); sys.modules["class_simulation_gpu"]=m; s.loader.exec_module(m); return m
import jax; jax.config.update("jax_enable_x64", True)
from class_simulation import Simulation
for RES in (40, 60, 80):
    p = build_json(2)
    jf = p + "/simulation_data.json"; cfg=json.load(open(jf)); cfg["resolution"]=RES; json.dump(cfg,open(jf,"w"))
    sm = Simulation(p); sm.run_simulation()
    mm = np.load(p+"/simulation/monitor_2.npz")["Ey"].flatten()
    csg = load_gpu(); sg = csg.SimulationGPU(folder_path=p, force_fullvector=True); sg.run()
    gg = np.load(p+"/simulation/monitor_2.npz")["Ey"].flatten()
    m=mm.ravel(); g=gg.ravel()
    ym=np.linspace(-3,3,len(m)); yg=np.linspace(-3,3,len(g)); gi=np.interp(ym,yg,g.real)+1j*np.interp(ym,yg,g.imag)
    tr=int(0.05*len(m)); a=m[tr:-tr]; b=gi[tr:-tr]
    phi=np.angle(np.vdot(b,a))
    print("RES=%d phase=%.2fdeg max-ratio=%.4f rel-L2=%.3f rel-L2(derot)=%.3f" % (
      RES, np.degrees(phi), np.abs(b).max()/np.abs(a).max(),
      np.linalg.norm(b-a)/np.linalg.norm(a), np.linalg.norm(b*np.exp(1j*phi)-a)/np.linalg.norm(a)))
print("DONE")
