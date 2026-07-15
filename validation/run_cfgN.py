import os, sys, numpy as np, shutil
sys.path.insert(0, "/home/cernez/resevoir")
N = int(sys.argv[1])
os.environ["LADDER_RUN_UNTIL"] = os.environ.get("LADDER_RUN_UNTIL", "200")
os.environ.pop("GPUMEEP_KOTTKE", None); os.environ.pop("MEEP_NO_SUBPIXEL", None)
from ladder.ladder import build_json
p = build_json(N)
from class_simulation import Simulation
sm = Simulation(p); sm.run_simulation()
mm = np.asarray(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()
shutil.copy(p + "/simulation/monitor_2.npz", p + "/simulation/monitor_2_meep.npz")
print("cfg%d MEEP |Ey| max=%.5f" % (N, np.abs(mm).max()))
import jax; jax.config.update("jax_enable_x64", True)
# load resevoir's class_simulation_gpu explicitly (avoid BlockOptimization shadow)
import importlib.util
sys.modules.pop("class_simulation_gpu", None)
_spec = importlib.util.spec_from_file_location("class_simulation_gpu", "/home/cernez/resevoir/class_simulation_gpu.py")
csg = importlib.util.module_from_spec(_spec); sys.modules["class_simulation_gpu"] = csg; _spec.loader.exec_module(csg)
print("loaded", csg.__file__)
sg = csg.SimulationGPU(folder_path=p, force_fullvector=True); sg.run()
gg = np.asarray(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()
shutil.copy(p + "/simulation/monitor_2.npz", p + "/simulation/monitor_2_gpumeep.npz")
print("cfg%d gpumeep |Ey| max=%.5f" % (N, np.abs(gg).max()))
m=mm.ravel(); g=gg.ravel()
ym=np.linspace(-3,3,len(m)); yg=np.linspace(-3,3,len(g)); gi=np.interp(ym,yg,g.real)+1j*np.interp(ym,yg,g.imag)
tr=int(0.05*len(m)); a=m[tr:-tr]; b=gi[tr:-tr]
cc=np.vdot(b,a)/(np.linalg.norm(b)*np.linalg.norm(a))
phi=np.angle(np.vdot(b,a))
rawL2=np.linalg.norm(b-a)/np.linalg.norm(a)
derotL2=np.linalg.norm(b*np.exp(1j*phi)-a)/np.linalg.norm(a)
print("cfg%d |corr|=%.4f phase=%.2fdeg max-ratio=%.4f rel-L2=%.3f rel-L2(derot)=%.3f" % (
    N, np.abs(cc), np.degrees(phi), np.abs(b).max()/np.abs(a).max(), rawL2, derotL2))
print("DONE%d" % N)
