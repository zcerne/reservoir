import os, sys, numpy as np
sys.path.insert(0, "/home/cernez/resevoir")
os.environ["LADDER_RUN_UNTIL"]="200"; os.environ.pop("GPUMEEP_KOTTKE",None)
os.environ.pop("GPUMEEP_SHARP",None); os.environ.pop("MEEP_NO_SUBPIXEL",None)  # subpixel ON both
from ladder.ladder import build_json
p = build_json(4)
from class_simulation import Simulation
sm = Simulation(p); sm._set_everything(); sm.simulation.init_sim()
print("MEEP dt =", sm.simulation.fields.dt, " Courant=", sm.simulation.Courant, " res=", sm.resolution)
sm.plot_setup(); sm._run_meep_once(); sm._save_all()
mm = np.load(p+"/simulation/monitor_2.npz")["Ey"].flatten()
print("MEEP |Ey| max=%.5f" % np.abs(mm).max())
import jax; jax.config.update("jax_enable_x64", True)
import importlib.util
sys.modules.pop("class_simulation_gpu",None)
s=importlib.util.spec_from_file_location("class_simulation_gpu","/home/cernez/resevoir/class_simulation_gpu.py")
csg=importlib.util.module_from_spec(s); sys.modules["class_simulation_gpu"]=csg; s.loader.exec_module(csg)
sg = csg.SimulationGPU(folder_path=p, force_fullvector=True); sg.run()
gg = np.load(p+"/simulation/monitor_2.npz")["Ey"].flatten()
print("gpu |Ey| max=%.5f" % np.abs(gg).max())
m=mm.ravel(); g=gg.ravel()
ym=np.linspace(-3,3,len(m)); yg=np.linspace(-3,3,len(g)); gi=np.interp(ym,yg,g.real)+1j*np.interp(ym,yg,g.imag)
tr=int(0.05*len(m)); a=m[tr:-tr]; b=gi[tr:-tr]
print("cfg4-DTMATCH complex-corr=%.4f max-ratio=%.4f rel-L2=%.3f" % (
  np.abs(np.vdot(b,a))/(np.linalg.norm(b)*np.linalg.norm(a)), np.abs(b).max()/np.abs(a).max(), np.linalg.norm(b-a)/np.linalg.norm(a)))
print("DONE")
