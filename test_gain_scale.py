"""Run config 3 at multiple coupling_scales to calibrate against MEEP."""
import os, sys, numpy as np
sys.path.insert(0, "/home/cernez/resevoir")
os.environ["LADDER_RUN_UNTIL"]="200"
from ladder.ladder import build_json
import importlib.util
def load_gpu():
    sys.modules.pop("class_simulation_gpu",None)
    s=importlib.util.spec_from_file_location("class_simulation_gpu","/home/cernez/resevoir/class_simulation_gpu.py")
    m=importlib.util.module_from_spec(s); sys.modules["class_simulation_gpu"]=m; s.loader.exec_module(m); return m
import jax; jax.config.update("jax_enable_x64", True)

# MEEP once (shared)
from class_simulation import Simulation
p0 = build_json(3)
sm = Simulation(p0); sm.run_simulation()
mm = np.load(p0+"/simulation/monitor_2.npz")["Ey"].flatten()
print("MEEP |Ey| max=%.5f" % np.abs(mm).max())

# Test two scales
import multilevel as ml; orig_norm = ml.MEEP_SIGMA_NORM
for scale in (1.0, 0.862, 0.75):
    ml.MEEP_SIGMA_NORM = scale
    p = build_json(3)
    csg = load_gpu(); sg = csg.SimulationGPU(folder_path=p, force_fullvector=True); sg.run()
    gg = np.load(p+"/simulation/monitor_2.npz")["Ey"].flatten()
    m=mm.ravel(); g=gg.ravel()
    ym=np.linspace(-3,3,len(m)); yg=np.linspace(-3,3,len(g)); gi=np.interp(ym,yg,g.real)+1j*np.interp(ym,yg,g.imag)
    tr=int(0.05*len(m)); a=m[tr:-tr]; b=gi[tr:-tr]; phi=np.angle(np.vdot(b,a))
    print("coupling_scale=%.3f  |Ey| max=%.5f  max-ratio=%.4f  phase=%.1fdeg  rel-L2(derot)=%.3f" % (
      scale, np.abs(gg).max(), np.abs(b).max()/np.abs(a).max(), np.degrees(phi),
      np.linalg.norm(b*np.exp(1j*phi)-a)/np.linalg.norm(a)))
ml.MEEP_SIGMA_NORM = orig_norm
print("DONE")
