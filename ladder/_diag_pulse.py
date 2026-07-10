"""Compare Gaussian pulse parameters + source position between gpu and MEEP,
and verify the monitor y-range alignment (MEEP has len=242, gpu len=240)."""
import os, sys, importlib
import numpy as np
_RES = 40
os.environ["LADDER_RES"] = str(_RES)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json  # noqa: E402

path = build_json(4)

# --- gpu pulse params ---
import jax; jax.config.update("jax_enable_x64", True)
gpu_src = os.environ.get("GPUMEEP_PATH")
sys.path.insert(0, gpu_src); sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
g = csg.SimulationGPU(folder_path=path); g.force_fullvector = True
g._set_data(); g._update_all_args()

# gpu uses JSON: lam=0.55, FWHM=20fs, res=40 → dx=1/40
lam = 0.5; f0 = 1.0/lam
_FS = 0.299792458
fwhm = 20.0; width = fwhm / _FS / 2.35482   # sigma = FWHM/2.355 in Meep units
print(f"gpu: f0={f0:.6f}  width(FWHM→sigma)={width:.8f}  start_time=0.0")
# MEEP uses peak_time = start_time + cutoff*width where cutoff=5.0 in ctor, then
# start_time=0 + 2*width*cutoff = 0 + 10*width in Python GaussianSource
# Python: gaussian_src_time(freq, width, start_time, start_time+2*width*cutoff)
_psrc = g.objects_args[5]    # source_1
gpeak = float(_psrc.get("pulse_delay_fs", 0.0)) / _FS + 5.0 * width   # gpu: start+cutoff*width
print(f"gpu peak_time={gpeak:.8f}")
# MEEP: peak_time = _psrc.start_time + cutoff*width → start=0, cutoff=5.0 → peak=5*width
# but Python GaussianSource passes end_time=start_time+2*width*cutoff, the ctor uses
# (start_time+end_time)/2 = start_time + width*cutoff
import meep as mp  # noqa: E402
from class_simulation import Simulation  # noqa: E402
sm = Simulation(path); sm._set_everything(); sm.simulation.init_sim()
# extract gaussian_src_time from MEEP source
for sv in sm.simulation.sources:
    if sv.c == mp.Ey:
        t = sv.t()
        print(f"MEEP: freq={t.freq:.6f}  width={t.width:.12f}  "
              f"peak_time={t.peak_time:.12f}  cutoff={t.cutoff:.12f}")
        print(f"MEEP: peak_time/width = {t.peak_time/t.width:.6f}")
        break
print(f"gpu:  peak_time/width = {gpeak/width:.6f}")
print(f"MEEP: is_integrated for Ey src = {t.is_integrated}")

# --- Monitor y-range ---
i_mon = csg._meep_to_grid_x(g.objects_args[6]["center_x_meep"], g.cx, g.dx)
j_lo, j_hi = csg._meep_to_grid_y_range(0.0, g.objects_args[6]["size_y_meep"], g.cy, g.dx)
print(f"gpu  monitor: x_idx={i_mon} y_range=[{j_lo},{j_hi}] ny={j_hi-j_lo}")
# MEEP monitor metadata
sen = sm.sensors[0]
xs, ys, zs, ws = sm.simulation.get_array_metadata(center=sen.center, size=sen.size)
print(f"MEEP monitor metadata: ny={len(ys)}  y[0]={ys[0]:.6f} y[-1]={ys[-1]:.6f} dy={ys[1]-ys[0]:.6f}")
print(f"MEEP sensor center=({float(sen.center.x):.6f}, {float(sen.center.y):.6f})  "
      f"size=({float(sen.size.x):.6f}, {float(sen.size.y):.6f})")
# gpu y-centre of monitor: j_lo to j_hi, y at j = (j+0.5)*dx - cy for Ey
print(f"gpu  monitor: y[0]={(j_lo+0.5)*g.dx-g.cy:.6f}  y[-1]={(j_hi-0.5)*g.dx-g.cy:.6f}")

print("DONE_PULSE")
