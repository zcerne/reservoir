"""Pinpoint the config-4 registration offset: compare where gpu vs MEEP place the
SOURCE and MONITOR in absolute x, for config 4 (non-integer cell) and config 1
(integer cell). A sub-pixel gpu-vs-MEEP offset that appears only for config 4 is
the source of the -4.3deg phase."""
import os, sys, importlib
import numpy as np
RESV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RESV); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json  # noqa: E402

import meep as mp  # noqa: E402
from class_simulation import Simulation  # noqa: E402


def probe(cfg):
    path = build_json(cfg)
    # --- gpu ---
    import jax; jax.config.update("jax_enable_x64", True)
    sys.modules.pop("class_simulation_gpu", None)
    csg = importlib.import_module("class_simulation_gpu")
    g = csg.SimulationGPU(folder_path=path); g.force_fullvector = True
    g._set_data(); g._update_all_args()
    src = next(o for o in g.objects_args if o.get("class") == "source")
    mon = next(o for o in g.objects_args if o.get("class") == "monitor")
    i_src = csg._meep_to_grid_x(src["center_x_meep"], g.gx, g.dx)
    i_mon = csg._meep_to_grid_x(mon["center_x_meep"], g.gx, g.dx)
    gpu_src_x = i_src * g.dx - g.cx
    gpu_mon_x = i_mon * g.dx - g.cx
    print(f"--- config {cfg} ---")
    print(f"gpu: cell_x={g.cell_x:.6f} gx={g.gx:.6f} (cell·res={g.cell_x*g.resolution:.3f}, Nx={g.Nx})")
    print(f"gpu: src x_meep={src['center_x_meep']:.6f} -> i={i_src} abs_x={gpu_src_x:.6f}")
    print(f"gpu: mon x_meep={mon['center_x_meep']:.6f} -> i={i_mon} abs_x={gpu_mon_x:.6f}")
    # --- MEEP ---
    sm = Simulation(path); sm._set_everything(); sm.simulation.init_sim()
    msrc_x = float(sm.simulation.sources[0].center.x)
    # monitor metadata x (dft region)
    sen = sm.sensors[0]
    mmon_x = float(sen.center.x) if hasattr(sen, "center") else float("nan")
    print(f"MEEP: src center.x={msrc_x:.6f}   mon center.x={mmon_x:.6f}")
    print(f"DELTA src (gpu-meep continuous) = {src['center_x_meep']-msrc_x:+.6f}")
    print(f"DELTA mon abs_x(gpu) - meep center.x = {gpu_mon_x-mmon_x:+.6f}  "
          f"(phase @k=2pi/0.5: {np.degrees((gpu_mon_x-mmon_x)*2*np.pi/0.5):+.3f} deg)")


for cfg in [1, 4]:
    probe(cfg)
print("DONE_POS")
