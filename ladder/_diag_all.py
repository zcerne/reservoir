"""Single-script fan-out: dt, step count, monitor y-range for config 4 res40."""
import os, sys, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["LADDER_RES"] = "40"
from ladder import build_json  # noqa: E402
import meep as mp  # noqa: E402
from class_simulation import Simulation  # noqa: E402

path = build_json(4)

# === 1. DT ===
s = Simulation(path); s._set_everything(); s.simulation.init_sim()
mdt = s.simulation.fields.dt
res = 40; gdt = 1.0 / (2 * res)
print(f"1_DT: MEEP dt={mdt:.15f} gpu dt={gdt:.15f} diff={mdt-gdt:.3e} ratio={mdt/gdt:.6f}")

# === 2. STEPS (run_until=300) ===
ru = 300
msteps = int(round((ru + 50) / mdt))
print(f"2_STEPS: MEEP steps={msteps} total_t={msteps*mdt:.6f}")
import jax; jax.config.update("jax_enable_x64", True)
cpu = os.environ.get("GPUMEEP_PATH", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/GPUmeep/src")
sys.path.insert(0, cpu); sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
g = csg.SimulationGPU(folder_path=path); g.force_fullvector = True
g._set_data(); g._update_all_args()
gdt2 = g.dt
decay = float(g.args.get("source_off_decay", 50.0))
gsteps = int((ru + decay) / gdt2)
print(f"2_STEPS: gpu  steps={gsteps} total_t={gsteps*gdt2:.6f} diff_total_t={msteps*mdt-gsteps*gdt2:.6f}")

# === 3. MONITOR Y-RANGE ===
sen = s.sensors[0]
xs, ys, zs, ws = s.simulation.get_array_metadata(center=sen.center, size=sen.size)
print(f"3_MONY: MEEP ny={len(ys)} y[0]={ys[0]:.6f} y[-1]={ys[-1]:.6f} dy={ys[1]-ys[0]:.6f}")
xmeep = g.objects_args[6]["center_x_meep"]; sy = g.objects_args[6]["size_y_meep"]
imon = csg._meep_to_grid_x(xmeep, g.cx, g.dx)
jlo, jhi = csg._meep_to_grid_y_range(0.0, sy, g.cy, g.dx)
print(f"3_MONY: gpu  ny={jhi-jlo} y[0]={(jlo+0.5)*g.dx-g.cy:.6f} y[-1]={(jhi-0.5)*g.dx-g.cy:.6f} dy={g.dx:.6f}")

# === 4. SOURCE POSITION ===
xsrc = g.objects_args[5]["center_x_meep"]
isrc = csg._meep_to_grid_x(xsrc, g.cx, g.dx)
print(f"4_SRC: gpu x_meep={xsrc:.6f} i_src={isrc} x_abs={isrc*g.dx-g.cx:.6f}")
msrc_x = float(s.simulation.sources[0].center.x)
print(f"4_SRC: MEEP x_center={msrc_x:.6f} diff={isrc*g.dx-g.cx-msrc_x:.6f}")

print("ALL_DIAG_DONE")
