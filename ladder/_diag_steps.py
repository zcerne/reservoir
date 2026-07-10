"""Are the total timestep counts identical? MEEP runs to run_until+50,
gpu to run_until+decay. Print step counts + total integration time."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meep as mp
from class_simulation import Simulation
path = __import__('ladder').build_json(4)
s = Simulation(path); s._set_everything(); s.simulation.init_sim()
mdt = s.simulation.fields.dt
ru = 120
# MEEP runs run_until, then change_sources, then until=50.
# MEEP step count = (run_until + 50) / dt, but dt is from Courant.
m_steps = int(round((ru + 50) / mdt))
m_time = m_steps * mdt
print(f"MEEP: dt={mdt:.12f}  run_until+50={ru+50}  steps≈{m_steps}  total_time={m_time:.6f}")
import jax, importlib
jax.config.update("jax_enable_x64", True)
gpu_src = os.environ.get("GPUMEEP_PATH")
sys.path.insert(0, gpu_src); sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
g = csg.SimulationGPU(folder_path=path); g.force_fullvector = True
g._set_data(); g._update_all_args()
gdt = g.dt
decay = float(g.args.get("source_off_decay", 50.0))
g_steps = int((ru + decay) / gdt)
g_time = g_steps * gdt
print(f"gpu:  dt={gdt:.12f}  run_until+decay={ru+decay}  steps={g_steps}  total_time={g_time:.6f}")
print(f"delta total_time = {m_time-g_time:.6f}  delta steps = {m_steps-g_steps}")
print("DONE_STEPS")
