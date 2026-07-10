"""Is dt identical? MEEP picks dt from Courant (1/sqrt(1/dx²+1/dy²)) while
gpu hardcodes dt=dx/2 (Courant=0.5 in vacuum). Compare."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meep as mp
from class_simulation import Simulation
path = __import__('ladder').build_json(4)
s = Simulation(path); s._set_everything(); s.simulation.init_sim()
mdt = s.simulation.fields.dt
print(f"MEEP dt={mdt:.15f}")
# gpu: dt = dx/2 = 1/(2*res)
res = int(os.environ.get("LADDER_RES", "40"))
gdt = 1.0 / (2.0 * res)
print(f"gpu  dt={gdt:.15f}")
print(f"delta dt = {mdt-gdt:.3e}  ratio = {mdt/gdt:.6f}")
print("DONE_DT")
