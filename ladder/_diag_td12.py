"""Time-domain point-probe comparison for ladder config 1.2 (periodic, no PML).

Records Ey(t) at the exact Yee sample (x=0, y=0.0125) in BOTH engines for the
first N steps and compares sample-by-sample. A constant ratio from the first
nonzero sample = source-scale mismatch; a shift = timing; growth = dynamics.
"""
import os, sys
import numpy as np

RESEVOIR = os.environ.get("RESEVOIR", "/home/cernez/resevoir")
sys.path.insert(0, RESEVOIR)
sys.path.insert(0, os.path.join(RESEVOIR, "ladder"))
N_STEPS = int(os.environ.get("DIAG_STEPS", "1600"))

import ladder  # noqa: E402

path = ladder.build_json(1.2)
print("config dir:", path)

# ---------------- MEEP ----------------
import meep as mp  # noqa: E402
from class_simulation import Simulation  # noqa: E402

sim = Simulation(path)
sim._set_everything()          # build cell/geometry/sources/pmls like run_simulation
s = sim.simulation
s.init_sim()
dt = s.fields.dt
print("MEEP dt =", dt)

probe_pt = mp.Vector3(0.0, 0.0125, 0.0)   # exact Ey Yee sample (i=210, j=120)
trace_m = []


def _rec(sim_):
    trace_m.append(sim_.get_field_point(mp.Ey, probe_pt))


s.run(_rec, until=N_STEPS * dt - 1e-9)
trace_m = np.array([complex(v) for v in trace_m])
print("MEEP trace:", len(trace_m), "max |Ey| =", np.abs(trace_m).max())

# ---------------- GPUmeep ----------------
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
gpu_src = os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src")
sys.path.insert(0, gpu_src)
import importlib  # noqa: E402
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
import fdtd_2d as f2  # noqa: E402
import jax.numpy as jnp  # noqa: E402

g = csg.SimulationGPU(folder_path=path)
g.force_fullvector = True
# replicate run()/_run_2d_sted setup without the full run
g._set_data()
g._update_all_args()
g._build_material()
g.dt = float(g.args.get("courant", 0.5)) * g.dx   # MEEP-matched dt
g._build_pml_full()
g._build_sources_sted()

i_pr = int(round((0.0 + g.cx) / g.dx))         # x = 0
j_pr = int(round((0.0125 + g.cy) / g.dx - 0.5))  # Ey at (j+1/2) → y = 0.0125
print(f"GPU probe index: ({i_pr}, {j_pr}), dt = {g.dt}")

grid, dtg, material, sources, pml = g.grid, g.dt, g.material, g.sources, g.pml
fields = f2.zero_fields_full(grid)
D = f2.zero_D_full(grid)


@jax.jit
def step(D, f, p, i):
    t = i * dtg
    for src in sources:
        D = src.apply_D(D, t)
    D, f, p = f2.step_2d_full_dform(D, f, grid, dtg, p, material)
    return D, f, p


trace_g = np.zeros(N_STEPS)
for i in range(N_STEPS):
    D, fields, pml = step(D, fields, pml, i)
    trace_g[i] = float(fields.Ey[i_pr, j_pr])

print("GPU trace:", len(trace_g), "max |Ey| =", np.abs(trace_g).max())

# ---------------- compare ----------------
m = np.real(trace_m[:N_STEPS])
gt = trace_g[: len(m)]
np.savez("/tmp/diag_td12.npz", meep=m, gpu=gt, dt=dt)

# step-function eval order in MEEP's run() can offset the trace by ±1 sample —
# report rel-L2 for several alignments and detail the best one.
n = len(m)
best = None
for sh in (-2, -1, 0, 1, 2):
    a = m[max(0, sh):n + min(0, sh)]
    b = gt[max(0, -sh):n - max(0, sh)]
    k = min(len(a), len(b)); a, b = a[:k], b[:k]
    rel = np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-300)
    print(f"shift {sh:+d}: rel-L2 = {rel:.6e}")
    if best is None or rel < best[1]:
        best = (sh, rel, a, b)
sh, rel, a, b = best
print(f"\nbest shift {sh:+d}, rel-L2 = {rel:.6e}")
nz = np.nonzero(np.abs(a) > 1e-12 * np.abs(a).max())[0]
first = nz[0] if len(nz) else 0
print(f"{'i':>6} {'MEEP':>15} {'GPU':>15} {'ratio':>10}")
for i in list(range(first, first + 8)) + list(range(first + 400, first + 406)) + \
         list(range(len(a) - 6, len(a))):
    if 0 <= i < len(a):
        r = b[i] / a[i] if abs(a[i]) > 1e-300 else float("nan")
        print(f"{i:6d} {a[i]:15.6e} {b[i]:15.6e} {r:10.5f}")
