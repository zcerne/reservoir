"""Full-field error-map diagnostic for config 1: where is the GPU−MEEP error born?

Captures Ey over the whole cell at several early times in both engines
(GPU collocated onto MEEP's get_array integer grid) and reports max|err|
per region: interior, left/right x-PML, bottom/top y-PML, corners.
"""
import os, sys
import numpy as np

RESEVOIR = os.environ.get("RESEVOIR", "/home/cernez/resevoir")
sys.path.insert(0, RESEVOIR)
sys.path.insert(0, os.path.join(RESEVOIR, "ladder"))
import ladder  # noqa: E402

path = ladder.build_json(1)
SNAP_STEPS = [400, 800, 1600, 3200]      # t = 5, 10, 20, 40

# ---------------- MEEP ----------------
import meep as mp  # noqa: E402
from class_simulation import Simulation  # noqa: E402

sim = Simulation(path)
sim._set_everything()
s = sim.simulation
s.init_sim()
dt = s.fields.dt
cell = s.cell_size
snaps_m = {}
count = [0]


def _rec(sim_):
    count[0] += 1
    if count[0] in SNAP_STEPS:
        snaps_m[count[0]] = np.array(sim_.get_array(
            center=mp.Vector3(), size=mp.Vector3(cell.x, cell.y, 0), component=mp.Ey))


s.run(_rec, until=(max(SNAP_STEPS) + 1) * dt - 1e-9)
print("MEEP snaps:", {k: v.shape for k, v in snaps_m.items()})

# ---------------- GPUmeep ----------------
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
gpu_src = os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src")
sys.path.insert(0, gpu_src)
import importlib  # noqa: E402
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
import fdtd_2d as f2  # noqa: E402

g = csg.SimulationGPU(folder_path=path)
g.force_fullvector = True
g._set_data(); g._update_all_args(); g._build_material()
g.dt = float(g.args.get("courant", 0.5)) * g.dx
g._build_pml_full(); g._build_sources_sted()
grid, dtg, material, sources, pml = g.grid, g.dt, g.material, g.sources, g.pml
Nx, Ny = g.Nx, g.Ny

D = f2.zero_D_full(g.grid); fields = f2.zero_fields_full(g.grid)
import jax.numpy as jnp  # noqa: E402


@jax.jit
def step(D, f, p, i):
    t = i * dtg
    def inj(D_):
        for s_ in sources:
            D_ = s_.apply_D(D_, t)
        return D_
    return f2.step_2d_full_dform(D, f, grid, dtg, p, material, inject=inj)


snaps_g = {}
for i in range(max(SNAP_STEPS) + 1):
    D, fields, pml = step(D, fields, pml, i)
    if (i + 1) in SNAP_STEPS:
        snaps_g[i + 1] = np.asarray(fields.Ey)

# ---------------- compare ----------------
n_pml = int(round(1.5 / g.dx))     # 60 cells


def region(i, j):
    rx = "L" if i < n_pml else ("R" if i >= Nx - n_pml else "-")
    ry = "B" if j < n_pml else ("T" if j >= Ny - n_pml else "-")
    if rx == "-" and ry == "-":
        return "interior"
    if rx != "-" and ry != "-":
        return "corner"
    return f"xpml-{rx}" if rx != "-" else f"ypml-{ry}"


for st in SNAP_STEPS:
    m = snaps_m[st]                       # MEEP get_array: integer grid (Nx+1, Ny+1)
    ggrid = snaps_g[st]                   # GPU Ey at (i, j+1/2)
    # collocate GPU Ey onto integer grid like get_array: avg in y
    gcol = 0.5 * (ggrid[:, :-1] + ggrid[:, 1:])     # (Nx, Ny-1) at (i, j) j=1..Ny-1
    mm = m[:Nx, 1:Ny]
    gg = gcol
    err = np.abs(gg - mm)
    peak = np.abs(mm).max()
    stats = {}
    for i in range(0, Nx, 4):
        for j in range(0, Ny - 1, 4):
            r = region(i, j + 1)
            v = err[i, j]
            if r not in stats or v > stats[r][0]:
                stats[r] = (v, i, j + 1)
    print(f"step {st} (t={st*dtg:.1f}): peak|Ey|={peak:.3e}")
    for r, (v, i, j) in sorted(stats.items()):
        print(f"   {r:9s} max|err|={v:.3e} ({v/peak:.2e} of peak) at (i={i}, j={j})")
