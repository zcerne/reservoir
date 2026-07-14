"""Time-domain point probes for a 3D ladder config: MEEP vs GPUmeep."""
import os, sys
import numpy as np

RESEVOIR = os.environ.get("RESEVOIR", "/home/cernez/resevoir")
sys.path.insert(0, RESEVOIR)
sys.path.insert(0, os.path.join(RESEVOIR, "ladder"))
N_STEPS = int(os.environ.get("DIAG_STEPS", "4400"))
CFG = int(os.environ.get("DIAG_CFG", "2"))

import ladder  # noqa: E402

path = ladder.build_json(CFG)

import meep as mp  # noqa: E402
from class_simulation import Simulation  # noqa: E402

sim = Simulation(path)
sim._set_everything()
s = sim.simulation
s.init_sim()
dt = s.fields.dt

PROBES = [("Ey_resv", mp.Ey, 0.25, 0.025, 0.0), ("Ey_mon", mp.Ey, 2.5, 0.025, 0.0)]
tm = {nm: [] for nm, *_ in PROBES}


def _rec(sim_):
    for nm, comp, x, y, z in PROBES:
        tm[nm].append(np.real(complex(sim_.get_field_point(comp, mp.Vector3(x, y, z)))))


s.run(_rec, until=N_STEPS * dt - 1e-9)
for k in tm:
    tm[k] = np.array(tm[k])
print("MEEP max:", {k: float(np.abs(v).max()) for k, v in tm.items()})

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src"))
import importlib  # noqa: E402
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
import fdtd3d_meep as f3  # noqa: E402
import jax.numpy as jnp  # noqa: E402

g = csg.SimulationGPU(folder_path=path)
g._set_data(); g._update_all_args(); g._build_material_3d()
g.dt = 0.5 * g.dx
n_pml = int(round(float(g.args.get("pml_size", 2.0)) / g.dx))
g.pml = f3.make_meep_upml_3d(g.grid, g.dt, n_pml=n_pml)
g._build_gain_3d(); g._build_sources_3d()

idx = []
for nm, comp, x, y, z in PROBES:
    i0 = int(round((x + g.cx) / g.dx))
    j0 = int(round((y + g.cy) / g.dx - 0.5))   # Ey face
    k0 = int(round((z + g.cz) / g.dx))
    idx.append((i0, j0, k0))
print("GPU probes:", idx, "dt", g.dt)

grid, dtg, material, sources, pml = g.grid, g.dt, g.material, g.sources, g.pml
run_until = float(g.args.get("run_until", 60.0))
n_src_on = int(round(run_until / dtg))


def _apply(d, t_):
    for s_ in sources:
        d = s_.apply_D(d, t_)
    return d


def body(state, i):
    D, f, p = state
    t = i * dtg
    inj = lambda D_: jax.lax.cond(i < n_src_on, lambda d: _apply(d, t),
                                  lambda d: d, D_)
    D, f, p = f3.step_3d_dform(D, f, grid, dtg, p, material, inject=inj)
    return (D, f, p), jnp.stack([f.Ey[i0, j0, k0] for i0, j0, k0 in idx])


state0 = (f3.zero_D_3d(g.grid), g.grid.zero_fields(), pml)
_, tr = jax.lax.scan(body, state0, jnp.arange(N_STEPS))
tr = np.asarray(tr)
print("GPU max:", {PROBES[k][0]: float(np.abs(tr[:, k]).max()) for k in range(len(PROBES))})

for k, (nm, *_r) in enumerate(PROBES):
    m = tm[nm][:N_STEPS]
    best = None
    for sh in (0, 1):
        a = m[sh:]; b = tr[: len(a), k]
        kk = min(len(a), len(b)); a, b = a[:kk], b[:kk]
        rel = np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-300)
        if best is None or rel < best[1]:
            best = (sh, rel, a, b)
    sh, rel, a, b = best
    print(f"{nm}: shift {sh:+d} rel-L2={rel:.6e}")
    W = 400
    for w0 in range(0, len(a) - W + 1, W * 2):
        aa = a[w0:w0 + W]; bb = b[w0:w0 + W]
        na = np.linalg.norm(aa)
        print(f"   win {w0:5d}-{w0+W:5d}: rel={np.linalg.norm(bb-aa)/(na+1e-300):.3e} "
              f"|meep|rms={na/np.sqrt(W):.3e} |gpu|rms={np.linalg.norm(bb)/np.sqrt(W):.3e}")
