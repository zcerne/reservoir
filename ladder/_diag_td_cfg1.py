"""Time-domain point-probe comparison for ladder config 1 (air + PML).

Probes Ey(t) at exact Yee samples: interior/monitor (x=2.75) and inside the
left x-PML (x=-4.5; PML spans x<-3.75). Bit-exact interior + differing PML
point → PML dynamics; both bit-exact → readout-side residual.
"""
import os, sys
import numpy as np

RESEVOIR = os.environ.get("RESEVOIR", "/home/cernez/resevoir")
sys.path.insert(0, RESEVOIR)
sys.path.insert(0, os.path.join(RESEVOIR, "ladder"))
N_STEPS = int(os.environ.get("DIAG_STEPS", "4000"))

import ladder  # noqa: E402

path = ladder.build_json(1)
print("config dir:", path)

PROBES = [("monitor", 2.75, 0.0125), ("pml", -4.5, 0.0125)]

# ---------------- MEEP ----------------
import meep as mp  # noqa: E402
from class_simulation import Simulation  # noqa: E402

sim = Simulation(path)
sim._set_everything()
s = sim.simulation
s.init_sim()
dt = s.fields.dt
traces_m = {name: [] for name, _, _ in PROBES}


def _rec(sim_):
    for name, x, y in PROBES:
        traces_m[name].append(complex(sim_.get_field_point(mp.Ey, mp.Vector3(x, y, 0))))


s.run(_rec, until=N_STEPS * dt - 1e-9)
for k in traces_m:
    traces_m[k] = np.real(np.array(traces_m[k]))
print("MEEP:", {k: float(np.abs(v).max()) for k, v in traces_m.items()})

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
g._set_data(); g._update_all_args(); g._build_material()
g.dt = float(g.args.get("courant", 0.5)) * g.dx
g._build_pml_full(); g._build_sources_sted()

idx = []
for name, x, y in PROBES:
    i = int(round((x + g.cx) / g.dx))
    j = int(round((y + g.cy) / g.dx - 0.5))
    idx.append((i, j))
print("GPU probe indices:", idx, "dt =", g.dt)

grid, dtg, material, sources, pml = g.grid, g.dt, g.material, g.sources, g.pml


def body(state, i):
    D, f, p = state
    t = i * dtg
    def inj(D_):
        for s_ in sources:
            D_ = s_.apply_D(D_, t)
        return D_
    D, f, p = f2.step_2d_full_dform(D, f, grid, dtg, p, material, inject=inj)
    return (D, f, p), jnp.stack([f.Ey[i0, j0] for i0, j0 in idx])


state0 = (f2.zero_D_full(g.grid), f2.zero_fields_full(g.grid), pml)
_, tr = jax.lax.scan(body, state0, jnp.arange(N_STEPS))
tr = np.asarray(tr)
print("GPU :", {PROBES[k][0]: float(np.abs(tr[:, k]).max()) for k in range(len(PROBES))})

# ---------------- compare (shift-aware) ----------------
for k, (name, _, _) in enumerate(PROBES):
    m = traces_m[name][:N_STEPS]
    gt = tr[: len(m), k]
    best = None
    for sh in (0, 1):
        a = m[sh:]; b = gt[: len(a)]
        kk = min(len(a), len(b)); a, b = a[:kk], b[:kk]
        rel = np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-300)
        if best is None or rel < best[1]:
            best = (sh, rel, a, b)
    sh, rel, a, b = best
    print(f"{name}: shift {sh:+d} rel-L2 = {rel:.6e}  (maxdiff {np.abs(b-a).max():.3e})")
    # first step where they diverge beyond 1e-12 of peak
    thr = 1e-12 * max(np.abs(a).max(), 1e-300)
    bad = np.nonzero(np.abs(b - a) > thr)[0]
    if len(bad):
        i0 = bad[0]
        print(f"  first divergence at step {i0}: meep={a[i0]:.9e} gpu={b[i0]:.9e}")
    else:
        print("  bit-exact within 1e-12 of peak")
