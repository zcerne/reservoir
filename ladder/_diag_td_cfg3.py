"""Time-domain probes for cfg3 (LC+dye): Ez pump at reservoir center + Ey signal.

Shift-aware sample-by-sample compare vs MEEP for the first N steps. Divergence
from the first nonzero pump sample => area-source injection mismatch; later
divergence growing with field => gain dynamics mismatch.
"""
import os, sys
import numpy as np

RESEVOIR = os.environ.get("RESEVOIR", "/home/cernez/resevoir")
sys.path.insert(0, RESEVOIR)
sys.path.insert(0, os.path.join(RESEVOIR, "ladder"))
N_STEPS = int(os.environ.get("DIAG_STEPS", "3000"))

import ladder  # noqa: E402

path = ladder.build_json(int(os.environ.get('DIAG_CFG', '3')))
print("config dir:", path)

import meep as mp  # noqa: E402
from class_simulation import Simulation  # noqa: E402

sim = Simulation(path)
sim._set_everything()
s = sim.simulation
s.init_sim()
dt = s.fields.dt

# probes: reservoir center (x=-2.25 area center), Ez node & Ey face
PROBES = [("Ez_resv", mp.Ez, -2.25, 0.0), ("Ey_resv", mp.Ey, -2.25, 0.0125),
          ("Ey_mon", mp.Ey, 2.75, 0.0125)]
traces_m = {nm: [] for nm, *_ in PROBES}


def _rec(sim_):
    for nm, comp, x, y in PROBES:
        traces_m[nm].append(complex(sim_.get_field_point(comp, mp.Vector3(x, y, 0))))


s.run(_rec, until=N_STEPS * dt - 1e-9)
for k in traces_m:
    traces_m[k] = np.real(np.array(traces_m[k]))
print("MEEP max:", {k: float(np.abs(v).max()) for k, v in traces_m.items()})

# ---------------- GPU ----------------
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
gpu_src = os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src")
sys.path.insert(0, gpu_src)
import importlib  # noqa: E402
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
import fdtd_2d as f2  # noqa: E402
import multilevel as mlmod  # noqa: E402
import jax.numpy as jnp  # noqa: E402

g = csg.SimulationGPU(folder_path=path)
g.force_fullvector = True
g._set_data(); g._update_all_args(); g._build_material()
g.dt = float(g.args.get("courant", 0.5)) * g.dx
g._build_pml_full(); g._build_gain(); g._build_sources_sted()

idx = []
for nm, comp, x, y in PROBES:
    if comp == mp.Ez:
        i0 = int(round((x + g.cx) / g.dx)); j0 = int(round((y + g.cy) / g.dx))
    else:
        i0 = int(round((x + g.cx) / g.dx)); j0 = int(round((y + g.cy) / g.dx - 0.5))
    idx.append((i0, j0, "Ez" if comp == mp.Ez else "Ey"))
print("GPU probes:", idx)

grid, dtg, material, sources, pml = g.grid, g.dt, g.material, g.sources, g.pml
coeffs = g.gain["coeffs"]; ml_state = g.gain["state"]


def body(state, i):
    D, f, p, mls = state
    t = i * dtg
    def inj(D_):
        for s_ in sources:
            D_ = s_.apply_D(D_, t)
        return D_
    D, f, p, mls = f2.step_2d_full_gain_dform(D, f, grid, dtg, p, material,
                                              mls, coeffs, inject=inj)
    vals = jnp.stack([(f.Ez if c == "Ez" else f.Ey)[i0, j0] for i0, j0, c in idx])
    return (D, f, p, mls), vals


state0 = (f2.zero_D_full(g.grid), f2.zero_fields_full(g.grid), pml, ml_state)
_, tr = jax.lax.scan(body, state0, jnp.arange(N_STEPS))
tr = np.asarray(tr)
print("GPU max:", {PROBES[k][0]: float(np.abs(tr[:, k]).max()) for k in range(len(PROBES))})

for k, (nm, *_rest) in enumerate(PROBES):
    m = traces_m[nm][:N_STEPS]
    gt = tr[: len(m), k]
    best = None
    for sh in (0, 1):
        a = m[sh:]; b = gt[: len(a)]
        kk = min(len(a), len(b)); a, b = a[:kk], b[:kk]
        rel = np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-300)
        if best is None or rel < best[1]:
            best = (sh, rel, a, b)
    sh, rel, a, b = best
    thr_base = np.abs(a).max()
    bad = np.nonzero(np.abs(b - a) > 1e-9 * thr_base)[0]
    first = bad[0] if len(bad) else -1
    print(f"{nm}: shift {sh:+d} rel-L2={rel:.6e} first-div(1e-9)={first}")
    # error growth curve: rel-L2 in windows of 1000 steps
    W = 1000
    for w0 in range(0, len(a) - W + 1, W):
        aa = a[w0:w0 + W]; bb = b[w0:w0 + W]
        na = np.linalg.norm(aa)
        r = np.linalg.norm(bb - aa) / (na + 1e-300)
        print(f"   window {w0:6d}-{w0+W:6d}: rel-L2={r:.3e}  |a|rms={na/np.sqrt(W):.3e}")
