"""Isolated MultilevelAtom comparison: vacuum + gain slab, periodic, no PML, no LC.

Cell 10x6, eps=1 everywhere; 4-level atom (cfg3 params) in slab |x|<=2.5.
Ey line signal at x=-4 (lam 0.5) + Ez area pump over the slab (lam 0.53, amp
LADDER-like). Probes Ey, Ez, Dz at slab center. With eps=1 and no off-diagonal,
P_z = Dz - Ez EXACTLY in both engines -> direct P comparison.
"""
import os, sys
import numpy as np

N_STEPS = int(os.environ.get("DIAG_STEPS", "3000"))
res = 40; dx = 1.0 / res; dt = 0.5 * dx
CX, CY = 10.0, 6.0
SLAB = 5.0
# cfg3 sted params
lbdA, gammaA, lbdE, gammaE = 0.53, 0.13, 0.55, 0.13
SGMA, N3_0, r43, r21 = 0.006, 25.0, 10.0, 100.0
PUMP_AMP = 300.0

import meep as mp  # noqa: E402

trans = [
    mp.Transition(1, 4, frequency=1 / lbdA, gamma=gammaA, sigma_diag=mp.Vector3(1, 1, 1)),
    mp.Transition(4, 3, transition_rate=r43),
    mp.Transition(2, 3, frequency=1 / lbdE, gamma=gammaE, sigma_diag=mp.Vector3(1, 1, 1)),
    mp.Transition(2, 1, transition_rate=r21),
]
atom_mp = mp.MultilevelAtom(sigma=SGMA, transitions=trans,
                            initial_populations=[0.0, 0.0, N3_0, 0.0])
gain_med = mp.Medium(epsilon=1.0, E_susceptibilities=[atom_mp])

sim = mp.Simulation(
    cell_size=mp.Vector3(CX, CY, 0), resolution=res, boundary_layers=[],
    k_point=mp.Vector3(0, 0, 0), force_complex_fields=False,
    geometry=[mp.Block(center=mp.Vector3(0, 0, 0),
                       size=mp.Vector3(SLAB, CY, mp.inf), material=gain_med)],
    sources=[
        mp.Source(mp.GaussianSource(1 / 0.5, width=(20.0 / 3.335640952) / 2.35482),
                  component=mp.Ey, center=mp.Vector3(-4.0, 0, 0),
                  size=mp.Vector3(0, CY, 0)),
        mp.Source(mp.GaussianSource(1 / 0.53, width=(200.0 / 3.335640952) / 2.35482),
                  component=mp.Ez, center=mp.Vector3(0, 0, 0),
                  size=mp.Vector3(SLAB, CY, 0), amplitude=PUMP_AMP),
    ])
sim.init_sim()
assert abs(sim.fields.dt - dt) < 1e-15

tm = {"Ey": [], "Ez": [], "Dz": []}
P0 = mp.Vector3(0.0125, 0.0125, 0)   # x=y=0.0125: Ey face & near-node sample points
PN = mp.Vector3(0.0, 0.0, 0)         # exact node for Ez/Dz


def _rec(sim_):
    tm["Ey"].append(np.real(complex(sim_.get_field_point(mp.Ey, mp.Vector3(0.0, 0.0125, 0)))))
    tm["Ez"].append(np.real(complex(sim_.get_field_point(mp.Ez, PN))))
    tm["Dz"].append(np.real(complex(sim_.get_field_point(mp.Dz, PN))))


sim.run(_rec, until=N_STEPS * dt - 1e-9)
for k in tm:
    tm[k] = np.array(tm[k])
print("MEEP max:", {k: float(np.abs(v).max()) for k, v in tm.items()})

# ---------------- GPU ----------------
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
gpu_src = os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src")
sys.path.insert(0, gpu_src)
import fdtd_2d as f2  # noqa: E402
import pml_meep  # noqa: E402
import multilevel as ml  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from class_simulation_gpu import _STEDSource, _src_overlap_weights, _src_delta_weights  # noqa: E402

Nx, Ny = int(CX * res), int(CY * res)
grid = f2.Grid2D(Nx, Ny, dx, dx)
one = jnp.ones((Nx, Ny)); zero = jnp.zeros((Nx, Ny))
mat = f2.AnisoFull2D(ixx_Ex=one, ixy_Ex=zero, ixz_Ex=zero,
                     ixy_Ey=zero, iyy_Ey=one, iyz_Ey=zero,
                     ixz_nd=zero, iyz_nd=zero, izz_nd=one)
pml = pml_meep.make_meep_upml(grid, dt, n_pml=0)

atom = ml.MultilevelAtom(4, [
    ml.Transition(1, 4, frequency=1 / lbdA, gamma=gammaA, sigma=1.0),
    ml.Transition(4, 3, transition_rate=r43),
    ml.Transition(2, 3, frequency=1 / lbdE, gamma=gammaE, sigma=1.0),
    ml.Transition(2, 1, transition_rate=r21),
], [0.0, 0.0, N3_0, 0.0], sigma=SGMA, sigma_diag=(1.0, 1.0, 1.0))
coeffs = ml.build_coeffs(atom, dt)

i = np.arange(Nx); j = np.arange(Ny)
half = 0.5 * dx; icx = Nx - Nx % 2; icy = Ny - Ny % 2


def _mask(sx_off, sy_off):
    X = ((2 * i + sx_off - icx) * half)[:, None]
    Y = ((2 * j + sy_off - icy) * half)[None, :]
    return jnp.asarray(((np.abs(X) <= 0.5 * SLAB) & (np.abs(Y) <= 0.5 * CY))
                       .astype(np.float64))


mst = ml.init_state_full(atom, coeffs, (Nx, Ny), _mask(0, 0),
                         masks=(_mask(1, 0), _mask(0, 1), _mask(0, 0)))

cx = (Nx // 2) * dx; cy = (Ny // 2) * dx
FS = 3.335640952
sources = []
# Ey line signal at x=-4
wy = _src_overlap_weights(-CY / 2, CY / 2, Ny, dx, 0.5, cy, wrap=True)
wx = _src_delta_weights(-4.0, Nx, dx, 0.0, cx)
sources.append(_STEDSource(component="Ey",
                           amp_map=jnp.asarray(wx[:, None] * wy[None, :]),
                           eps_inv_map=one, freq=1 / 0.5,
                           width=(20.0 / FS) / 2.35482, start_time=0.0,
                           cutoff=5.0, dt=dt, src_scale=float(res)))
# Ez area pump over slab
wxp = _src_overlap_weights(-SLAB / 2, SLAB / 2, Nx, dx, 0.0, cx, wrap=True)
wyp = _src_overlap_weights(-CY / 2, CY / 2, Ny, dx, 0.0, cy, wrap=True)
sources.append(_STEDSource(component="Ez",
                           amp_map=jnp.asarray(PUMP_AMP * wxp[:, None] * wyp[None, :]),
                           eps_inv_map=one, freq=1 / 0.53,
                           width=(200.0 / FS) / 2.35482, start_time=0.0,
                           cutoff=5.0, dt=dt, src_scale=float(res) * dx))

i_pr = int(round((0.0 + cx) / dx)); j_pr_ey = int(round((0.0125 + cy) / dx - 0.5))
j_pr_nd = int(round((0.0 + cy) / dx))
i_pr_ey = int(round((0.0 + cx) / dx))


def body(state, k):
    D, f, p, mls = state
    t = k * dt
    def inj(D_):
        for s_ in sources:
            D_ = s_.apply_D(D_, t)
        return D_
    D, f, p, mls = f2.step_2d_full_gain_dform(D, f, grid, dt, p, mat, mls, coeffs,
                                              inject=inj)
    return (D, f, p, mls), jnp.stack([f.Ey[i_pr_ey, j_pr_ey],
                                      f.Ez[i_pr, j_pr_nd], D[2][i_pr, j_pr_nd]])


state0 = (f2.zero_D_full(grid), f2.zero_fields_full(grid), pml, mst)
_, tr = jax.lax.scan(body, state0, jnp.arange(N_STEPS))
tr = np.asarray(tr)
gp = {"Ey": tr[:, 0], "Ez": tr[:, 1], "Dz": tr[:, 2]}
print("GPU max:", {k: float(np.abs(v).max()) for k, v in gp.items()})

for nm in ("Ey", "Ez", "Dz"):
    m = tm[nm][:N_STEPS]
    best = None
    for sh in (0, 1):
        a = m[sh:]; b = gp[nm][: len(a)]
        kk = min(len(a), len(b)); a, b = a[:kk], b[:kk]
        rel = np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-300)
        if best is None or rel < best[1]:
            best = (sh, rel, a, b)
    sh, rel, a, b = best
    bad = np.nonzero(np.abs(b - a) > 1e-9 * np.abs(a).max())[0]
    first = bad[0] if len(bad) else -1
    print(f"{nm}: shift {sh:+d} rel-L2={rel:.6e} first-div(1e-9)={first}")

# P_z = Dz − Ez (eps=1, no offdiag) — compare with best shift 1
a = (tm["Dz"] - tm["Ez"])[1:N_STEPS]
b = (gp["Dz"] - gp["Ez"])[: len(a)]
rel = np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-300)
print(f"Pz = Dz-Ez: rel-L2={rel:.6e}  meep max={np.abs(a).max():.4e} gpu max={np.abs(b).max():.4e}")
i0 = np.nonzero(np.abs(a) > 1e-10 * np.abs(a).max())[0]
if len(i0):
    k0 = i0[0]
    for k in range(k0, min(k0 + 6, len(a))):
        print(f"   step {k}: meepP={a[k]:+.9e} gpuP={b[k]:+.9e}")
