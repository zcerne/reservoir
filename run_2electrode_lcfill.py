"""Shaped electrode EMBEDDED in LC (no air recess) — true geometry encoding.

Domain (2D, light along +x):  x ∈ [−EXT, GAP], y ∈ [−SPAN/2, SPAN/2]
  * shaped conductor at V0: metal fills x ≤ −d(y), d(y) = Σ c_k B_k(y) spline
    recess (c_k ∈ [0, EXT−1]) — a single connected piece by construction;
  * LC fills EVERYTHING else in the domain (no air divider);
  * ground plane at x = GAP; graded far y-walls in the Poisson solve.
Optically the electrode body is ITO-on-glass ≈ index-matched: modeled as
uniform director φ=0 (index n_o for the Ey-polarized light) — stated
approximation. The FDTD reservoir spans the full extended domain.

MODE=compare : evaluate the SAME shape (mapped from the optimized spline-V
               design via d = GAP·(V0/V(y)−1)) with (a) air around the metal
               vs (b) LC-filled; report LC-region field strength + focus frac.
MODE=optimize: BOBYQA over the 12 recess coefficients (LC-filled), Gaussian
               center target, resume from history.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODE = os.environ.get("MODE", "optimize")
GAP, EXT, SPAN, NCTRL = 12.0, 8.0, 24.0, 12
V0 = 2.75
TARGET_SIGMA = 1.5
RES_LC = 4                                   # LC/Poisson grid (pts per um)
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "twoelectrode", f"lcfill_{MODE}")

SX = EXT + GAP                                # reservoir x-size (20 um)
cfg = {
    "resolution": 40, "use_cw": False, "run_until": 70,
    "dimention": 2, "cell_size_y": SPAN + 3.0, "periodic": False,
    "pml_size": 1.5, "background_index": 1.0,
    "snapshot_t1": 1e9, "snapshot_t2": 1e9, "snapshot_dt": 1.0,
    "solver": "gpumeep",
    "guide_1": {"class": "guide", "index": 1.0, "sizes": [0.5, SPAN]},
    "reservoir": {
        "class": "voltage_reservoir", "sizes": [SX, SPAN], "resolution": RES_LC,
        "n_o": 1.52, "n_e": 1.71,
        "boundary_conditions": ["free", "free", "free"],
        "elastic_constants": {"K1": 11.1, "K2": 2.0, "K3": 17.1, "q0": 0.0},
        "eps_perp_dc": 5.0, "eps_a_dc": 10.0,
        "poisson_rtol": 1e-7, "poisson_maxiter": 20000,
        "voltages_x_min": [], "voltages_x_max": [],
    },
    "guide_2": {"class": "guide", "index": 1.0, "sizes": [5.0, SPAN]},
    "source_1": {"class": "source", "position": {"on_object": "guide_1",
                 "position": "center", "size": [0.0, SPAN, 0.0]},
                 "amplitude": [1.0], "component": "Ey", "source_type": "pulsed",
                 "lam": 0.5, "dlam": 0.0, "pulse_fwhm_fs": 20.0,
                 "pulse_delay_fs": 0.0},
    "monitor_2": {"class": "monitor", "type": "1Ddft", "on_object": "guide_2",
                  "position": {"position": "right", "size": SPAN},
                  "lam_range": [0.5, 0.5], "n_lam": 1},
    "object_order": ["guide_1", "reservoir", "guide_2", "source_1", "monitor_2"],
}
os.makedirs(os.path.join(BASE, "simulation"), exist_ok=True)
json.dump(cfg, open(os.path.join(BASE, "simulation_data.json"), "w"), indent=2)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import electrostatics_jax as esj
from class_lc_from_field import LCFromField

# ---- grids ----
nx = int(round(SX * RES_LC)) + 1              # 81
ny = int(round(SPAN * RES_LC)) + 1            # 97
nz = 5
dx = SX / (nx - 1); dy = SPAN / (ny - 1); dz = 4.0 / RES_LC / (nz - 1)
xg = np.linspace(-EXT, GAP, nx)               # x=0 is the nominal LC-cell start
yg = np.linspace(-SPAN / 2, SPAN / 2, ny)

# graded y padding (walls at infinity)
w_y, ylo_i, yhi_i = esj.graded_axis_widths(ny, dy, 20, growth=1.2, w_max=2.0)
ny_p = len(w_y)
pad = ylo_i
spacings = (dx, np.asarray(w_y), dz)
gshape_p = (nx, ny_p, nz)
core = (slice(None), slice(pad, pad + ny), slice(None))

_lc = LCFromField(BASE)


def _bspline_matrix(y, lo, hi, n, deg=3):
    def cdb(t, kn, k, d):
        if d == 0:
            last = kn[k + 1] == kn[-1]
            return np.where((t >= kn[k]) & ((t < kn[k + 1]) | last), 1.0, 0.0)
        out = np.zeros_like(t)
        d1 = kn[k + d] - kn[k]
        if d1 > 0:
            out = out + (t - kn[k]) / d1 * cdb(t, kn, k, d - 1)
        d2 = kn[k + d + 1] - kn[k + 1]
        if d2 > 0:
            out = out + (kn[k + d + 1] - t) / d2 * cdb(t, kn, k + 1, d - 1)
        return out
    kn = np.concatenate([np.zeros(deg), np.linspace(0, 1, n + deg + 1 - 2 * deg),
                         np.ones(deg)])
    t = np.clip((y - lo) / (hi - lo), 0, 1)
    return np.stack([cdb(t, kn, k, deg) for k in range(n)], axis=1)


_B = _bspline_matrix(yg, -SPAN / 2, SPAN / 2, NCTRL)


def metal_mask(coeffs):
    """(nx, ny) bool: metal at x ≤ −d(y)."""
    d = np.clip(_B @ np.asarray(coeffs), 0.0, EXT - 1.0)
    return xg[:, None] <= -d[None, :]


def solve_poisson(metal2d, lc_everywhere, phi, theta):
    """Poisson on the padded grid. metal cells = Dirichlet V0; ground at
    x=GAP wall; outside-LC (air recess case) ε=1."""
    mask = np.zeros(gshape_p, dtype=bool)
    Vd = np.zeros(gshape_p)
    m3 = np.repeat(metal2d[:, :, None], nz, axis=2)
    mask[core] = m3
    Vd[core] = np.where(m3, V0, 0.0)
    mask[-1, :, :] = True                     # ground plane (all y incl. padding)
    # ε: LC tensor from director; metal + (air case: x<0 non-metal) overridden
    phi_p = np.zeros(gshape_p); theta_p = np.full(gshape_p, np.pi / 2)
    phi_p[core] = phi; theta_p[core] = theta
    eps = esj.build_eps_diag_jax(jnp.asarray(phi_p), jnp.asarray(theta_p),
                                 5.0, 10.0)
    override = np.zeros(gshape_p, dtype=bool)
    override[core] = m3                       # metal body: ε irrelevant (Dirichlet)
    pad_reg = np.ones(gshape_p, dtype=bool)
    pad_reg[core] = False                     # y-padding region
    if not lc_everywhere:
        airreg = np.zeros(gshape_p, dtype=bool)
        airreg[core] = np.repeat((xg[:, None] < 0)[:, :, None], nz, axis=2) & ~m3
        eps = jnp.where(jnp.asarray(airreg)[None], 1.0, eps)
        eps = jnp.where(jnp.asarray(pad_reg)[None], 1.0, eps)
    else:
        eps = jnp.where(jnp.asarray(pad_reg)[None], 5.0, eps)   # LC-ish outside
    V = esj.solve_poisson_jax(eps, spacings, jnp.asarray(mask),
                              jnp.asarray(Vd), periodic=(False, False, True),
                              rtol=1e-7, maxiter=20000)
    E = esj.gradient_V_jax(V, spacings)
    return np.asarray(V)[core], np.asarray(E)[(slice(None),) + core]


SCL_MAX, SCL_TOL = 10, 1e-4
_phi_warm = [None]


def relax(coeffs, lc_everywhere=True):
    m2 = metal_mask(coeffs)
    phi = np.zeros((nx, ny, nz)) if _phi_warm[0] is None else _phi_warm[0]
    theta = np.full((nx, ny, nz), np.pi / 2)
    for it in range(SCL_MAX):
        V, E = solve_poisson(m2, lc_everywhere, phi, theta)
        phi_new, theta = _lc.compute(E, (nx, ny, nz), (dx, dy, dz),
                                     phi_init=phi, full_3d=False)
        err = np.linalg.norm(phi_new - phi) / max(np.linalg.norm(phi_new), 1e-20)
        phi = phi_new
        if err < SCL_TOL:
            break
    _phi_warm[0] = phi
    # electrode body: optically index-matched glass/ITO ≈ φ=0 (n_o for Ey);
    # air case: recess region has no LC → also φ=0 (plus n_o≈vacuum mismatch
    # noted — the optical contrast of the air region is NOT modeled; compare
    # uses the same optical model so the difference isolates the E-field).
    m3 = np.repeat(m2[:, :, None], nz, axis=2)
    phi_opt = np.where(m3, 0.0, phi)
    np.savez(os.path.join(BASE, "simulation", "lc_fields.npz"),
             phi=phi_opt, theta=theta, x=np.linspace(-SX/2, SX/2, nx),
             y=yg, z=np.linspace(0, dz*(nz-1), nz))
    return phi, E, m2


def forward():
    import importlib
    gpu_src = os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src")
    if gpu_src not in sys.path:
        sys.path.insert(0, gpu_src)
    sys.modules.pop("class_simulation_gpu", None)
    csg = importlib.import_module("class_simulation_gpu")
    sim = csg.SimulationGPU(folder_path=BASE)
    sim.force_fullvector = True
    sim.run()
    I = np.abs(np.load(os.path.join(BASE, "simulation", "monitor_2.npz"))["Ey"][0]) ** 2
    y = np.linspace(-SPAN/2, SPAN/2, len(I))
    G = np.exp(-y**2/(2*TARGET_SIGMA**2))
    return float((I*G).sum()/max(I.sum(), 1e-300)), I, y


_history = []


def evaluate(coeffs, lc_everywhere=True, tag=""):
    t0 = time.time()
    phi, E, m2 = relax(coeffs, lc_everywhere)
    frac, I, y = forward()
    # LC-region field metric: mean |E| in the nominal cell (0..GAP)
    lcreg = (xg >= 0)
    Emag = np.linalg.norm(E[:2], axis=0)[:, :, nz//2]
    Elc = float(Emag[lcreg].mean())
    _history.append((list(map(float, coeffs)), frac, Elc))
    np.savez(os.path.join(BASE, "opt_history.npz"),
             coeffs=np.array([h[0] for h in _history]),
             frac=np.array([h[1] for h in _history]),
             Elc=np.array([h[2] for h in _history]), best_I=I, y=y)
    print(f"[lcfill]{tag} frac={frac:.4f} <|E|>_LC={Elc:.4f} V/um "
          f"({time.time()-t0:.0f}s)", flush=True)
    return frac


def main():
    if MODE == "compare":
        # shape mapped from the optimized spline-V design
        vco = np.array([1.5046772628734646, 1.4407980919662493, 1.2956922179690635,
                        1.4043265651739856, 2.7506995901849938, 0.7735502413014868,
                        1.3424101744210812, 1.63421061707266, 1.3502857780752817,
                        1.5493345559163563, 1.540217985011447, 1.4341406482134025])
        Vy_ctrl = vco
        d_ctrl = np.clip(GAP * (Vy_ctrl.max() / Vy_ctrl - 1.0), 0, EXT - 1.0)
        print(f"[lcfill] shape recess coeffs: {np.round(d_ctrl, 2).tolist()}")
        _phi_warm[0] = None
        evaluate(d_ctrl, lc_everywhere=False, tag=" AIR   ")
        _phi_warm[0] = None
        evaluate(d_ctrl, lc_everywhere=True, tag=" LCFILL")
        print("[lcfill] COMPARE DONE", flush=True)
        return
    import nlopt
    opt = nlopt.opt(nlopt.LN_BOBYQA, NCTRL)
    opt.set_lower_bounds([0.0] * NCTRL)
    opt.set_upper_bounds([EXT - 1.0] * NCTRL)
    opt.set_min_objective(lambda c, g=None: 1.0 - evaluate(c))
    opt.set_maxeval(int(os.environ.get("OPT_MAXEVAL", "120")))
    x0 = np.full(NCTRL, 4.0)
    hist = os.path.join(BASE, "opt_history.npz")
    if os.path.exists(hist):
        h = np.load(hist)
        k = int(np.argmax(h["frac"]))
        x0 = np.asarray(h["coeffs"][k], dtype=np.float64)
        print(f"[lcfill] resuming from best of {len(h['frac'])} evals", flush=True)
    try:
        opt.optimize(x0)
    except KeyboardInterrupt:
        pass
    best = max(_history, key=lambda h: h[1])
    print(f"[lcfill] DONE best frac={best[1]:.4f} recess={np.round(best[0],2).tolist()}",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
