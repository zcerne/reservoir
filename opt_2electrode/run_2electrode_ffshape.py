"""SHAPED single electrode at ONE potential — far-field steering targets.

The design goal (user 2026-07-14): one connected metal electrode whose SHAPE
is the optimization variable, held at a single potential V0; ground plane on
the opposite wall. Metal fills x <= -d(y), d(y) = spline recess (12 control
points in [0, EXT-1]); LC fills everything else (LC-fill variant — 4.1x field
gain). Optionally V0 is ONE extra scalar dof (FF_OPT_V0=1).

LC walls: SOFT anchoring (Rapini-Papoular, W = FF_W) phi0 = pi/2 on the y
walls — the winning A/B configuration. Cost: absolute phase-aware far-field
overlap |sum Ey_far * t(theta)|^2, t = Gaussian at THETA0_DEG with SIGMA_DEG,
projected with GPUmeep's MEEP-exact near2far from a 2Ddft strip in the exit
guide.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GAP, EXT, SPAN, NCTRL = 12.0, 8.0, 24.0, 12
RES_LC = 4
SIGMA_DEG = float(os.environ.get("SIGMA_DEG", "3.0"))
THETA0_DEG = float(os.environ.get("THETA0_DEG", "0.0"))
FF_W = float(os.environ.get("FF_W", "11.1"))
OPT_V0 = os.environ.get("FF_OPT_V0", "1") == "1"
V0_FIXED = float(os.environ.get("V0", "2.75"))
TAG = os.environ.get("TAG", f"ffshape_t{int(THETA0_DEG)}")
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "twoelectrode", TAG)
R_FAR, N_TH, TH_MAX = 2000.0, 361, np.deg2rad(45.0)

SX = EXT + GAP
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
        "lc_mode": "fwdbwd_relax", "lc_relax_maxeval": 600,
        "face_phi": [None, None, 1.5707963, 1.5707963, None, None],
        "face_anchor_W": [None, None, FF_W, FF_W, None, None],
    },
    "guide_2": {"class": "guide", "index": 1.0, "sizes": [5.0, SPAN]},
    "source_1": {"class": "source", "position": {"on_object": "guide_1",
                 "position": "center", "size": [0.0, SPAN, 0.0]},
                 "amplitude": [1.0], "component": "Ey", "source_type": "pulsed",
                 "lam": 0.5, "dlam": 0.0, "pulse_fwhm_fs": 20.0,
                 "pulse_delay_fs": 0.0},
    "monitor_2": {"class": "monitor", "type": "2Ddft", "on_object": "guide_2",
                  "position": {"position": "center", "size": [0.2, SPAN]},
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


def _bspline_matrix(y, lo, hi, n, deg=3):
    """Clamped uniform B-spline basis (copy of run_2electrode_lcfill's — that
    module has import side effects, so inlined here)."""
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

# ---- grids (identical structure to run_2electrode_lcfill) ----
nx = int(round(SX * RES_LC)) + 1
ny = int(round(SPAN * RES_LC)) + 1
nz = 5
dx = SX / (nx - 1); dy = SPAN / (ny - 1); dz = 4.0 / RES_LC / (nz - 1)
xg = np.linspace(-EXT, GAP, nx)
yg = np.linspace(-SPAN / 2, SPAN / 2, ny)

w_y, ylo_i, yhi_i = esj.graded_axis_widths(ny, dy, 20, growth=1.2, w_max=2.0)
ny_p = len(w_y)
pad = ylo_i
spacings = (dx, np.asarray(w_y), dz)
gshape_p = (nx, ny_p, nz)
core = (slice(None), slice(pad, pad + ny), slice(None))

_lc = LCFromField(BASE)
_B = _bspline_matrix(yg, -SPAN / 2, SPAN / 2, NCTRL)


def metal_mask(coeffs):
    d = np.clip(_B @ np.asarray(coeffs), 0.0, EXT - 1.0)
    return xg[:, None] <= -d[None, :]


def solve_poisson(metal2d, V0, phi, theta):
    mask = np.zeros(gshape_p, dtype=bool)
    Vd = np.zeros(gshape_p)
    m3 = np.repeat(metal2d[:, :, None], nz, axis=2)
    mask[core] = m3
    Vd[core] = np.where(m3, V0, 0.0)
    mask[-1, :, :] = True
    phi_p = np.zeros(gshape_p); theta_p = np.full(gshape_p, np.pi / 2)
    phi_p[core] = phi; theta_p[core] = theta
    eps = esj.build_eps_diag_jax(jnp.asarray(phi_p), jnp.asarray(theta_p),
                                 5.0, 10.0)
    pad_reg = np.ones(gshape_p, dtype=bool)
    pad_reg[core] = False
    eps = jnp.where(jnp.asarray(pad_reg)[None], 5.0, eps)
    V = esj.solve_poisson_jax(eps, spacings, jnp.asarray(mask),
                              jnp.asarray(Vd), periodic=(False, False, True),
                              rtol=1e-7, maxiter=20000)
    E = esj.gradient_V_jax(V, spacings)
    return np.asarray(V)[core], np.asarray(E)[(slice(None),) + core]


SCL_MAX, SCL_MAX_COLD, SCL_TOL, SCL_DEPTH = 10, 24, 1e-4, 5
_phi_warm = [None]


def relax(coeffs, V0):
    """Anderson-mixed self-consistent E ⇌ LC loop (type-I, depth SCL_DEPTH).
    Cold start (first eval) gets SCL_MAX_COLD iterations so the E-LC fixed
    point is fully converged before the optimization trusts any cost value."""
    m2 = metal_mask(coeffs)
    cold = _phi_warm[0] is None
    phi = np.zeros((nx, ny, nz)) if cold else _phi_warm[0]
    theta = np.full((nx, ny, nz), np.pi / 2)
    X, F = [], []
    n_it, err = 0, np.inf
    for it in range(SCL_MAX_COLD if cold else SCL_MAX):
        V, E = solve_poisson(m2, V0, phi, theta)
        phi_new, theta = _lc.compute(E, (nx, ny, nz), (dx, dy, dz),
                                     phi_init=phi, full_3d=False)
        r = (phi_new - phi).ravel()
        err = np.linalg.norm(r) / max(np.linalg.norm(phi_new.ravel()), 1e-20)
        n_it = it + 1
        if err < SCL_TOL:
            phi = phi_new
            break
        X.append(phi_new.ravel().copy()); F.append(r.copy())
        if len(X) > SCL_DEPTH:
            X.pop(0); F.pop(0)
        if len(F) >= 2:
            Fm = np.stack(F, axis=1)
            G = Fm.T @ Fm + 1e-12 * np.eye(Fm.shape[1])
            a = np.linalg.solve(G, np.ones(Fm.shape[1])); a /= a.sum()
            phi = (np.stack(X, axis=1) @ a).reshape((nx, ny, nz))
        else:
            phi = phi_new
    if cold:
        print(f"[scl] cold start converged in {n_it} iters (err {err:.2e})",
              flush=True)
    _phi_warm[0] = phi
    m3 = np.repeat(m2[:, :, None], nz, axis=2)
    phi_opt = np.where(m3, 0.0, phi)     # metal body: index-matched, phi=0
    np.savez(os.path.join(BASE, "simulation", "lc_fields.npz"),
             phi=phi_opt, theta=theta, x=np.linspace(-SX/2, SX/2, nx),
             y=yg, z=np.linspace(0, dz*(nz-1), nz))


# ---- far-field projection ----
DX = 1.0 / cfg["resolution"]
CELL_X = 1.5 + 0.5 + SX + 5.0 + 1.5
CX, CY = CELL_X / 2.0, cfg["cell_size_y"] / 2.0
X_LINE = -CX + 1.5 + 0.5 + SX + 2.5
X_EXIT = -CX + 1.5 + 0.5 + SX
THETAS = np.linspace(-TH_MAX, TH_MAX, N_TH)
FAR_PTS = np.stack([X_EXIT + R_FAR * np.cos(THETAS),
                    R_FAR * np.sin(THETAS)], axis=1)
_TMAT = [None]


def _far_field(mpath):
    import near2far_2d as n2f
    m = np.load(mpath)
    Ey2, Hz2 = m["Ey"][0], m["Hz"][0]
    if _TMAT[0] is None:
        ni, nj = Ey2.shape
        i_lo, j_lo = int(m["i_lo"]), int(m["j_lo"])
        i_line = int(round((X_LINE + CX) / DX))
        k = i_line - i_lo
        assert 0 < k < ni - 1, (ni, k)
        ys = (np.arange(nj) + j_lo + 0.5) * DX - CY
        T_E, T_H = n2f.transfer_matrices_line_x(FAR_PTS, i_line * DX - CX,
                                                ys, DX, DX, 2.0)
        _TMAT[0] = (T_E[:, [0, 1, 5]], T_H[:, [0, 1, 5]], k)
        print(f"[n2f] strip {Ey2.shape}, line col {k}", flush=True)
    T_E, T_H, k = _TMAT[0]
    EH = n2f.farfield_line_x(T_E, T_H, Ey2[k], Hz2[k - 1], Hz2[k])
    return EH[:, 1]                                   # Ey_far


def evaluate(x):
    coeffs = np.asarray(x[:NCTRL], dtype=np.float64)
    V0 = float(x[NCTRL]) if OPT_V0 else V0_FIXED
    relax(coeffs, V0)
    import importlib
    gpu_src = os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src")
    if gpu_src not in sys.path:
        sys.path.insert(0, gpu_src)
    sys.modules.pop("class_simulation_gpu", None)
    csg = importlib.import_module("class_simulation_gpu")
    sim = csg.SimulationGPU(folder_path=BASE)
    sim.force_fullvector = True
    sim.run()
    Ey_f = _far_field(os.path.join(BASE, "simulation", "monitor_2.npz"))
    I = np.abs(Ey_f) ** 2
    sig = np.deg2rad(SIGMA_DEG); th0 = np.deg2rad(THETA0_DEG)
    G = np.exp(-(THETAS - th0) ** 2 / (2.0 * sig ** 2))
    frac = float((I * G).sum() / max(I.sum(), 1e-300))
    ovl = float(np.abs((Ey_f * G).sum()) ** 2)
    return frac, ovl, I, V0


def main():
    import nlopt
    ndof = NCTRL + (1 if OPT_V0 else 0)
    opt = nlopt.opt(nlopt.LN_BOBYQA, ndof)
    lb = [0.0] * NCTRL + ([0.5] if OPT_V0 else [])
    ub = [EXT - 1.0] * NCTRL + ([7.0] if OPT_V0 else [])
    opt.set_lower_bounds(lb); opt.set_upper_bounds(ub)
    opt.set_maxeval(int(os.environ.get("OPT_MAXEVAL", "160")))
    _hist = []

    def cost(x, grad=None):
        t0 = time.time()
        frac, ovl, I, V0 = evaluate(x)
        _hist.append((list(map(float, x)), frac, ovl))
        np.savez(os.path.join(BASE, "opt_history.npz"),
                 xs=np.array([h[0] for h in _hist]),
                 frac=np.array([h[1] for h in _hist]),
                 ovl=np.array([h[2] for h in _hist]),
                 best_I=I, thetas=THETAS)
        print(f"[{TAG}] eval {len(_hist)}: ovl={ovl:.4g} frac={frac:.4f} "
              f"V0={V0:.2f} ({time.time()-t0:.0f}s)", flush=True)
        return -ovl

    opt.set_min_objective(cost)
    # Off-axis targets need a SYMMETRY-BROKEN start: from a flat contour a
    # local optimizer sees ~zero gradient toward a tilted beam (symmetric
    # shape -> symmetric far field). FF_INIT=ramp seeds a linear contour ramp
    # (sign = target side) so the beam already tilts and BOBYQA can climb.
    if os.environ.get("FF_INIT") == "ramp":
        s = 1.0 if THETA0_DEG >= 0 else -1.0
        ramp = np.clip(4.0 + s * np.linspace(-3.5, 3.5, NCTRL), 0.0, EXT - 1.0)
        x0 = np.concatenate([ramp, [V0_FIXED] if OPT_V0 else []])
    else:
        x0 = np.concatenate([np.full(NCTRL, 4.0), [V0_FIXED] if OPT_V0 else []])
    hist = os.path.join(BASE, "opt_history.npz")
    if os.path.exists(hist):
        h = np.load(hist)
        x0 = np.asarray(h["xs"][int(np.argmax(h["ovl"]))], dtype=np.float64)
        print(f"[{TAG}] resuming from best of {len(h['ovl'])} evals", flush=True)
    try:
        opt.optimize(x0)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if not _hist:
            raise
        print(f"[{TAG}] ended: {e}", flush=True)
    b = int(np.argmax([h[2] for h in _hist]))
    print(f"[{TAG}] DONE best ovl={_hist[b][2]:.4g} frac={_hist[b][1]:.4f} "
          f"x={np.round(_hist[b][0], 2).tolist()}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
