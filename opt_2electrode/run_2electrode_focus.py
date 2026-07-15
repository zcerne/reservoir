"""2-electrode spline design — center-focus optimization (first design).

Geometry (2D, light along +x):
    [PML] air guide 0.5 | LC cell 12 x 24 um (spline electrode x_min, ground
    x_max, free LC BCs, graded y-padding for the Poisson solve) | air guide
    10 um | [PML];  1Ddft sensor at guide_2 CENTER = 5 um from the reservoir.

eval(coeffs): spline V(y) -> Poisson (graded) -> LC relax (free BCs) ->
lc_fields.npz -> GPUmeep forward -> I(y) at sensor -> cost = 1 - P_center/P_tot.
Optimizer: nlopt BOBYQA (12 dof, bounds [0, 3.5] V) with on-disk snapshots.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "twoelectrode", "focus12")
GAP, SPAN, NCTRL, VMAX = 12.0, 24.0, 12, 3.5
CENTER_BAND = 2.0            # +/- um around y=0 counted as "focused"

cfg = {
    "resolution": 40, "use_cw": False, "run_until": 60,
    "dimention": 2, "cell_size_y": SPAN + 2 * 1.5, "periodic": False,
    "pml_size": 1.5, "background_index": 1.0,
    "snapshot_t1": 1e9, "snapshot_t2": 1e9, "snapshot_dt": 1.0,
    "solver": "gpumeep",
    "guide_1": {"class": "guide", "index": 1.0, "sizes": [0.5, SPAN]},
    "reservoir": {
        "class": "voltage_reservoir", "sizes": [GAP, SPAN], "resolution": 4,
        "n_o": 1.52, "n_e": 1.71,
        "boundary_conditions": ["free", "free", "free"],
        "elastic_constants": {"K1": 11.1, "K2": 2.0, "K3": 17.1, "q0": 0.0},
        "eps_perp_dc": 5.0, "eps_a_dc": 10.0,
        "poisson_rtol": 1e-7, "poisson_maxiter": 20000,
        "domain_padding": {"enabled": True, "faces": ["y_min", "y_max"],
                            "n_pad": 20, "growth": 1.2, "w_max_um": 2.0,
                            "eps_outside": 1.0},
        "spline_electrode": {"enabled": True, "face": "x_min",
                              "coeffs": [1.0] * NCTRL, "span_um": None,
                              "degree": 3, "ground_face": "x_max"},
    },
    "guide_2": {"class": "guide", "index": 1.0, "sizes": [5.0, SPAN]},
    "source_1": {"class": "source", "position": {"on_object": "guide_1",
                 "position": "center", "size": [0.0, SPAN, 0.0]},
                 "amplitude": [1.0], "component": "Ey", "source_type": "pulsed",
                 "lam": 0.5, "dlam": 0.0, "pulse_fwhm_fs": 20.0,
                 "pulse_delay_fs": 0.0},
    # sensor at guide_2 RIGHT edge = 5 um from the reservoir exit
    "monitor_2": {"class": "monitor", "type": "1Ddft", "on_object": "guide_2",
                  "position": {"position": "right", "size": SPAN},
                  "lam_range": [0.5, 0.5], "n_lam": 1},
    "object_order": ["guide_1", "reservoir", "guide_2", "source_1", "monitor_2"],
}
os.makedirs(os.path.join(BASE, "simulation"), exist_ok=True)
json.dump(cfg, open(os.path.join(BASE, "simulation_data.json"), "w"), indent=2)

from class_voltage_reservoir import VoltageReservoir

_vr = VoltageReservoir(BASE)
_phi_warm = [None]

SCL_MAX, SCL_TOL, SCL_DEPTH = 12, 1e-4, 5


def _anderson_scl(coeffs, phi0):
    """Anderson-mixed self-consistent loop: director ⇌ E until fixed point.
    x_{k+1} = G(x_k) with G = LC_relax(E(Poisson(ε(x_k)))); Anderson type-I
    mixing over the flattened φ field (depth SCL_DEPTH)."""
    el = _vr.electrodes
    _vr.electrodes.set_voltages(spline=np.asarray(coeffs))
    phi = np.zeros(el.gshape) if phi0 is None else phi0
    X, F = [], []                      # iterates and residuals for Anderson
    theta = np.full(el.gshape, np.pi / 2)
    for it in range(SCL_MAX):
        V, E = _vr.poisson.solve(phi, theta)
        phi_new, theta = _vr.lc.compute(E, el.gshape, el.spacings,
                                        phi_init=phi, full_3d=False)
        r = (phi_new - phi).ravel()
        err = np.linalg.norm(r) / max(np.linalg.norm(phi_new.ravel()), 1e-20)
        if err < SCL_TOL:
            phi = phi_new
            break
        X.append(phi_new.ravel().copy()); F.append(r.copy())
        if len(X) > SCL_DEPTH:
            X.pop(0); F.pop(0)
        if len(F) >= 2:
            # Anderson type-I: minimise ||Σ a_i F_i||, Σ a_i = 1
            Fm = np.stack(F, axis=1)
            ones = np.ones(Fm.shape[1])
            G = Fm.T @ Fm + 1e-12 * np.eye(Fm.shape[1])
            a = np.linalg.solve(G, ones); a /= a.sum()
            phi = (np.stack(X, axis=1) @ a).reshape(el.gshape)
        else:
            phi = phi_new
    _vr.V, _vr.E, _vr.phi, _vr.theta = V, E, phi, theta
    return phi, theta, it + 1, err


def lc_from_coeffs(coeffs):
    phi, theta, n_it, err = _anderson_scl(coeffs, _phi_warm[0])
    print(f"[scl] converged in {n_it} outer iters (err {err:.2e})", flush=True)
    _phi_warm[0] = phi
    el = _vr.electrodes
    x = np.linspace(-el.sx / 2, el.sx / 2, el.nx)
    y = np.linspace(-el.sy / 2, el.sy / 2, el.ny)
    z = np.linspace(0, el.dz * (el.nz - 1), el.nz)
    np.savez(os.path.join(BASE, "simulation", "lc_fields.npz"),
             phi=phi, theta=theta, x=x, y=y, z=z)


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
    m = np.load(os.path.join(BASE, "simulation", "monitor_2.npz"))
    Ey = m["Ey"][0]
    return np.abs(Ey) ** 2


_history = []


def cost(coeffs, grad=None):
    t0 = time.time()
    lc_from_coeffs(coeffs)
    I = forward()
    y = np.linspace(-SPAN / 2, SPAN / 2, len(I))
    frac = I[np.abs(y) <= CENTER_BAND].sum() / max(I.sum(), 1e-300)
    c = 1.0 - frac
    _history.append((list(map(float, coeffs)), float(c), float(frac)))
    np.savez(os.path.join(BASE, "opt_history.npz"),
             coeffs=np.array([h[0] for h in _history]),
             cost=np.array([h[1] for h in _history]),
             frac=np.array([h[2] for h in _history]),
             best_I=I, y=y)
    print(f"[opt] eval {len(_history)}: frac_center={frac:.4f} "
          f"cost={c:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return c


def main():
    try:
        import nlopt
        opt = nlopt.opt(nlopt.LN_BOBYQA, NCTRL)
        opt.set_lower_bounds([0.0] * NCTRL)
        opt.set_upper_bounds([VMAX] * NCTRL)
        opt.set_min_objective(cost)
        opt.set_maxeval(int(os.environ.get("OPT_MAXEVAL", "120")))
        x0 = np.full(NCTRL, 1.5)
        hist = os.path.join(BASE, "opt_history.npz")
        if os.path.exists(hist):
            h = np.load(hist)
            k = int(np.argmin(h["cost"]))
            x0 = np.asarray(h["coeffs"][k], dtype=np.float64)
            print(f"[opt] resuming from best of {len(h['cost'])} prior evals "
                  f"(frac={h['frac'][k]:.4f})", flush=True)
        try:
            xbest = opt.optimize(x0)
        except KeyboardInterrupt:
            xbest = min(_history, key=lambda h: h[1])[0]
    except ImportError:
        from scipy.optimize import minimize
        r = minimize(cost, np.full(NCTRL, 1.5), method="Powell",
                     bounds=[(0.0, VMAX)] * NCTRL,
                     options={"maxfev": int(os.environ.get("OPT_MAXEVAL", "120"))})
        xbest = r.x
    best = min(_history, key=lambda h: h[1])
    print(f"[opt] DONE best frac_center={best[2]:.4f} coeffs={best[0]}")


if __name__ == "__main__":
    raise SystemExit(main())
