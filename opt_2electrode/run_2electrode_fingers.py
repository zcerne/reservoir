"""IPS/FFS finger-comb optimization: 8 IDENTICAL fingers (1.5 um wide,
3 um pitch) on the left wall, full ground plane on the right wall, LC free
BCs. Optimization dof = the 8 per-finger voltages in [-VMAX, +VMAX] —
adjacent-finger differences make the field ARC (angle encoding), the mean
sets the through-field bias. Gaussian center-focus target. BOBYQA, resume.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GAP, SPAN, NFING = 12.0, 24.0, 8
VMAX = 3.5
TARGET_SIGMA = 1.5
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "twoelectrode", "fingers8")

cfg = {
    "resolution": 40, "use_cw": False, "run_until": 60,
    "dimention": 2, "cell_size_y": SPAN + 3.0, "periodic": False,
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
        "electrode_width_um": 1.5,                # finger width
        "electrode_widths": {"x_max": SPAN / NFING},   # ground = full coverage
        "voltages_x_min": [0.0] * NFING,
        "voltages_x_max": [0.0] * NFING,
        "domain_padding": {"enabled": True, "faces": ["y_min", "y_max"],
                            "n_pad": 20, "growth": 1.2, "w_max_um": 2.0,
                            "eps_outside": 1.0},
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

from class_voltage_reservoir import VoltageReservoir

_vr = VoltageReservoir(BASE)
_phi_warm = [None]
SCL_MAX, SCL_TOL, SCL_DEPTH = 12, 1e-4, 5


def _anderson_scl(vxmin, phi0):
    el = _vr.electrodes
    el.set_voltages(x_min=vxmin, x_max=np.zeros(NFING))
    phi = np.zeros(el.gshape) if phi0 is None else phi0
    theta = np.full(el.gshape, np.pi / 2)
    X, F = [], []
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
            Fm = np.stack(F, axis=1)
            G = Fm.T @ Fm + 1e-12 * np.eye(Fm.shape[1])
            a = np.linalg.solve(G, np.ones(Fm.shape[1])); a /= a.sum()
            phi = (np.stack(X, axis=1) @ a).reshape(el.gshape)
        else:
            phi = phi_new
    _vr.phi, _vr.theta = phi, theta
    return phi, theta


def evaluate(volts):
    vx = np.asarray(volts, dtype=np.float64)
    phi, theta = _anderson_scl(vx, _phi_warm[0])
    _phi_warm[0] = phi
    el = _vr.electrodes
    np.savez(os.path.join(BASE, "simulation", "lc_fields.npz"),
             phi=phi, theta=theta,
             x=np.linspace(-el.sx/2, el.sx/2, el.nx),
             y=np.linspace(-el.sy/2, el.sy/2, el.ny),
             z=np.linspace(0, el.dz*(el.nz-1), el.nz))
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


def main():
    import nlopt
    opt = nlopt.opt(nlopt.LN_BOBYQA, NFING)
    opt.set_lower_bounds([-VMAX] * NFING)
    opt.set_upper_bounds([VMAX] * NFING)
    _hist = []

    def cost(v, grad=None):
        t0 = time.time()
        frac, I, y = evaluate(v)
        _hist.append((list(map(float, v)), frac))
        np.savez(os.path.join(BASE, "opt_history.npz"),
                 volts=np.array([h[0] for h in _hist]),
                 frac=np.array([h[1] for h in _hist]), best_I=I, y=y)
        print(f"[fing] eval {len(_hist)}: frac={frac:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return 1.0 - frac

    opt.set_min_objective(cost)
    opt.set_maxeval(int(os.environ.get("OPT_MAXEVAL", "150")))
    # alternating-polarity comb start (the IPS baseline)
    x0 = np.array([2.0 * (-1) ** k for k in range(NFING)])
    hist = os.path.join(BASE, "opt_history.npz")
    if os.path.exists(hist):
        h = np.load(hist)
        k = int(np.argmax(h["frac"]))
        x0 = np.asarray(h["volts"][k], dtype=np.float64)
        print(f"[fing] resuming from best of {len(h['frac'])} evals", flush=True)
    try:
        opt.optimize(x0)
    except KeyboardInterrupt:
        pass
    best = max(_hist, key=lambda h: h[1])
    print(f"[fing] DONE best frac={best[1]:.4f} volts={np.round(best[0],2).tolist()}",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
