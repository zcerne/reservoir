"""2-electrode spline design, FAR-FIELD target (geometry 1: angular Gaussian
at theta=0). Same cell as the 0.306 champion (spline V(y) on x_min, ground
x_max, free LC BCs, 12x24 um gap); the near field is captured on a 2Ddft
strip in the middle of the 5 um exit air guide (2.5 um clear of the PML) and
projected with GPUmeep's MEEP-exact near2far (green2d Hankel transform) to a
semicircle at R = 2 mm.

Costs (env FF_COST):
  fraction : 1 - sum(I*G)/sum(I),  G = Gaussian in theta, sigma = SIGMA_DEG
             (far-field analogue of the champion cost)          [smaug1]
  overlap  : -|sum_theta Ey_far * t(theta)|^2, t = REAL Gaussian amplitude
             (phase-aware + absolute/unnormalized), with sigma ANNEALED
             12 -> 8 -> 5 -> 3 deg in equal eval blocks          [smaug2]
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FF_COST = os.environ.get("FF_COST", "fraction")
TAG = os.environ.get("TAG", f"ff_{FF_COST[:4]}")
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "twoelectrode", TAG)
GAP, SPAN, NCTRL = 12.0, 24.0, 12
VMAX = float(os.environ.get("VMAX", "3.5"))
SIGMA_DEG = float(os.environ.get("SIGMA_DEG", "3.0"))
ANNEAL = [12.0, 8.0, 5.0, 3.0]
R_FAR, N_TH, TH_MAX = 2000.0, 361, np.deg2rad(45.0)
# FF_ANCHOR=soft → Rapini-Papoular soft anchoring phi0=pi/2 on the y walls,
# strength W [pN/µm] from FF_W (default K1/1µm — extrapolation length 1 µm).
FF_ANCHOR = os.environ.get("FF_ANCHOR", "free")
FF_W = float(os.environ.get("FF_W", "11.1"))

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
    # near-field strip at guide_2 CENTER (2.5 um clear of reservoir and PML)
    "monitor_2": {"class": "monitor", "type": "2Ddft", "on_object": "guide_2",
                  "position": {"position": "center", "size": [0.2, SPAN]},
                  "lam_range": [0.5, 0.5], "n_lam": 1},
    "object_order": ["guide_1", "reservoir", "guide_2", "source_1", "monitor_2"],
}
if FF_ANCHOR == "soft":
    cfg["reservoir"]["lc_mode"] = "fwdbwd_relax"
    cfg["reservoir"]["lc_relax_maxeval"] = 600
    cfg["reservoir"]["face_phi"] = [None, None, 1.5707963, 1.5707963, None, None]
    cfg["reservoir"]["face_anchor_W"] = [None, None, FF_W, FF_W, None, None]
os.makedirs(os.path.join(BASE, "simulation"), exist_ok=True)
json.dump(cfg, open(os.path.join(BASE, "simulation_data.json"), "w"), indent=2)

from class_voltage_reservoir import VoltageReservoir

_vr = VoltageReservoir(BASE)
_phi_warm = [None]
SCL_MAX, SCL_TOL, SCL_DEPTH = 12, 1e-4, 5


def _anderson_scl(coeffs, phi0):
    el = _vr.electrodes
    el.set_voltages(spline=np.asarray(coeffs))
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


# ---------------- far-field projection (precomputed transfer matrices) ------
DX = 1.0 / cfg["resolution"]
CELL_X = 1.5 + 0.5 + GAP + 5.0 + 1.5
CX, CY = CELL_X / 2.0, cfg["cell_size_y"] / 2.0
X_LINE = -CX + 1.5 + 0.5 + GAP + 2.5          # guide_2 center (MEEP coords)
X_EXIT = -CX + 1.5 + 0.5 + GAP                 # reservoir exit
THETAS = np.linspace(-TH_MAX, TH_MAX, N_TH)
FAR_PTS = np.stack([X_EXIT + R_FAR * np.cos(THETAS),
                    R_FAR * np.sin(THETAS)], axis=1)
_TMAT = [None]        # (T_E, T_H, i_col, ys) built after the first forward


def _far_field(mpath):
    """Load the 2Ddft strip npz -> project -> (Ex_far, Ey_far, Hz_far)."""
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
        T_E, T_H = n2f.transfer_matrices_line_x(
            FAR_PTS, i_line * DX - CX, ys, DX, DX, 1.0 / cfg["monitor_2"]["lam_range"][0])
        _TMAT[0] = (T_E[:, [0, 1, 5]], T_H[:, [0, 1, 5]], k)
        print(f"[n2f] strip {Ey2.shape}, line col {k}, {N_TH} far pts", flush=True)
    T_E, T_H, k = _TMAT[0]
    EH = n2f.farfield_line_x(T_E, T_H, Ey2[k], Hz2[k - 1], Hz2[k])
    return EH[:, 0], EH[:, 1], EH[:, 2]


def evaluate(coeffs, sigma_deg):
    phi, theta = _anderson_scl(np.asarray(coeffs), _phi_warm[0])
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
    Ex_f, Ey_f, Hz_f = _far_field(os.path.join(BASE, "simulation", "monitor_2.npz"))
    I = np.abs(Ex_f) ** 2 + np.abs(Ey_f) ** 2
    sig = np.deg2rad(sigma_deg)
    G = np.exp(-THETAS ** 2 / (2.0 * sig ** 2))
    frac = float((I * G).sum() / max(I.sum(), 1e-300))
    ovl = float(np.abs((Ey_f * G).sum()) ** 2)
    return frac, ovl, I, Ey_f


def main():
    import nlopt
    maxeval = int(os.environ.get("OPT_MAXEVAL", "160"))
    _hist = []

    def run_block(x0, sigma_deg, n_ev):
        opt = nlopt.opt(nlopt.LN_BOBYQA, NCTRL)
        opt.set_lower_bounds([0.0] * NCTRL)
        opt.set_upper_bounds([VMAX] * NCTRL)
        opt.set_maxeval(n_ev)

        def cost(v, grad=None):
            t0 = time.time()
            frac, ovl, I, Ey_f = evaluate(v, sigma_deg)
            c = (1.0 - frac) if FF_COST == "fraction" else -ovl
            _hist.append((list(map(float, v)), frac, ovl, sigma_deg))
            np.savez(os.path.join(BASE, "opt_history.npz"),
                     volts=np.array([h[0] for h in _hist]),
                     frac=np.array([h[1] for h in _hist]),
                     ovl=np.array([h[2] for h in _hist]),
                     sig=np.array([h[3] for h in _hist]),
                     best_I=I, best_Ey=Ey_f, thetas=THETAS)
            print(f"[{TAG}] eval {len(_hist)} (sig={sigma_deg:g}): "
                  f"frac={frac:.4f} ovl={ovl:.4g} ({time.time()-t0:.0f}s)",
                  flush=True)
            return c

        opt.set_min_objective(cost)
        try:
            return np.asarray(opt.optimize(np.asarray(x0, dtype=np.float64)))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if not _hist:
                raise           # first eval failed — a real bug, surface it
            print(f"[{TAG}] block ended: {e}", flush=True)
            return np.asarray(_hist[-1][0])

    x0 = np.full(NCTRL, 1.0)
    hist = os.path.join(BASE, "opt_history.npz")
    if os.path.exists(hist):
        h = np.load(hist)
        key = h["frac"] if FF_COST == "fraction" else h["ovl"]
        x0 = np.asarray(h["volts"][int(np.argmax(key))], dtype=np.float64)
        print(f"[{TAG}] resuming from best of {len(key)} evals", flush=True)

    if FF_COST == "overlap" and os.environ.get("FF_ANNEAL", "1") == "1":
        for sig in ANNEAL:
            x0 = run_block(x0, sig, maxeval // len(ANNEAL))
    else:
        x0 = run_block(x0, SIGMA_DEG, maxeval)

    key = [h[1] for h in _hist] if FF_COST == "fraction" else [h[2] for h in _hist]
    b = int(np.argmax(key))
    print(f"[{TAG}] DONE best frac={_hist[b][1]:.4f} ovl={_hist[b][2]:.4g} "
          f"coeffs={np.round(_hist[b][0], 2).tolist()}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
