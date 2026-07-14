"""Discrete electrode-position optimization (no splines): the left-wall
electrode is ONE CONNECTED strip of metal at V0 — a contiguous run of the
12 2-um slots (start s, width w); everything else is floating wall. Full
ground plane on the right wall. EXHAUSTIVE over all 78 contiguous blocks
(includes the 6 symmetric centered ones). Gaussian center-focus target.

Variants (env DESIGN):
  free     : free LC BCs, V0 = 3.0 V           (design A-discrete)
  anchored : phi=pi/2 pinned on y walls, V0 recalculated from the anchored
             Freedericksz threshold: anchoring spans SPAN=24 um, field acts
             over GAP=12 um -> V_th = GAP*(pi/SPAN)*sqrt(K1/(eps0*deps)) =
             0.557 V; V0 = 1.5 V = 2.7x V_th   (design B-discrete)
"""
import os, sys, json, time, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DESIGN = os.environ.get("DESIGN", "free")
GAP, SPAN, NSEG = 12.0, 24.0, 12
V0 = float(os.environ.get("V0", 3.0 if DESIGN == "free" else 1.5))
TARGET_SIGMA = 1.5
_tag = os.environ.get("TAG", DESIGN)
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "twoelectrode", f"discrete_{_tag}")

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
        "electrode_width_um": SPAN / NSEG,     # contiguous 2-um slots
        "voltages_x_min": [V0] * NSEG,          # overridden per config
        "voltages_x_max": [0.0] * NSEG,         # ground plane (full coverage)
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
if DESIGN == "anchored":
    cfg["reservoir"]["lc_mode"] = "fwdbwd_relax"
    cfg["reservoir"]["lc_relax_maxeval"] = 600
    cfg["reservoir"]["face_phi"] = [None, None, 1.5707963, 1.5707963, None, None]
os.makedirs(os.path.join(BASE, "simulation"), exist_ok=True)
json.dump(cfg, open(os.path.join(BASE, "simulation_data.json"), "w"), indent=2)

from class_voltage_reservoir import VoltageReservoir

_vr = VoltageReservoir(BASE)
_phi_warm = [None]
SCL_MAX, SCL_TOL, SCL_DEPTH = 12, 1e-4, 5


def _anderson_scl(vxmin, phi0):
    el = _vr.electrodes
    el.set_voltages(x_min=vxmin, x_max=np.zeros(NSEG))
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


def evaluate(bits):
    vx = np.where(np.asarray(bits) > 0, V0, np.nan)
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
    hist = []
    out = os.path.join(BASE, "discrete_history.npz")
    done = set()
    if os.path.exists(out):
        h = np.load(out)
        for b, f in zip(h["bits"], h["frac"]):
            done.add(tuple(int(x) for x in b)); hist.append((list(b), float(f)))
        print(f"[disc] resuming: {len(done)} configs done", flush=True)
    best_I = None; y = None
    # all CONNECTED electrode strips: contiguous slots [s, s+w)
    configs = [(s, w) for w in range(1, NSEG + 1) for s in range(NSEG - w + 1)]
    for (s, w) in configs:
        bits = tuple(1 if s <= k < s + w else 0 for k in range(NSEG))
        if bits in done:
            continue
        t0 = time.time()
        frac, I, y = evaluate(np.array(bits))
        hist.append((list(bits), frac))
        if best_I is None or frac >= max(h[1] for h in hist):
            best_I = I
        np.savez(out, bits=np.array([h[0] for h in hist]),
                 frac=np.array([h[1] for h in hist]),
                 best_I=best_I if best_I is not None else np.zeros(1),
                 y=y if y is not None else np.zeros(1))
        print(f"[disc] strip s={s} w={w} -> frac {frac:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    best = max(hist, key=lambda h: h[1])
    bb = np.asarray(best[0])
    sym = "symmetric-centered" if np.all(bb == bb[::-1]) else "off-center"
    print(f"[disc] DONE {DESIGN}: best frac={best[1]:.4f} strip={best[0]} "
          f"({sym}, V0={V0} V)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
