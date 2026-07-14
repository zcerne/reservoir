"""Rigorous MEEP-vs-GPUmeep ladder comparison (per-config).

Sensor placed off the PML edge (center of output guide) — the PML boundary
inflates MEEP's DFT field spuriously (known cross-solver readout pitfall).
"""
import os, sys, json, shutil, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

RES = 40
PML = 1.5
INT_Y = 6.0
CELL_Y = INT_Y + 2 * PML          # 9.0
G1, RESV, G2 = 0.5, 5.0, 5.0      # x-lengths
LAM_SIG = 0.5
PULSE_FWHM_FS = 20.0
N_O, N_E = 1.52, 1.71
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "ladder")

CONFIGS = {
    1: dict(lc=False, dye=False, mirrors=False, name="air"),
    # 1.2: same as 1 but periodic BCs on ALL sides, no PML anywhere.
    # Isolates the core Yee update from the PML implementation.
    1.2: dict(lc=False, dye=False, mirrors=False, periodic=True, name="air_periodic"),
    2: dict(lc=True,  dye=False, mirrors=False, name="LC"),
    3: dict(lc=True,  dye=True,  mirrors=False, name="LC+dye"),
    4: dict(lc=False, dye=False, mirrors=True,  name="mirrors+air"),
    5: dict(lc=True,  dye=False, mirrors=True,  name="mirrors+LC"),
    6: dict(lc=True,  dye=True,  mirrors=True,  name="mirrors+LC+dye"),
}


def build_json(n):
    c = CONFIGS[n]
    # LADDER_DIM=3 -> reduced-size 3D variant of the same 6 configs
    dim = int(os.environ.get("LADDER_DIM", "2"))
    if dim == 3:
        res = int(os.environ.get("LADDER_RES", 20))
        pml = float(os.environ.get("LADDER_PML", 1.0))
        int_y = float(os.environ.get("LADDER_INT_Y", 3.0))
        int_z = float(os.environ.get("LADDER_INT_Z", 1.5))
        g1 = float(os.environ.get("LADDER_G1", 0.5))
        resv_x = float(os.environ.get("LADDER_RESV", 2.5))
        g2 = float(os.environ.get("LADDER_G2", 2.5))
        run_until = int(os.environ.get("LADDER_RUN_UNTIL", "60"))
    else:
        res = int(os.environ.get("LADDER_RES", RES))
        pml = float(os.environ.get("LADDER_PML", PML))
        int_y = INT_Y; int_z = 0.0
        g1, resv_x, g2 = G1, RESV, G2
        run_until = int(os.environ.get("LADDER_RUN_UNTIL", "120"))
    lam_sig = float(os.environ.get("LADDER_SIG_LAM", LAM_SIG))
    periodic = bool(c.get("periodic", False))
    if periodic:
        pml = 0.0
    cell_y = int_y + 2 * pml
    order = ["guide_1"]
    d = {
        "resolution": res, "use_cw": False,
        "run_until": run_until,
        "dimention": dim, "cell_size_y": cell_y, "periodic": periodic,
        "pml_size": pml, "background_index": 1.0,
        # snapshots disabled (window set beyond run_until)
        "snapshot_t1": 1e9, "snapshot_t2": 1e9, "snapshot_dt": 1.0,
    }
    if dim == 3:
        d["cell_size_z"] = int_z + 2 * pml
    d["guide_1"] = {"class": "guide", "index": 1.0, "sizes": [g1, int_y]}
    if c["mirrors"]:
        d["mirror_1"] = {"class": "mirror", "lam": 0.55, "n_indexes": [1.46, 2.4],
                         "transmission": 0.1, "size_y": int_y}
        order.append("mirror_1")
    # reservoir slot: LC reservoir object, or an air guide
    if c["lc"]:
        resv_sizes = [resv_x, int_y] if dim == 2 else [resv_x, int_y, int_z]
        resv = {"class": "reservoir", "sizes": resv_sizes, "resolution": 10,
                "boundary_conditions": ["free", "free", "free"],
                "face_phi": [None]*6, "face_theta": [None]*6,
                "elastic_constants": {"K1": 11.1, "K2": 2.0, "K3": 17.1, "q0": 0.0},
                "n_o": N_O, "n_e": N_E, "S": 1.0, "maxeval": 5000, "f_tolerance": 1e-6,
                "optimize_phi_theta": [True, False],
                "boundary_function": ("sinus_random_2d" if dim == 2 else "sinus_3d"),
                "boundary_n_periods": 3,
                "boundary_phase_shift": 3.14159, "boundary_noise_level": 0.6,
                "boundary_scale": 15.0, "boundary_seed": 7,
                "lc_param": "Q3D", "S_eq": 0.8}
        if c["dye"]:
            n3 = float(os.environ.get("LADDER_N3", "25.0"))     # inversion level (low → linear gain)
            resv["sted"] = {"enabled": True, "lbdA": 0.53, "gammaA": 0.13,
                            "lbdE": 0.55, "gammaE": 0.13, "SGMA": 0.006,
                            "N1_0": 0.0, "N3_0": n3,            # PRE-INVERTED
                            "rate_43": 10.0, "rate_21": 100.0}
        d["reservoir"] = resv
        order.append("reservoir")
    else:
        d["guide_res"] = {"class": "guide", "index": 1.0, "sizes": [resv_x, int_y]}
        order.append("guide_res")
    if c["mirrors"]:
        d["mirror_2"] = {"class": "mirror", "lam": 0.55, "n_indexes": [1.46, 2.4],
                         "transmission": 0.1, "size_y": int_y}
        order.append("mirror_2")
    d["guide_2"] = {"class": "guide", "index": 1.0, "sizes": [g2, int_y]}
    order.append("guide_2")

    # signal source at guide_1 center (Ey, pulsed λ=0.5, plane)
    src_sy = float(os.environ.get("LADDER_SRC_SY", int_y))   # <int_y pulls src off PML
    d["source_1"] = {"class": "source", "position": {"on_object": "guide_1",
                     "position": "center", "size": [0.0, src_sy, int_z]},
                     "amplitude": [1.0], "component": "Ey", "source_type": "pulsed",
                     "lam": lam_sig, "dlam": 0.0, "pulse_fwhm_fs": PULSE_FWHM_FS,
                     "pulse_delay_fs": 0.0}
    order.append("source_1")
    # STED pump (Ez area over reservoir) for doped configs
    pump_amp = float(os.environ.get("LADDER_PUMP_AMP", "300.0"))
    if c["dye"] and pump_amp > 0:
        d["source_2"] = {"class": "source", "position": {"on_object": "reservoir",
                         "position": "center", "size": [resv_x, int_y, int_z]},
                         "amplitude": [pump_amp], "component": "Ez", "source_type": "pulsed",
                         "lam": 0.53, "dlam": 0.0, "pulse_fwhm_fs": 200.0,
                         "pulse_delay_fs": 0.0}
        order.append("source_2")
    # sensor: DFT complex Ey(y) at the END (right) of guide_2
    mon_obj = os.environ.get("LADDER_MON_OBJ", "guide_2")   # near-src test: guide_1
    n_lam = int(os.environ.get("LADDER_NLAM", "1"))
    lam_lo = float(os.environ.get("LADDER_LAM_LO", lam_sig))
    lam_hi = float(os.environ.get("LADDER_LAM_HI", lam_sig))
    d["monitor_2"] = {"class": "monitor",
                      "type": ("1Ddft" if dim == 2 else "2Ddft"),
                      "on_object": mon_obj,
                      "position": {"position": "center",  # off PML edge (HW bug fixed)
                                   "size": int_y},
                      "lam_range": [lam_lo, lam_hi], "n_lam": n_lam}
    order.append("monitor_2")
    if os.environ.get("LADDER_FLUX"):
        d["monitor_3"] = {"class": "monitor", "type": "flux", "on_object": mon_obj,
                          "position": {"position": "center", "size": int_y},
                          "lam_range": [lam_lo, lam_hi], "n_lam": n_lam}
        order.append("monitor_3")
    d["object_order"] = order

    suffix = "" if dim == 2 else "_3d"
    path = os.path.join(BASE, f"config_{n}_{c['name'].replace('+','_')}{suffix}")
    os.makedirs(os.path.join(path, "simulation"), exist_ok=True)
    with open(os.path.join(path, "simulation_data.json"), "w") as f:
        json.dump(d, f, indent=2)
    return path


def ensure_lc(path):
    """Relax the LC director for a reservoir config (writes simulation/lc_fields.npz)."""
    lc = os.path.join(path, "simulation", "lc_fields.npz")
    if os.path.exists(lc):
        return
    from class_reservoir import Reservoir
    r = Reservoir(path)
    r.run_minimization()
    r.save_fields()


def run_meep(path):
    from class_simulation import Simulation
    sim = Simulation(path)
    sim.run_simulation()
    return _load_sensor(os.path.join(path, "simulation", "monitor_2.npz"))


def run_gpumeep(path):
    import jax, importlib
    jax.config.update("jax_enable_x64", True)
    # Import the CANONICAL gpumeep driver from GPUmeep/src (source of truth;
    # GPUMEEP_PATH points at it). BlockOptimization has a different, older copy
    # the LC-relax import chain can pull into sys.modules — force the canonical.
    gpu_src = os.environ.get("GPUMEEP_PATH") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "GPUmeep", "src")
    sys.path.insert(0, gpu_src)
    sys.modules.pop("class_simulation_gpu", None)
    csg = importlib.import_module("class_simulation_gpu")
    assert os.path.dirname(csg.__file__) == gpu_src, f"wrong module: {csg.__file__}"
    sim = csg.SimulationGPU(folder_path=path)
    sim.force_fullvector = True
    sim.run()
    return _load_sensor(os.path.join(path, "simulation", "monitor_2.npz"))


def _load_sensor(p):
    d = np.load(p)
    Ey = np.asarray(d["Ey"])
    return Ey.reshape(-1) if Ey.ndim == 1 else Ey[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=float, required=True)
    ap.add_argument("--engine", choices=["meep", "gpumeep", "both"], default="both")
    args = ap.parse_args()
    # integral configs stay int so existing paths (config_1_air, …) are unchanged
    cfg = int(args.config) if args.config.is_integer() else args.config
    path = build_json(cfg)
    print(f"config {cfg} ({CONFIGS[cfg]['name']}) → {path}")
    if CONFIGS[cfg]["lc"]:
        ensure_lc(path)
    sim_dir = os.path.join(path, "simulation")

    if args.engine in ("meep", "both"):
        ey = run_meep(path)
        shutil.copy(os.path.join(sim_dir, "monitor_2.npz"),
                    os.path.join(sim_dir, "monitor_2_meep.npz"))
        if os.path.exists(os.path.join(sim_dir, "monitor_3.npz")):
            shutil.copy(os.path.join(sim_dir, "monitor_3.npz"),
                        os.path.join(sim_dir, "monitor_3_meep.npz"))
        print(f"MEEP sensor: len={len(ey)} |Ey| max={np.abs(ey).max():.4g} mean={np.abs(ey).mean():.4g}")
    if args.engine in ("gpumeep", "both"):
        ey = run_gpumeep(path)
        shutil.copy(os.path.join(sim_dir, "monitor_2.npz"),
                    os.path.join(sim_dir, "monitor_2_gpumeep.npz"))
        if os.path.exists(os.path.join(sim_dir, "monitor_3.npz")):
            shutil.copy(os.path.join(sim_dir, "monitor_3.npz"),
                        os.path.join(sim_dir, "monitor_3_gpumeep.npz"))
        print(f"gpumeep sensor: len={len(ey)} |Ey| max={np.abs(ey).max():.4g} mean={np.abs(ey).mean():.4g}")


if __name__ == "__main__":
    raise SystemExit(main())
