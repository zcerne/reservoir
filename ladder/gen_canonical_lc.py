"""Generate ONE canonical LC design: Perlin-noise BCs + Q-tensor relaxation.

Relaxes a single reservoir with perlin_2d boundaries via the Landau–de Gennes
Q-tensor solver, saves φ/θ + the raw Q field (q5) to lc_fields.npz, and renders
the director + S field for approval. Once approved, this exact field is reused
across all ladder configs (both engines).

  python ladder/gen_canonical_lc.py            # relax + save + plot
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

RESV_X, RESV_Y = 5.0, 5.0
LC_RES = 10           # same as reservoir (15_2D_sted_resonator)
SEED = 7
SCALE = 5.0            # Perlin smoothness (µm) — ~1 feature across a 5µm reservoir → smooth
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "ladder", "canonical_lc")


def build():
    d = {
        "resolution": 40, "use_cw": False, "run_until": 100, "dimention": 2,
        "cell_size_y": RESV_Y + 3.0, "periodic": False, "pml_size": 1.5,
        "background_index": 1.0, "object_order": ["reservoir"],
        "reservoir": {
            # EXACT reservoir params (15_2D_sted_resonator), θ NOT optimized (in-plane),
            # only the BC changed to Perlin noise.
            "class": "reservoir", "sizes": [RESV_X, RESV_Y], "resolution": LC_RES,
            "boundary_conditions": ["free", "free", "free"],
            "face_phi": [None] * 6, "face_theta": [None] * 6,
            "elastic_constants": {"K1": 11.1, "K2": 2.0, "K3": 17.1, "q0": 0.0},
            "n_o": 1.52, "n_e": 1.71, "S": 1.0, "maxeval": 5000, "f_tolerance": 1e-6,
            "optimize_phi_theta": [True, False],
            "boundary_function": "perlin_2d", "boundary_scale": SCALE,
            "boundary_seed": SEED, "lc_param": "Q3D", "S_eq": 0.8,
        },
    }
    os.makedirs(os.path.join(BASE, "simulation"), exist_ok=True)
    with open(os.path.join(BASE, "simulation_data.json"), "w") as f:
        json.dump(d, f, indent=2)
    return BASE


def main():
    path = build()
    from class_reservoir import Reservoir
    r = Reservoir(path)
    r.run_minimization()
    r.save_fields()
    lc = np.load(os.path.join(path, "simulation", "lc_fields.npz"))
    print("saved keys:", list(lc.keys()))
    mid = lc["phi"].shape[2] // 2
    phi = np.asarray(lc["phi"])[:, :, mid]; theta = np.asarray(lc["theta"])[:, :, mid]
    has_Q = "Qxx" in lc
    print(f"phi range [{np.degrees(phi.min()):.0f},{np.degrees(phi.max()):.0f}]deg, "
          f"theta [{np.degrees(theta.min()):.0f},{np.degrees(theta.max()):.0f}]deg, Q saved: {has_Q}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.asarray(lc["x"]); y = np.asarray(lc["y"])
    ncol = 4 if has_Q else 3
    fig, ax = plt.subplots(1, ncol, figsize=(5 * ncol, 4.3))
    im0 = ax[0].imshow(phi.T, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
                       cmap="twilight", aspect="auto")
    ax[0].set_title("φ azimuth (rad)"); plt.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(np.degrees(theta).T, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
                       cmap="viridis", aspect="auto")
    ax[1].set_title("θ polar (deg, 90=in-plane)"); plt.colorbar(im1, ax=ax[1])
    nx = np.sin(theta) * np.cos(phi); ny = np.sin(theta) * np.sin(phi); sk = 2
    ax[2].quiver(x[::sk][:, None] + 0 * y[::sk][None, :], 0 * x[::sk][:, None] + y[::sk][None, :],
                 nx[::sk, ::sk].T, ny[::sk, ::sk].T, pivot="mid", headwidth=1, headlength=0, scale=25)
    ax[2].set_title("in-plane director"); ax[2].set_aspect("auto")
    if has_Q:
        Qxx = np.asarray(lc["Qxx"])[:, :, mid]; Qyy = np.asarray(lc["Qyy"])[:, :, mid]
        Qzz = -(Qxx + Qyy)
        S = 1.5 * np.maximum.reduce([np.abs(Qxx), np.abs(Qyy), np.abs(Qzz)])   # rough scalar order
        im3 = ax[3].imshow(S.T, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
                           cmap="magma", aspect="auto")
        ax[3].set_title("~S (order param, from Q)"); plt.colorbar(im3, ax=ax[3])
    fig.suptitle(f"Canonical LC — Perlin BCs (seed {SEED}, scale {SCALE}µm), Q-tensor relax", fontsize=13)
    plt.tight_layout(); plt.savefig("/tmp/canonical_lc.png", dpi=120)
    print("saved /tmp/canonical_lc.png")


if __name__ == "__main__":
    main()
