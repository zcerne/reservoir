"""Confirm director-relax and Q-tensor-relax give the SAME linear reservoir.

Loads the two T-matrices (G) built from identical 2D reservoirs (same random BCs)
relaxed two ways, and checks:
  1. both are LINEAR — probe each G with random complex inputs, fit a BLA, 1-R^2 ~ 0.
  2. SAME abilities — compare singular-value spectra, n_eff, and G_director vs G_Q
     directly (relative Frobenius difference + per-element correlation).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a_best_linear_approx import best_linear_approx

RES = next((p for p in ("/home/cernez/resevoir/data",
                        os.path.expanduser("~/Orion/resevoir/data"),
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
            if os.path.isdir(p)), os.path.expanduser("~/Orion/resevoir/data"))


def load_T(tag):
    p = os.path.join(RES, f"cmp_{tag}_2d", "simulation_T", "T_matrix.npz")
    return np.load(p)["T_Ey"].astype(complex)              # (N_y, N_in)


def linear_check(G, n_probe=200, seed=0):
    """probe G with random complex inputs -> BLA -> recovered G' and 1-R^2."""
    rng = np.random.default_rng(seed)
    nin = G.shape[1]
    Ein = rng.normal(size=(n_probe, nin)) + 1j * rng.normal(size=(n_probe, nin))
    Eout = (G @ Ein.T).T
    res = best_linear_approx({"inputs": Ein, "outputs": Eout}, test_frac=0.3)
    recov_err = np.linalg.norm(res["G"] - G) / (np.linalg.norm(G) + 1e-30)
    return res, recov_err


def main():
    Gd, Gq = load_T("director"), load_T("qtensor")
    print(f"G_director {Gd.shape}   G_qtensor {Gq.shape}")

    for tag, G in [("director", Gd), ("qtensor", Gq)]:
        res, rec = linear_check(G)
        s = np.linalg.svd(G, compute_uv=False)
        print(f"\n[{tag}]  n_eff={res['n_eff']:.3f}  significant={res['n_significant']}  "
              f"cond={res['cond']:.2g}")
        print(f"         singular values: {np.array2string(s, precision=3)}")
        print(f"         LINEAR? 1-R^2={res['residual_fraction']:.2e}  "
              f"(BLA recovers G to {rec:.2e} rel.err)  -> {'LINEAR' if res['residual_fraction']<1e-6 else 'nonlinear'}")

    # --- equivalence of the two operators ---
    # remove global complex gauge (overall phase/scale) before comparing
    scale = np.vdot(Gd, Gq) / (np.vdot(Gd, Gd) + 1e-30)     # best complex fit Gq ~ scale*Gd
    rel = np.linalg.norm(Gq - scale * Gd) / (np.linalg.norm(Gq) + 1e-30)
    corr = np.abs(np.vdot(Gd, Gq)) / (np.linalg.norm(Gd) * np.linalg.norm(Gq) + 1e-30)
    sd, sq = np.linalg.svd(Gd, compute_uv=False), np.linalg.svd(Gq, compute_uv=False)
    spec_rel = np.linalg.norm(sd - sq) / (np.linalg.norm(sd) + 1e-30)
    print("\n=== director vs Q-tensor EQUIVALENCE ===")
    print(f"  complex correlation |<Gd,Gq>|/(|Gd||Gq|) = {corr:.4f}  (1 = identical up to gauge)")
    print(f"  relative difference after gauge-match     = {rel:.4f}")
    print(f"  singular-value-spectrum rel. difference   = {spec_rel:.4f}")
    print(f"  n_eff: director {(sd**2).sum()**2/(sd**4).sum():.3f}  "
          f"qtensor {(sq**2).sum()**2/(sq**4).sum():.3f}")
    verdict = "SAME reservoir (Q3D ≈ director)" if corr > 0.98 and spec_rel < 0.05 else "DIFFER"
    print(f"  -> {verdict}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(sd, "o-", label="director"); ax[0].plot(sq, "s--", label="Q-tensor")
        ax[0].set_title("Singular values (capacity)"); ax[0].set_xlabel("channel"); ax[0].set_ylabel("s"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].scatter((scale * Gd).real.ravel(), Gq.real.ravel(), s=3, alpha=.3)
        lim = np.abs(Gq.real).max(); ax[1].plot([-lim, lim], [-lim, lim], "r--", lw=1)
        ax[1].set_title(f"G element-wise (corr {corr:.3f})"); ax[1].set_xlabel("director (gauge-matched)"); ax[1].set_ylabel("Q-tensor"); ax[1].grid(alpha=.3)
        out = os.path.join(RES, "director_vs_qtensor.png"); plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\nsaved {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
