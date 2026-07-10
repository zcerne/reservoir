"""Decisive ε comparison for config 4 (isotropic DBR): the exact permittivity
MEEP builds vs the exact permittivity gpumeep builds, on the FDTD grid.

Extracts:
  * MEEP  ε (scalar; config 4 is isotropic) via init_sim + get_array(Dielectric).
  * gpumeep ε_yy = 1/iyy_Ey and ε_xx = 1/ixx_Ex from the AnisoFull2D tensor.
Plots a horizontal cut (y=0) through the mirror stack and prints where they differ.
"""
import os, sys, importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RESV)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json, CONFIGS  # noqa: E402


def meep_eps(path):
    import meep as mp
    from class_simulation import Simulation
    sim = Simulation(path)
    sim._set_everything()
    sim.simulation.init_sim()
    # snap=True → RAW per-Yee-pixel epsilon (no centering interpolation), which is
    # the actual averaged ε the solver steps with (reaches n² at thick layers).
    eps = np.asarray(sim.simulation.get_epsilon(snap=True))
    md = sim.simulation.get_array_metadata()
    xco = np.asarray(md[0]); yco = np.asarray(md[1])
    print(f"MEEP snap eps shape={eps.shape}  x[0:3]={xco[:3]}  x[-1]={xco[-1]:.5f}  n_x={len(xco)}")
    ny = eps.shape[1]
    cut = eps[:, ny // 2]
    x = xco[:eps.shape[0]]
    # Dump the actual MEEP geometry block x-extents (mirror layers) for alignment check
    print("--- MEEP mirror-layer blocks (center_x, size_x, x_lo, x_hi) ---")
    for g in sim.geometry:
        try:
            cx = float(g.center.x); sx = float(g.size.x)
        except Exception:
            continue
        if 0 < sx < 0.5:   # DBR layers are thin
            print(f"  cx={cx:+.5f} sx={sx:.5f}  x=[{cx-sx/2:+.5f}, {cx+sx/2:+.5f}]")
    return x, cut


def gpu_eps(path):
    import jax
    jax.config.update("jax_enable_x64", True)
    sys.modules.pop("class_simulation_gpu", None)
    csg = importlib.import_module("class_simulation_gpu")
    assert os.path.dirname(csg.__file__) == RESV, csg.__file__
    sim = csg.SimulationGPU(folder_path=path)
    sim.force_fullvector = True
    sim._set_data(); sim._update_all_args(); sim._build_material()
    sim._setup_lc_interp()
    # Build gpu ε at the CELL-CENTER (i+1/2, j+1/2) — MEEP's Dielectric location — so
    # we compare the ε-AVERAGING algorithm apples-to-apples (same Yee offset).
    i = np.arange(sim.Nx); j = np.arange(sim.Ny)
    Xc = ((i + 0.5) * sim.dx - sim.gx / 2)[:, None] * np.ones((1, sim.Ny))
    Yc = ((j + 0.5) * sim.dx - sim.gy / 2)[None, :] * np.ones((sim.Nx, 1))
    e6c = list(sim._eps_sharp_at(Xc, Yc))
    e6c = sim._overlay_iso_kottke(e6c, Xc, Yc)
    # MEEP get_eps (monitor.cpp:172) = nc / Σ ε⁻¹_cc = HARMONIC mean of the diagonal
    eps_cc = 3.0 / (1.0 / e6c[0] + 1.0 / e6c[1] + 1.0 / e6c[2])
    ny = eps_cc.shape[1]
    x = (np.arange(sim.Nx) + 0.5) * sim.dx - sim.gx / 2
    print(f"gpu cell-center x[0:3]={x[:3]}  dx={sim.dx}  Nx={sim.Nx}")
    return x, eps_cc[:, ny // 2], eps_cc[:, ny // 2], sim.dx


def main():
    path = build_json(4)
    xm, em = meep_eps(path)
    xg, eyy, exx, dx = gpu_eps(path)

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax[0].plot(xm, em, "b-", lw=2, label="MEEP ε (get_array Dielectric)")
    ax[0].plot(xg, eyy, "r--", lw=1.3, label="gpumeep ε_yy (=1/iyy_Ey)")
    ax[0].plot(xg, exx, "g:", lw=1.3, label="gpumeep ε_xx (=1/ixx_Ex, harmonic)")
    ax[0].set_ylabel("ε"); ax[0].legend(); ax[0].set_title("config 4 DBR — ε cut at y=0")
    # interpolate MEEP onto gpu-yy x-grid to diff
    em_i = np.interp(xg, xm, em)
    ax[1].plot(xg, eyy - em_i, "r-", lw=1, label="ε_yy(gpu) − ε(MEEP)")
    ax[1].plot(xg, exx - em_i, "g-", lw=1, label="ε_xx(gpu) − ε(MEEP)")
    ax[1].set_xlabel("x (µm)"); ax[1].set_ylabel("Δε"); ax[1].legend()
    ax[1].set_title("difference")
    plt.tight_layout()
    out = "/home/cernez/resevoir/ladder/eps_cut_config4.png"
    plt.savefig(out, dpi=120)
    print("saved", out)
    print(f"MEEP  ε: min={em.min():.4f} max={em.max():.4f} n_pts={len(em)}")
    print(f"gpu ε_yy: min={eyy.min():.4f} max={eyy.max():.4f} n_pts={len(eyy)}")
    print(f"gpu ε_xx: min={exx.min():.4f} max={exx.max():.4f}")
    print(f"max|Δ ε_yy|(interp) = {np.abs(eyy - em_i).max():.4f}   "
          f"RMS = {np.sqrt(np.mean((eyy - em_i)**2)):.4f}")
    # INDEX-ALIGNED diff (both 588 pts, both centered grids) — no interpolation error
    if len(eyy) == len(em):
        # Rule out a 1-pixel diagnostic misalignment (snap value vs metadata coord):
        # scan integer shifts and report the best. If shift 0 is best → real ε diff;
        # if ±1 collapses it → the "difference" was a metadata off-by-one.
        for sh in (-2, -1, 0, 1, 2):
            a = eyy[max(0, sh):len(eyy)+min(0, sh)]
            b = em[max(0, -sh):len(em)+min(0, -sh)]
            print(f"  shift={sh:+d}: max|Δ|={np.abs(a-b).max():.4f}  "
                  f"RMS={np.sqrt(np.mean((a-b)**2)):.4f}  "
                  f"n|Δ|>0.01={int(np.sum(np.abs(a-b)>0.01))}")
        d_idx = eyy - em
        print(f"max|Δ ε_yy|(index) = {np.abs(d_idx).max():.4f}   "
              f"RMS = {np.sqrt(np.mean(d_idx**2)):.4f}   "
              f"n_pixels|Δ|>0.01 = {int(np.sum(np.abs(d_idx) > 0.01))}/{len(d_idx)}")
        # show the worst-mismatch pixels
        w = np.argsort(-np.abs(d_idx))[:6]
        for k in w:
            print(f"    x={xg[k]:+.4f}  gpu ε_yy={eyy[k]:.4f}  MEEP ε={em[k]:.4f}  Δ={d_idx[k]:+.4f}")
    # count layers (contiguous ε>1.05 regions) in each
    def nlayers(e, x):
        hi = e > 1.05
        return int(np.sum(hi[1:] & ~hi[:-1])) + int(hi[0])
    print(f"MEEP hi-index segments={nlayers(em, xm)}  gpu ε_yy segments={nlayers(eyy, xg)}")


if __name__ == "__main__":
    main()
