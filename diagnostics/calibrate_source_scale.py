"""Calibrate the GPUmeep current-source absolute scale against MEEP (vacuum).

Injects the SAME plane Ey GaussianSource (amplitude 1) in MEEP and in the GPUmeep
core, both in vacuum with identical geometry/PML, and measures |Ey| at a downstream
DFT probe. The ratio MEEP/gpumeep is the absolute source-scale factor that the
current-source injection (E += -dt·ε⁻¹·J) is missing — needed because the STED gain
is nonlinear (absolute field level sets the saturation), so a global scale is NOT
free. Run on smaug (pmp env has both meep and jax).
"""
import os, sys
sys.path.insert(0, os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import fdtd_2d as f2

RES = int(os.environ.get("CAL_RES", "40"))
SX, SY = 12.0, 8.0
LAM = 0.55; FREQ = 1.0 / LAM
WIDTH = 2.0; CUT = 5.0
NPML = 2.0
dx = 1.0 / RES
T = 2 * CUT * WIDTH + (SX)          # enough for the pulse to cross to the probe
src_x = -SX / 2 + NPML + 0.5
probe_x = SX / 2 - NPML - 0.5


def run_meep():
    import meep as mp
    sim = mp.Simulation(
        cell_size=mp.Vector3(SX, SY, 0), resolution=RES,
        sources=[mp.Source(mp.GaussianSource(FREQ, width=WIDTH, cutoff=CUT), mp.Ey,
                            center=mp.Vector3(src_x, 0), size=mp.Vector3(0, SY - 2 * NPML))],
        boundary_layers=[mp.PML(NPML)])
    d = sim.add_dft_fields([mp.Ey], FREQ, 0, 1,
                           center=mp.Vector3(probe_x, 0), size=mp.Vector3(0, SY - 2 * NPML))
    sim.run(until=T)
    Ey = np.abs(np.asarray(sim.get_dft_array(d, mp.Ey, 0)))
    return Ey


def run_gpumeep():
    Nx, Ny = int(SX * RES), int(SY * RES)
    grid = f2.Grid2D(Nx=Nx, Ny=Ny, dx=dx, dy=dx)
    dt = float(f2.courant_dt_2d(grid, 0.5))
    n_pml = int(round(NPML / dx))
    pml = f2.make_cpml_2d(grid, dt, n_pml=(n_pml, n_pml))
    o = jnp.ones((Nx, Ny)); z = jnp.zeros((Nx, Ny))
    mat = f2.Aniso2DYee(o, z, o, z)
    i_src = int(round((src_x + SX / 2) / dx))
    i_prb = int(round((probe_x + SX / 2) / dx))
    j0 = int(round(NPML / dx)); j1 = Ny - j0
    amp = np.zeros(Ny); amp[j0:j1] = 1.0
    src = f2.CurrentSource2D(axis=0, index=i_src, component="Ey",
                             amplitude_1d=jnp.asarray(amp), frequency=FREQ, dt=dt,
                             width=WIDTH, cutoff=CUT)
    n_total = int(T / dt)
    omega = 2 * np.pi * FREQ
    fld = grid.zero_fields(); re = np.zeros(Ny); im = np.zeros(Ny)
    for n in range(n_total):
        t = n * dt
        fld = src.apply(fld, t)
        fld, pml = f2.step_2d(fld, grid, dt, pml, mat)
        ey = np.asarray(fld.Ey[i_prb, :]); re += np.cos(omega * t) * ey; im -= np.sin(omega * t) * ey
    # MEEP DFT convention: F(ω)=Σ f e^{iωt} dt (NO 1/N — the ×2/n_total averaging is
    # only valid for CW steady-state, WRONG for a finite pulse).
    scale = dt
    # restrict to MEEP's monitor span (y ∈ [-(SY-2NPML)/2, +]) so shapes align
    return np.abs((re + 1j * im) * scale)[j0:j1]


if __name__ == "__main__":
    print("running MEEP vacuum..."); em = run_meep()
    print("running gpumeep vacuum..."); eg = run_gpumeep()
    n = min(len(em), len(eg)); em = em[:n]; eg = eg[:n]
    ratio = em.max() / eg.max() if eg.max() > 0 else float("nan")
    dt = float(f2.courant_dt_2d(f2.Grid2D(int(SX*RES), int(SY*RES), dx, dx), 0.5))
    print(f"MEEP |Ey| max={em.max():.6g} mean={em.mean():.6g}")
    print(f"gpumeep |Ey| max={eg.max():.6g} mean={eg.mean():.6g}")
    print(f"SOURCE SCALE (MEEP/gpumeep) = {ratio:.4f}")
    print(f"  compare: 0.5/dt = {0.5/dt:.4f}   1/dt = {1/dt:.4f}   dt={dt:.5f}")
    print(f"  shape-corr = {np.corrcoef(em, eg)[0,1]:.4f}")
