"""Calibrate the GPUmeep AREA (volume) source scale vs MEEP.

The line-source scale (0.39·res) over-injects an area source (the pump), because a
volume current is not N stacked plane sources. This measures the correct scale for a
2D area Ez source: inject the SAME MEEP Ez area source in vacuum in both engines,
compare the DFT |Ez| at a probe line just downstream. ratio = MEEP/gpumeep is the
area-source correction relative to the current per-cell injection.
"""
import os, sys
sys.path.insert(0, os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src"))
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import fdtd_2d as f2

RES = 40
SX, SY = 14.0, 10.0
LAM = 0.53; FREQ = 1.0 / LAM
WIDTH = 2.0; CUT = 5.0; NPML = 2.0
dx = 1.0 / RES
AW = 3.0                      # area source region (AW×AW µm square, centered)
T = 2 * CUT * WIDTH + SX
probe_x = SX / 2 - NPML - 0.5
LINE_SCALE = 0.390625 * RES   # current line-source scale
# Calibrate the pump field WHERE IT COUPLES: DFT |Ez| averaged over the source
# region itself (that's the field that inverts the medium + leaks to Ey), NOT a
# downstream near-field probe. This matches the physically-relevant quantity.
MEASURE_IN_REGION = True


def run_meep():
    import meep as mp
    sim = mp.Simulation(
        cell_size=mp.Vector3(SX, SY, 0), resolution=RES,
        sources=[mp.Source(mp.GaussianSource(FREQ, width=WIDTH, cutoff=CUT), mp.Ez,
                            center=mp.Vector3(0, 0), size=mp.Vector3(AW, AW))],
        boundary_layers=[mp.PML(NPML)])
    # DFT over the SOURCE REGION (where the pump couples), not downstream
    d = sim.add_dft_fields([mp.Ez], FREQ, 0, 1,
                           center=mp.Vector3(0, 0), size=mp.Vector3(AW, AW))
    sim.run(until=T)
    arr = np.abs(np.asarray(sim.get_dft_array(d, mp.Ez, 0)))
    return arr


def run_gpu(area_scale):
    Nx, Ny = int(SX * RES), int(SY * RES)
    grid = f2.Grid2D(Nx=Nx, Ny=Ny, dx=dx, dy=dx)
    dt = float(f2.courant_dt_2d(grid, 0.5))
    n_pml = int(round(NPML / dx))
    pml = f2.make_cpml_full_2d(grid, dt, n_pml=(n_pml, n_pml))
    ezz = jnp.ones((Nx, Ny))
    o = jnp.ones((Nx, Ny)); z = jnp.zeros((Nx, Ny))
    mat = f2.Aniso2DYee(o, z, o, z)
    ic = Nx // 2; jc = Ny // 2; half = int(AW / 2 / dx)
    amp = np.zeros((Nx, Ny))
    amp[ic - half:ic + half, jc - half:jc + half] = 1.0
    amp = jnp.asarray(amp)
    n_total = int(T / dt); omega = 2 * np.pi * FREQ
    fld = f2.zero_fields_full(grid)
    # DFT accumulator over the source region (2D box)
    re = np.zeros((Nx, Ny)); im = np.zeros((Nx, Ny))
    t0 = CUT * WIDTH
    for n in range(n_total):
        t = n * dt
        J = np.exp(-((t - t0) ** 2) / (2 * WIDTH ** 2)) * np.cos(2 * np.pi * FREQ * (t - t0))
        fld = fld._replace(Ez=fld.Ez + (-dt * area_scale * J) * amp)
        fld, pml = f2.step_2d_full(fld, grid, dt, pml, mat, ezz)
        ez = np.asarray(fld.Ez); re += np.cos(omega * t) * ez; im += np.sin(omega * t) * ez
    A = np.abs((re + 1j * im) * dt)
    return A[ic - half:ic + half, jc - half:jc + half]


if __name__ == "__main__":
    print("MEEP area source..."); em = run_meep()
    print("gpumeep area source (×dx scale)..."); eg = run_gpu(LINE_SCALE * dx)
    # compare mean over region (robust to shape) + peak
    mm = em.mean(); mg = eg.mean()
    ratio_mean = mm / mg if mg > 0 else float("nan")
    print(f"MEEP |Ez| region-mean={mm:.5g} max={em.max():.5g}")
    print(f"gpumeep |Ez| (scale=line×dx={LINE_SCALE*dx:.4f}) region-mean={mg:.5g} max={eg.max():.5g}")
    print(f"IN-REGION CORRECTION (MEEP/gpumeep, mean) = {ratio_mean:.5f}")
    print(f"  → correct area scale = {LINE_SCALE * dx * ratio_mean:.5f}")
    # flatten for shape corr
    a = em.ravel(); b = eg.ravel(); n = min(len(a), len(b))
    print(f"  shape-corr (in region) = {np.corrcoef(a[:n], b[:n])[0,1]:.4f}")
