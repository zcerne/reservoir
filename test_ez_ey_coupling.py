"""Isolate the Ez→Ey coupling (LC εyz off-diagonal) — MEEP vs GPUmeep.

Uniform anisotropic medium with a FIXED tilted director (constant φ, θ) so εyz≠0,
driven by a pure Ez plane source. Measures the Ey that the εyz coupling generates.
Config 2 drove Ey and read Ey (never tested an Ez drive), so this is the first direct
check of the Ez→Ey direction — where config-3's pump-leak discrepancy lives.
"""
import os, sys
sys.path.insert(0, os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src"))
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import fdtd_2d as f2

RES = 40
SX, SY = 12.0, 8.0
LAM = 0.53; FREQ = 1.0 / LAM
WIDTH = 2.0; CUT = 5.0; NPML = 2.0
dx = 1.0 / RES
N_O, N_E = 1.52, 1.71
PHI = 0.6           # azimuth (rad)
THETA = 1.0         # polar (rad) — tilted out of plane so nz≠0 → εyz≠0
T = 2 * CUT * WIDTH + SX
probe_i = int(round((SX / 2 - NPML - 1.0 + SX / 2) / dx))


def eps_tensor():
    eps_perp = N_O ** 2; delta = N_E ** 2 - N_O ** 2
    nx = np.sin(THETA) * np.cos(PHI); ny = np.sin(THETA) * np.sin(PHI); nz = np.cos(THETA)
    return dict(xx=eps_perp + delta * nx * nx, yy=eps_perp + delta * ny * ny,
                zz=eps_perp + delta * nz * nz, xy=delta * nx * ny,
                xz=delta * nx * nz, yz=delta * ny * nz)


def run_meep():
    import meep as mp
    e = eps_tensor()
    med = mp.Medium(epsilon_diag=mp.Vector3(e["xx"], e["yy"], e["zz"]),
                    epsilon_offdiag=mp.Vector3(e["xy"], e["xz"], e["yz"]))
    sim = mp.Simulation(cell_size=mp.Vector3(SX, SY, 0), resolution=RES,
                        default_material=med,
                        sources=[mp.Source(mp.GaussianSource(FREQ, width=WIDTH, cutoff=CUT), mp.Ez,
                                 center=mp.Vector3(-SX/2 + NPML + 1.0, 0), size=mp.Vector3(0, SY - 2*NPML))],
                        boundary_layers=[mp.PML(NPML)])
    px = SX/2 - NPML - 1.0
    dez = sim.add_dft_fields([mp.Ez], FREQ, 0, 1, center=mp.Vector3(px, 0), size=mp.Vector3(0, SY-2*NPML))
    dey = sim.add_dft_fields([mp.Ey], FREQ, 0, 1, center=mp.Vector3(px, 0), size=mp.Vector3(0, SY-2*NPML))
    sim.run(until=T)
    ez = np.abs(np.asarray(sim.get_dft_array(dez, mp.Ez, 0)))
    ey = np.abs(np.asarray(sim.get_dft_array(dey, mp.Ey, 0)))
    return ez, ey


def run_gpu():
    Nx, Ny = int(SX*RES), int(SY*RES)
    grid = f2.Grid2D(Nx=Nx, Ny=Ny, dx=dx, dy=dx)
    dt = float(f2.courant_dt_2d(grid, 0.5, n_max=N_E))
    n_pml = int(round(NPML/dx))
    pml = f2.make_cpml_full_2d(grid, dt, n_pml=(n_pml, n_pml))
    e = eps_tensor()
    o = np.ones((Nx, Ny))
    def inv3(xx, yy, zz, xy, xz, yz):
        det = xx*(yy*zz-yz*yz) - xy*(xy*zz-yz*xz) + xz*(xy*yz-yy*xz)
        return ((yy*zz-yz*yz)/det, (xx*zz-xz*xz)/det, (xx*yy-xy*xy)/det,
                (xz*yz-xy*zz)/det, (xy*yz-xz*yy)/det, (xz*xy-xx*yz)/det)
    ixx, iyy, izz, ixy, ixz, iyz = inv3(*(e[k]*o for k in ("xx","yy","zz","xy","xz","yz")))
    J = lambda a: jnp.asarray(a)
    mat = f2.AnisoFull2D(ixx_Ex=J(ixx), ixy_Ex=J(ixy), ixz_Ex=J(ixz),
                         ixy_Ey=J(ixy), iyy_Ey=J(iyy), iyz_Ey=J(iyz),
                         ixz_nd=J(ixz), iyz_nd=J(iyz), izz_nd=J(izz))
    i_src = int(round((-SX/2 + NPML + 1.0 + SX/2)/dx))
    i_prb = int(round((SX/2 - NPML - 1.0 + SX/2)/dx))
    j0 = int(round(NPML/dx)); j1 = Ny - j0
    amp = np.zeros(Ny); amp[j0:j1] = 1.0; amp = jnp.asarray(amp)
    scale = 0.390625 * RES
    izz_line = jnp.asarray(izz[i_src, :])     # eps_inv_zz at the source line (as real source does)
    n_total = int(T/dt); omega = 2*np.pi*FREQ; t0 = CUT*WIDTH
    fld = f2.zero_fields_full(grid)
    reZ = np.zeros(Ny); imZ = np.zeros(Ny); reY = np.zeros(Ny); imY = np.zeros(Ny)
    for n in range(n_total):
        t = n*dt
        Jt = np.exp(-((t-t0)**2)/(2*WIDTH**2))*np.cos(2*np.pi*FREQ*(t-t0))
        fld = fld._replace(Ez=fld.Ez.at[i_src, :].add(-dt*scale*izz_line*Jt*amp))
        fld, pml = f2.step_2d_full(fld, grid, dt, pml, mat, J(izz))
        ez = np.asarray(fld.Ez[i_prb, :]); ey = np.asarray(fld.Ey[i_prb, :])
        reZ += np.cos(omega*t)*ez; imZ += np.sin(omega*t)*ez
        reY += np.cos(omega*t)*ey; imY += np.sin(omega*t)*ey
    return np.abs((reZ+1j*imZ)*dt), np.abs((reY+1j*imY)*dt)


if __name__ == "__main__":
    e = eps_tensor()
    print(f"eps: xx={e['xx']:.3f} yy={e['yy']:.3f} zz={e['zz']:.3f} xy={e['xy']:.3f} xz={e['xz']:.3f} yz={e['yz']:.3f}")
    print("MEEP..."); mz, my = run_meep()
    print("gpumeep..."); gz, gy = run_gpu()
    n = min(len(mz), len(gz))
    mz, my, gz, gy = mz[:n], my[:n], gz[:n], gy[:n]
    print(f"MEEP:    |Ez|max={mz.max():.5g}  |Ey|max={my.max():.5g}  Ey/Ez={my.max()/mz.max():.4f}")
    print(f"gpumeep: |Ez|max={gz.max():.5g}  |Ey|max={gy.max():.5g}  Ey/Ez={gy.max()/gz.max():.4f}")
    print(f"Ez match (gpu/meep) = {gz.max()/mz.max():.4f}")
    print(f"Ey(coupled) match (gpu/meep) = {gy.max()/my.max():.4f}")
    print(f"coupling ratio (gpu Ey/Ez) / (meep Ey/Ez) = {(gy.max()/gz.max())/(my.max()/mz.max()):.4f}")
