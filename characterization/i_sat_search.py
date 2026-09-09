#!/usr/bin/env python
"""i_sat_search.py — find the STATIONARY operating point of the CW space reservoir.

The FDTD can only reach ~33 ps, but the stationary state is the µs pump-vs-burn
balance of the inversion. Because the optical field equilibrates ~10^5-10^8x
faster than the inversion moves, the stationary state is the SELF-CONSISTENT pair
(N3*, I*) satisfying both:

    I*   = steady field for a FROZEN inversion N3*        (fast: solve optics)
    N3*  = N3_0 / (1 + I*/I_sat)                          (slow: pump-vs-burn balance)

We find it by fixed-point iteration (adiabatic elimination of the fast field):
    1. freeze N3(x,y)              (start full)
    2. solve the steady field  ->  I(x,y)
    3. update  N3 <- N3_0/(1+I/I_sat)
    4. repeat until (N3, I) reproduce each other.
The loop is negative feedback (more I -> less N3 -> less gain -> less I), so it
converges; we damp the N3 update to be safe.

Then we SEARCH the input drive: sweep s = I_in/I_sat and report the useful
"graded" window (crystal partially saturated, not linear, not bleached) — the
operating regime, and, for N co-driven channels, the per-channel level
(~I_sat/N so the local sum lands at ~I_sat where the cross-saturation mixing is
strongest).

FIELD BACKEND: default is a self-contained saturable-gain beam-propagation
model (paraxial split-step: transverse diffraction + longitudinal saturable
gain) — runnable now with numpy, captures both saturation and the 4-beam
transverse mixing. To use the real gpumeep FDTD as the field solver instead,
implement `steady_field_fdtd` (needs three gpumeep hooks: inject a per-cell
inversion profile, hold populations fixed during the optical solve, read the 2D
in-crystal intensity) and pass --backend fdtd. The iteration itself is unchanged.

Units are dimensionless: intensities in units of I_sat, so I_sat = 1 here.
Absolute amplitudes map to the sim via the one calibration point you have
(small-signal gL, or a known amp<->intensity run).

Usage:
    python characterization/i_sat_search.py --design data/signal_modulation/design04_4source
    python characterization/i_sat_search.py --design ... --drive 1.0 --plot out.png
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


# --------------------------------------------------------------------------
# Field backend: steady field at a FROZEN inversion (paraxial saturable-gain BPM)
# --------------------------------------------------------------------------
def steady_field_bpm(N3, cfg, drive, n_bg, lam, g0L):
    """Steady |field|^2 = I(x,y) [units of I_sat] for a frozen inversion N3(Nx,Ny).

    Split-step along x (propagation): half diffraction, full saturable-gain, half
    diffraction. Gain acts on the field amplitude: dA/dx = (g/2) A with local
    small-signal g(x,y) = (g0L/Lx) * N3(x,y)/N3_0.  N3 is held FIXED here (that is
    the point of the freeze) — saturation enters only through the iteration that
    updates N3 between calls.  Input = the 4 strips at `drive`.
    """
    Nx, Ny = N3.shape
    Lx, Ly = cfg["Lx"], cfg["Ly"]
    dx, dy = Lx / Nx, Ly / Ny
    N3_0 = cfg["N3_0"]
    # transverse wavenumbers for the paraxial diffraction step
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    k0 = 2 * np.pi * n_bg / lam
    diff_half = np.exp(-1j * (ky**2) / (2 * k0) * (dx / 2))   # half-step propagator
    gain_line = (g0L / Lx) * dx                                # g0*dx per step (small-signal)

    # input field: sum of the strips, each carrying sqrt(drive) amplitude
    y = (np.arange(Ny) - Ny / 2) * dy
    A = np.zeros(Ny, dtype=complex)
    for yc, w in cfg["strips"]:
        A += np.sqrt(drive) * (np.abs(y - yc) <= w / 2).astype(float)
    I_map = np.zeros((Nx, Ny))
    for i in range(Nx):
        A = np.fft.ifft(np.fft.fft(A) * diff_half)            # diffraction half-step
        A *= np.exp(0.5 * gain_line * (N3[i] / N3_0))         # saturable gain (N3 frozen)
        A = np.fft.ifft(np.fft.fft(A) * diff_half)            # diffraction half-step
        I_map[i] = np.abs(A) ** 2
    return I_map


# --------------------------------------------------------------------------
# Self-consistent fixed-point iteration
# --------------------------------------------------------------------------
def stationary_state(cfg, drive, n_bg=1.82, lam=1.064, g0L=2.32,
                     max_iter=60, tol=1e-4, damp=0.5, verbose=False):
    """Return (N3*, I*, iters, converged) for a given input drive (in I_sat)."""
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    N3_0 = cfg["N3_0"]
    N3 = np.full((Nx, Ny), N3_0)          # iteration-zero: full inversion
    for it in range(1, max_iter + 1):
        I_map = steady_field_bpm(N3, cfg, drive, n_bg, lam, g0L)
        N3_new = N3_0 / (1.0 + I_map)     # saturation (I in units of I_sat)
        rel = np.max(np.abs(N3_new - N3)) / N3_0
        N3 = (1 - damp) * N3 + damp * N3_new
        if verbose:
            print(f"  iter {it:2d}: max|dN3|/N3_0={rel:.2e}  mean N3/N3_0={N3.mean()/N3_0:.3f}")
        if rel < tol:
            return N3, I_map, it, True
    return N3, I_map, max_iter, False


# --------------------------------------------------------------------------
# Load geometry/calibration from the design config
# --------------------------------------------------------------------------
def load_cfg(design_dir, Nx=240, Ny=176):
    d = json.load(open(os.path.join(design_dir, "simulation_data.json")))
    Lx, Ly = d["crystal"]["sizes"]
    strips = []
    for k in sorted(d):
        if k.startswith("source_") and isinstance(d[k], dict) and d[k].get("component") == "Ey":
            p = d[k]["position"]
            yc = float(p.get("y", 0.0)); w = float(p["size"][1])
            strips.append((yc, w))
    n_bg = d.get("crystal", {}).get("n", d.get("background_index", 1.82))
    return {"Lx": Lx, "Ly": Ly, "Nx": Nx, "Ny": Ny, "N3_0": 1.0, "strips": strips, "n_bg": n_bg}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True, help="design folder with simulation_data.json")
    ap.add_argument("--drive", type=float, default=None,
                    help="single input drive in units of I_sat; if omitted, sweep a ladder")
    ap.add_argument("--g0L", type=float, default=2.32, help="small-signal gain-length (design gL)")
    ap.add_argument("--Nx", type=int, default=240); ap.add_argument("--Ny", type=int, default=176)
    ap.add_argument("--plot", default=None, help="save a plot to this path")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    cfg = load_cfg(a.design, a.Nx, a.Ny)
    cfg["n_bg"] = cfg.get("n_bg", 1.82)
    N = len(cfg["strips"])
    print(f"design: {a.design}  crystal {cfg['Lx']}x{cfg['Ly']} um  {N} input strips  g0L={a.g0L}")

    drives = [a.drive] if a.drive is not None else list(np.logspace(-1.5, 1.5, 13))
    rows = []
    for s in drives:
        N3, I_map, it, conv = stationary_state(cfg, s, n_bg=cfg["n_bg"], g0L=a.g0L, verbose=a.verbose)
        sat = 1.0 - N3.mean() / cfg["N3_0"]                 # mean fractional depletion
        out = I_map[-1]                                     # output intensity profile
        rows.append((s, sat, out, N3, conv, it))
        print(f"  drive/I_sat={s:8.3f}  mean depletion={sat:5.3f}  "
              f"{'OK' if conv else 'NOCONV'} ({it} it)  peak I_out={out.max():.3g}")

    # operating point: mean depletion closest to 0.5 (half-saturated = graded)
    s_op = min(rows, key=lambda r: abs(r[1] - 0.5))[0]
    print(f"\noperating point (mean depletion ~0.5): per-strip drive I_in ~ {s_op:.3f} I_sat "
          f"(all {N} strips co-driven equally). NOTE: this is the PER-STRIP input intensity; "
          f"where the beams overlap in the gain region the local sum is higher, which is what "
          f"pulls the mean depletion to ~0.5 at a per-strip drive above I_sat.")
    print("useful sweep ladder around it (x I_sat):",
          [round(s_op * f, 3) for f in (0.1, 0.25, 0.5, 1, 2, 4, 8)])

    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ss = [r[0] for r in rows]; sats = [r[1] for r in rows]
        ax[0].semilogx(ss, sats, "o-"); ax[0].axhline(0.5, color="0.6", ls="--")
        ax[0].axvline(s_op, color="r", ls=":"); ax[0].set_xlabel("I_in / I_sat")
        ax[0].set_ylabel("mean inversion depletion"); ax[0].set_title("saturation vs drive"); ax[0].grid(alpha=.3)
        rop = min(rows, key=lambda r: abs(r[0] - s_op))
        y = (np.arange(cfg["Ny"]) - cfg["Ny"] / 2) * (cfg["Ly"] / cfg["Ny"])
        ax[1].plot(y, rop[2]); ax[1].set_xlabel("y (um)"); ax[1].set_ylabel("I_out"); ax[1].set_title(f"output @ {s_op:.2f} I_sat"); ax[1].grid(alpha=.3)
        im = ax[2].imshow(rop[3].T / cfg["N3_0"], aspect="auto", origin="lower",
                          extent=[0, cfg["Lx"], y.min(), y.max()], cmap="viridis")
        ax[2].set_xlabel("x (um)"); ax[2].set_ylabel("y (um)"); ax[2].set_title("stationary N3/N3_0"); fig.colorbar(im, ax=ax[2])
        fig.tight_layout(); os.makedirs(os.path.dirname(os.path.abspath(a.plot)), exist_ok=True)
        fig.savefig(a.plot, dpi=130); print("wrote", a.plot)


if __name__ == "__main__":
    raise SystemExit(main())
