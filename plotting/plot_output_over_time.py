"""Input and output signal over time (1D) for the memory / pulse designs.

The n1–n7 characterizations all collapse a run to scalars; this one keeps the
time axis, which is the only view where fading memory is directly visible: how
long an input feature stays legible in the output after the drive has moved on.

Top panel is the drive u(t) reconstructed from the source spec — nothing about
u is stored on disk, so it is rebuilt exactly the way the run built it:

  * source_type "symbols"  : piecewise-constant u(n) ~ U(amp_range) held
    `symbol_length` t.u. each, regenerated from `seed` via
    symbols_source.symbol_sequence (the same call the readout uses).
  * source_type "gaussian" : the pulse envelope exp(-((t-t0)/w)^2 / 2), i.e.
    the decay_*/decayamp_* runs, so the same function serves the echo ladders.

Bottom panel is the measured output at the point sensor: the raw field plus the
|E|^2 envelope smoothed over ~2 optical periods (the same envelope definition
memory_curve.py and echo_energies.py regress on, so what you see here is what
those analyses actually consume).

    python plotting/plot_output_over_time.py --path data/memory_testing/mem_R0.5_p150_s0
    python plotting/plot_output_over_time.py --path data/memory_testing/decay_p150_R0.5
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point-sensor snapshots are written by whichever backend ran; MEEP is the only
# one with CustomSource, so the symbol runs are always simulation_meep.
SNAP_DIRS = ("simulation_meep", "simulation_gpumeep", "simulation")
SNAP_NAME = "point_snap.npz"
SMOOTH_PERIODS = 2.0
#: MEEP GaussianSource peaks this many widths after start_time (its default).
GAUSSIAN_CUTOFF = 5.0


def envelope(sig: np.ndarray, dt: float, periods: float = SMOOTH_PERIODS,
             lam: float = 0.55) -> np.ndarray:
    """|sig|^2 smoothed over `periods` optical periods (period = lam t.u.).

    Same definition as analysis_T/memory_curve.py and echo_energies.py — keep
    the three in step, the memory numbers are defined on this envelope.
    """
    w = max(3, int(round(periods * lam / dt)))
    return np.convolve(sig ** 2, np.ones(w) / w, mode="same")


def find_snapshot(folder: str | Path) -> Path | None:
    """<folder>/<backend>/point_snap.npz, falling back to the design's mirrors.

    The Nextcloud checkout versions only simulation_data.json (data/** is
    gitignored), so the snapshot usually lives on Orion — reuse the Validator's
    mirror roots rather than keeping a second list of them in step.
    """
    from characterization.class_reservoir_validator import Validator

    roots = [Path(folder)] + [Path(c) for c in
                              Validator._mirror_candidates(str(folder))]
    seen = set()
    for root in roots:
        if str(root) in seen:
            continue
        seen.add(str(root))
        for sub in SNAP_DIRS:
            p = root / sub / SNAP_NAME
            if Validator._guard(p.exists, False):
                return p
    return None


def _drive_source(cfg: dict) -> dict | None:
    """The signal source (not the CW pump) from a simulation_data.json."""
    signal = None
    for k, v in cfg.items():
        if not isinstance(v, dict) or "source_type" not in v:
            continue
        if v["source_type"] == "continuous":       # the pump, not the drive
            continue
        if signal is None or k.endswith("_1"):
            signal = v
    return signal


def input_signal(cfg: dict, t: np.ndarray) -> tuple[np.ndarray, str]:
    """(u(t) on the snapshot time grid, label). Zeros if the type is unknown."""
    src = _drive_source(cfg)
    if src is None:
        return np.zeros_like(t), "input (no signal source found)"

    kind = src.get("source_type")
    amp = src.get("amplitude", 1.0)
    amp = float(amp[0]) if isinstance(amp, (list, tuple)) else float(amp)

    if kind == "symbols":
        from symbols_source import symbol_sequence
        T = float(src["symbol_length"])
        end_time = float(src.get("end_time", cfg.get("run_until", t[-1])))
        n_sym = int(np.ceil(end_time / T)) + 1
        u = symbol_sequence(src.get("seed", 0), n_sym,
                            src.get("amp_range", [0.5, 1.5]))
        idx = np.clip((t / T).astype(int), 0, len(u) - 1)
        held = u[idx] * amp
        held[t > end_time] = 0.0     # drive stops even if the run rings down
        return held, (f"input u(t) — symbols, T={T:g} t.u., seed "
                      f"{src.get('seed', 0)}, amp x{amp:g}")

    if kind == "gaussian":
        # Mirror simplesim.source.Source: the JSON gives a wavelength linewidth
        # dlam, which becomes fwidth = 1/(lam-dlam) - 1/(lam+dlam), and MEEP's
        # GaussianSource peaks `cutoff` (default 5) widths after start_time.
        w = src.get("width")
        if w is None:
            fw = src.get("fwidth")
            if fw is None and src.get("dlam"):
                lam_s, dlam = float(src["lam"]), float(src["dlam"])
                fw = 1.0 / (lam_s - dlam) - 1.0 / (lam_s + dlam)
            w = 1.0 / float(fw) if fw else 5.0
        w = float(w)
        t0 = float(src.get("start_time", 0.0)) + GAUSSIAN_CUTOFF * w
        return (amp * np.exp(-0.5 * ((t - t0) / w) ** 2),
                f"input u(t) — gaussian pulse, amp {amp:g}, width {w:.3g} t.u., "
                f"peak t={t0:.3g}")

    return np.zeros_like(t), f"input (source_type '{kind}' not reconstructable)"


def plot_output_over_time(folder: str | Path, fig_dir: str | Path | None = None,
                          component: str = "Ey", suffix: str = "",
                          t_max: float | None = None,
                          mark_symbols: bool = True) -> Path:
    """Two stacked panels sharing the time axis: drive u(t) over measured out(t).

    `folder` is a design dir (holds simulation_data.json). Returns the png path.
    """
    folder = Path(folder)
    cfg = json.load(open(folder / "simulation_data.json"))

    snap = find_snapshot(folder)
    if snap is None:
        raise FileNotFoundError(
            f"no {SNAP_NAME} under {folder} (looked in {', '.join(SNAP_DIRS)})")
    z = np.load(snap)
    if component not in z.files:
        raise KeyError(f"{snap} has no '{component}' (has {sorted(z.files)})")

    t = np.asarray(z["t"], dtype=float).reshape(-1)
    out = np.asarray(z[component], dtype=float).reshape(len(t), -1)[:, 0]
    if t_max is not None:
        keep = t <= float(t_max)
        t, out = t[keep], out[keep]

    dt = float(np.median(np.diff(t)))
    src = _drive_source(cfg) or {}
    lam = float(src.get("lam", 0.55))
    env = envelope(out, dt, lam=lam)
    u, u_label = input_signal(cfg, t)

    fig, (ax_in, ax_out) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.6], "hspace": 0.12})

    # ---------------- input ----------------
    ax_in.plot(t, u, color="C0", lw=1.2)
    ax_in.fill_between(t, 0, u, color="C0", alpha=0.18)
    ax_in.set_ylabel("input amplitude")
    ax_in.set_title(u_label, fontsize=10)
    ax_in.grid(alpha=0.25)

    # symbol boundaries: the windows the readout integrates over
    T = float(src.get("symbol_length", 0.0) or 0.0)
    if mark_symbols and T > 0:
        for k in range(1, int(t[-1] // T) + 1):
            for ax in (ax_in, ax_out):
                ax.axvline(k * T, color="grey", lw=0.4, alpha=0.45, zorder=0)

    # ---------------- output ----------------
    ax_out.plot(t, out, color="0.65", lw=0.5, label=f"{component}(t)")
    ax_out.plot(t, np.sqrt(np.maximum(env, 0.0)), color="C3", lw=1.4,
                label=f"envelope √⟨{component}²⟩ ({SMOOTH_PERIODS:g} periods)")
    ax_out.set_xlabel("time [t.u.]")
    ax_out.set_ylabel(f"output {component}")
    ax_out.legend(fontsize=8, loc="upper right")
    ax_out.grid(alpha=0.25)

    pump = next((v.get("amplitude") for v in cfg.values()
                 if isinstance(v, dict) and v.get("source_type") == "continuous"),
                None)
    head = folder.name + (f"  |  pump {pump:g}"
                          if isinstance(pump, (int, float)) else "")
    fig.suptitle(f"input vs output over time — {head}", fontsize=11)

    fig_dir = Path(fig_dir) if fig_dir else folder / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"output_over_time{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ====================================================================== __main__
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Plot input and output vs time (1D)")
    ap.add_argument("--path", required=True,
                    help="design dir (has simulation_data.json)")
    ap.add_argument("--fig-dir", default=None, help="default: <path>/figures")
    ap.add_argument("--component", default="Ey",
                    help="stored polarization (default Ey)")
    ap.add_argument("--t-max", type=float, default=None,
                    help="truncate the time axis")
    ap.add_argument("--suffix", default="",
                    help="appended to the output filename")
    a = ap.parse_args()

    p = plot_output_over_time(a.path, fig_dir=a.fig_dir, component=a.component,
                              t_max=a.t_max, suffix=a.suffix)
    print(f"[plot_output_over_time] {p}", flush=True)
