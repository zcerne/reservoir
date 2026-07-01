"""Plots for the voltage-electrode reservoir.

Reads `simulation/voltage_reservoir.npz` (saved by VoltageReservoir.save())
and `simulation/opt_sensor.npz` (saved by class_simulation_gpu) and emits
five figures into `figures/`:

  1. potential.png       — V (electric potential) heatmap with electrode positions
  2. E_vectors.png       — quiver of E direction on top of |E| colormap
  3. E_amplitude.png     — |E| heatmap (log scale)
  4. E_angle.png         — atan2(Ey, Ex) heatmap (HSV cyclic)
  5. director_field.png  — director line-segments + φ heatmap (HSV)
  6. opt_sensor.png      — 1D intensity I(y) at output guide

Mid-z slice used for all 2D fields. Director shown as line segments (no
arrow heads) because the LC director is a headless vector (n ↔ −n).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _load_voltage(folder: Path) -> dict:
    p = folder / "simulation" / "voltage_reservoir.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run run_voltage_reservoir.py first")
    return dict(np.load(p, allow_pickle=True))


def _load_sensor(folder: Path, sim_args: dict) -> tuple[str, dict] | None:
    """Find the first 1Ddft monitor in object_order and load its npz.
    Returns (key, data) or None if no 1Ddft monitor / missing npz."""
    for key in sim_args.get("object_order", []):
        obj = sim_args.get(key, {})
        if obj.get("class") == "monitor" and obj.get("type") == "1Ddft":
            p = folder / "simulation" / f"{key}.npz"
            if p.exists():
                return key, dict(np.load(p))
    return None


def _mid_z(arr: np.ndarray) -> np.ndarray:
    """Take the mid-z slice of a (..., nz) array."""
    return arr[..., arr.shape[-1] // 2]


# ---------------- Potential ----------------

def plot_potential(folder: Path, data: dict) -> Path:
    """V (electric potential) heatmap with electrode positions marked.

    Mid-z slice; symmetric color range around 0 (diverging colormap).
    """
    V = _mid_z(data["V"])
    sx, sy = float(data["sizes"][0]), float(data["sizes"][1])
    extent = (-sx / 2, sx / 2, -sy / 2, sy / 2)
    vmax = float(np.abs(V).max()) if np.abs(V).max() > 0 else 1.0
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(V.T, origin="lower", extent=extent, cmap="RdBu_r",
                   aspect="equal", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="V (volts)")
    # Mark electrodes on each of the 4 in-plane faces.
    for face in ("x_min", "x_max", "y_min", "y_max"):
        vs = np.asarray(data.get(f"voltages_{face}", []))
        if vs.size == 0:
            continue
        if face.startswith("y"):
            pitch = sx / vs.size
            y_pos = sy / 2.0 if face == "y_max" else -sy / 2.0
            for k in range(vs.size):
                xc = -sx / 2.0 + (k + 0.5) * pitch
                ax.plot(xc, y_pos, "o", color="black", markersize=7,
                        markerfacecolor=("crimson" if vs[k] > 0 else "navy" if vs[k] < 0 else "white"))
                ax.annotate(f"{vs[k]:+.1f}", (xc, y_pos),
                            textcoords="offset points",
                            xytext=(0, 8 if face == "y_max" else -14),
                            ha="center", fontsize=7)
        else:
            pitch = sy / vs.size
            x_pos = sx / 2.0 if face == "x_max" else -sx / 2.0
            for k in range(vs.size):
                yc = -sy / 2.0 + (k + 0.5) * pitch
                ax.plot(x_pos, yc, "o", color="black", markersize=7,
                        markerfacecolor=("crimson" if vs[k] > 0 else "navy" if vs[k] < 0 else "white"))
                ax.annotate(f"{vs[k]:+.1f}", (x_pos, yc),
                            textcoords="offset points",
                            xytext=(8 if face == "x_max" else -8, 0),
                            ha="left" if face == "x_max" else "right",
                            va="center", fontsize=7)
    ax.set_title(f"Electric potential V  (|V|max = {vmax:.2f} V)")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    fig.tight_layout()
    out = folder / "figures" / "potential.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


# ---------------- E field plots ----------------

def plot_e_vectors(folder: Path, data: dict, stride: int = 4) -> Path:
    """E direction quiver overlaid on |E| log colormap, mid-z slice."""
    E = _mid_z(data["E"])           # (3, nx, ny)
    Ex2 = np.asarray(E[0]); Ey2 = np.asarray(E[1])
    Emag = np.sqrt(Ex2 ** 2 + Ey2 ** 2)
    sx, sy = float(data["sizes"][0]), float(data["sizes"][1])
    extent = (-sx / 2, sx / 2, -sy / 2, sy / 2)
    nx, ny = Ex2.shape

    fig, ax = plt.subplots(figsize=(8, 5))
    vmin = max(Emag[Emag > 0].min() if (Emag > 0).any() else 1e-6, Emag.max() * 1e-4)
    im = ax.imshow(Emag.T, origin="lower", extent=extent, cmap="inferno",
                   aspect="equal", norm=LogNorm(vmin=vmin, vmax=Emag.max()))
    plt.colorbar(im, ax=ax, label="|E| (log) [V/µm]")
    # Direction-only quiver (unit vectors).
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)
    XX, YY = np.meshgrid(xs[::stride], ys[::stride], indexing="ij")
    Ex_q = Ex2[::stride, ::stride]; Ey_q = Ey2[::stride, ::stride]
    norm_q = np.sqrt(Ex_q ** 2 + Ey_q ** 2)
    Ex_dir = np.where(norm_q > 0, Ex_q / norm_q, 0.0)
    Ey_dir = np.where(norm_q > 0, Ey_q / norm_q, 0.0)
    arrow_len = 0.6 * max(sx, sy) / (max(nx, ny) / stride)
    ax.quiver(XX, YY, Ex_dir, Ey_dir, color="cyan", pivot="mid",
              scale_units="xy", angles="xy", scale=1.0 / arrow_len, width=0.0025)
    ax.set_title("E direction (cyan arrows) on |E| log colormap")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    fig.tight_layout()
    out = folder / "figures" / "E_vectors.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def plot_e_amplitude(folder: Path, data: dict) -> Path:
    """|E| heatmap (log color scale), mid-z slice."""
    E = _mid_z(data["E"])
    Emag = np.sqrt(E[0] ** 2 + E[1] ** 2)
    sx, sy = float(data["sizes"][0]), float(data["sizes"][1])
    extent = (-sx / 2, sx / 2, -sy / 2, sy / 2)
    fig, ax = plt.subplots(figsize=(8, 5))
    vmin = max(Emag[Emag > 0].min() if (Emag > 0).any() else 1e-6, Emag.max() * 1e-4)
    im = ax.imshow(Emag.T, origin="lower", extent=extent, cmap="inferno",
                   aspect="equal", norm=LogNorm(vmin=vmin, vmax=Emag.max()))
    plt.colorbar(im, ax=ax, label="|E| (log) [V/µm]")
    ax.set_title(f"|E| amplitude  (max = {Emag.max():.2f} V/µm)")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    fig.tight_layout()
    out = folder / "figures" / "E_amplitude.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def plot_e_angle(folder: Path, data: dict) -> Path:
    """atan2(Ey, Ex) heatmap with HSV cyclic colormap, mid-z slice."""
    E = _mid_z(data["E"])
    ang = np.arctan2(E[1], E[0])
    sx, sy = float(data["sizes"][0]), float(data["sizes"][1])
    extent = (-sx / 2, sx / 2, -sy / 2, sy / 2)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(ang.T, origin="lower", extent=extent, cmap="hsv",
                   aspect="equal", vmin=-np.pi, vmax=np.pi, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="angle (rad)")
    ax.set_title("E field angle — atan2(Ey, Ex)")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    fig.tight_layout()
    out = folder / "figures" / "E_angle.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


# ---------------- Director ----------------

def plot_director(folder: Path, data: dict, stride: int = 4) -> Path:
    """Director field as headless line segments on top of φ heatmap (HSV)."""
    phi = _mid_z(data["phi"])
    phi_w = (phi + np.pi) % (2 * np.pi) - np.pi
    sx, sy = float(data["sizes"][0]), float(data["sizes"][1])
    extent = (-sx / 2, sx / 2, -sy / 2, sy / 2)
    nx, ny = phi.shape

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(phi_w.T, origin="lower", extent=extent, cmap="hsv",
                   aspect="equal", vmin=-np.pi, vmax=np.pi, interpolation="nearest",
                   alpha=0.65)
    plt.colorbar(im, ax=ax, label="φ wrapped (rad)")
    # Headless line segments — director is a tensor (n ↔ −n)
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)
    XX, YY = np.meshgrid(xs[::stride], ys[::stride], indexing="ij")
    nx_q = np.cos(phi[::stride, ::stride]); ny_q = np.sin(phi[::stride, ::stride])
    seg_half = 0.5 * 0.7 * max(sx, sy) / (max(nx, ny) / stride)
    for ix in range(XX.shape[0]):
        for iy in range(XX.shape[1]):
            x0, y0 = XX[ix, iy], YY[ix, iy]
            ax.plot([x0 - seg_half * nx_q[ix, iy], x0 + seg_half * nx_q[ix, iy]],
                    [y0 - seg_half * ny_q[ix, iy], y0 + seg_half * ny_q[ix, iy]],
                    color="black", lw=1.0, solid_capstyle="round")
    ax.set_title(f"Director field (lines = n, no head)  —  φ range "
                 f"[{phi_w.min():+.2f}, {phi_w.max():+.2f}]")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    fig.tight_layout()
    out = folder / "figures" / "director_field.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


# ---------------- Output sensor I(y) ----------------

def plot_reservoir_dft(folder: Path, sensor_key: str, sensor: dict) -> Path:
    """2D |E|² heatmap from a 2Ddft monitor. Saved as <key>.png."""
    Ex = sensor["Ex"][0] if sensor["Ex"].ndim == 3 else sensor["Ex"]
    Ey = sensor["Ey"][0] if sensor["Ey"].ndim == 3 else sensor["Ey"]
    I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(I.T, origin="lower", cmap="inferno", aspect="auto")
    plt.colorbar(im, ax=ax, label="|E|² (peak)")
    ax.set_title(f"{sensor_key} — 2D DFT intensity  (max {I.max():.3f}, shape {I.shape})")
    ax.set_xlabel("x cell index"); ax.set_ylabel("y cell index")
    fig.tight_layout()
    out = folder / "figures" / f"{sensor_key}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def plot_sensor(folder: Path, sensor_key: str, sensor: dict, sim_args: dict) -> Path:
    """1D intensity I(y) at a named 1Ddft sensor."""
    Ex = sensor["Ex"][0] if sensor["Ex"].ndim == 2 else sensor["Ex"]
    Ey = sensor["Ey"][0] if sensor["Ey"].ndim == 2 else sensor["Ey"]
    I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    sy = float(sim_args[sensor_key]["position"]["size"])
    y = np.linspace(-sy / 2, sy / 2, I.size)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(y, I, lw=1.8, color="C0")
    ax.set_xlabel("y (µm)")
    ax.set_ylabel("|E|² (FDTD units)")
    ax.set_title(f"{sensor_key} I(y)  —  peak {I.max():.3f}, "
                 f"integral {I.sum() * (y[1] - y[0]):.3f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = folder / "figures" / f"{sensor_key}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def plot_all(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    with open(folder / "simulation_data.json") as f:
        sim_args = json.load(f)
    data = _load_voltage(folder)
    written: list[Path] = []
    written.append(plot_potential(folder, data))
    written.append(plot_e_vectors(folder, data))
    written.append(plot_e_amplitude(folder, data))
    written.append(plot_e_angle(folder, data))
    written.append(plot_director(folder, data))
    # 1Ddft sensors → I(y) line plot
    sensor_kv = _load_sensor(folder, sim_args)
    if sensor_kv is not None:
        sensor_key, sensor_data = sensor_kv
        written.append(plot_sensor(folder, sensor_key, sensor_data, sim_args))
    else:
        print(f"[plot_voltage_reservoir] no 1Ddft sensor npz found in simulation/ — skipped I(y)")
    # 2Ddft sensors → 2D intensity heatmap, one PNG per monitor.
    for key in sim_args.get("object_order", []):
        obj = sim_args.get(key, {})
        if obj.get("class") == "monitor" and obj.get("type") == "2Ddft":
            p = folder / "simulation" / f"{key}.npz"
            if p.exists():
                written.append(plot_reservoir_dft(folder, key, dict(np.load(p))))
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True, help="design folder")
    args = ap.parse_args()
    for p in plot_all(args.path):
        print(f"saved {p}")
