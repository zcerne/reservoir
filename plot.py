import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _monitor_size_y(mon_args, sim_args):
    pos = mon_args.get("position", {})
    if isinstance(pos, dict):
        req = float(pos.get("size", 0))
    else:
        req = 0.0
    return req if req > 0 else float(sim_args.get("cell_size_y", 10.0))


def plot_flux_monitors(sim_args, sim_dir, fig_dir):
    monitors = {}
    for key in sim_args.get("object_order", []):
        mon = sim_args.get(key)
        if mon is None or mon.get("class") != "monitor":
            continue
        if mon.get("type") != "flux":
            continue
        npz_path = os.path.join(sim_dir, f"{key}.npz")
        if not os.path.exists(npz_path):
            print(f"No data for {key}, skipping")
            continue
        monitors[key] = np.load(npz_path)

    if len(monitors) < 2:
        return

    keys = list(monitors.keys())
    d0   = monitors[keys[0]]
    d1   = monitors[keys[1]]
    lams  = 1.0 / np.array(d0["freqs"]) * 1000
    f0    = np.abs(np.array(d0["fluxes"]))
    f1    = np.abs(np.array(d1["fluxes"]))
    ratio = np.where(f0 > 0, f1 / f0 * 100.0, np.nan)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(lams, ratio, color="royalblue")
    ax.axhline(100.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("λ (nm)")
    ax.set_ylabel(f"T = {keys[1]} / {keys[0]}  (%)")
    ax.set_title("Transmission")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(fig_dir, "monitors_flux.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def _compute_T(sim_args, sim_dir):
    flux_keys = [k for k in sim_args.get("object_order", [])
                 if sim_args.get(k, {}).get("class") == "monitor"
                 and sim_args.get(k, {}).get("type") == "flux"
                 and os.path.exists(os.path.join(sim_dir, f"{k}.npz"))]
    if len(flux_keys) < 2:
        return None
    f0 = np.abs(np.load(os.path.join(sim_dir, f"{flux_keys[0]}.npz"))["fluxes"])
    f1 = np.abs(np.load(os.path.join(sim_dir, f"{flux_keys[1]}.npz"))["fluxes"])
    return float(np.mean(np.where(f0 > 0, f1 / f0 * 100.0, np.nan)))


def _load_dft_curves(sim_args, sim_dir):
    """Return list of (key, y, I[0], lam_nm) for all DFT monitors found."""
    curves = []
    for key in sim_args.get("object_order", []):
        mon = sim_args.get(key)
        if mon is None or mon.get("class") != "monitor":
            continue
        if mon.get("type") not in ("1Ddft", "2Ddft"):
            continue
        npz_path = os.path.join(sim_dir, f"{key}.npz")
        if not os.path.exists(npz_path):
            continue
        data   = np.load(npz_path)
        freqs  = data["freqs"]
        I      = (np.abs(data["Ex"]) ** 2
                  + np.abs(data["Ey"]) ** 2
                  + np.abs(data["Ez"]) ** 2)
        size_y = _monitor_size_y(mon, sim_args)
        y      = np.linspace(-size_y / 2, size_y / 2, I.shape[-1])
        lam_nm = 1.0 / freqs[0] * 1000
        curves.append((key, y, I[0], lam_nm))
    return curves


def _load_dft_components(sim_args, sim_dir):
    """Return list of (key, y, Ex2, Ey2, Ez2, lam_nm) for all DFT monitors."""
    out = []
    for key in sim_args.get("object_order", []):
        mon = sim_args.get(key)
        if mon is None or mon.get("class") != "monitor":
            continue
        if mon.get("type") not in ("1Ddft", "2Ddft"):
            continue
        npz_path = os.path.join(sim_dir, f"{key}.npz")
        if not os.path.exists(npz_path):
            continue
        data   = np.load(npz_path)
        freqs  = data["freqs"]
        Ex2    = np.abs(data["Ex"][0]) ** 2
        Ey2    = np.abs(data["Ey"][0]) ** 2
        Ez2    = np.abs(data["Ez"][0]) ** 2
        size_y = _monitor_size_y(mon, sim_args)
        y      = np.linspace(-size_y / 2, size_y / 2, Ex2.shape[-1])
        lam_nm = 1.0 / freqs[0] * 1000
        out.append((key, y, Ex2, Ey2, Ez2, lam_nm))
    return out


def plot_dft_components(sim_args, sim_dir, fig_dir, sim_dir_empty=None):
    lc_curves  = _load_dft_components(sim_args, sim_dir)
    air_curves = _load_dft_components(sim_args, sim_dir_empty) if sim_dir_empty else []
    if not lc_curves and not air_curves:
        return

    lam_nm = (lc_curves[0] if lc_curves else air_curves[0])[5]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    comp_labels = ["|Ex|²", "|Ey|²", "|Ez|²"]

    color_idx = 0
    for key, y, Ex2, Ey2, Ez2 in [(k, y, a, b, c) for k, y, a, b, c, _ in lc_curves]:
        c = colors[color_idx % len(colors)]
        for ax, comp in zip(axes, [Ex2, Ey2, Ez2]):
            ax.plot(y, comp, color=c, label=f"LC  {key}")
        color_idx += 1
    for key, y, Ex2, Ey2, Ez2 in [(k, y, a, b, c) for k, y, a, b, c, _ in air_curves]:
        c = colors[color_idx % len(colors)]
        for ax, comp in zip(axes, [Ex2, Ey2, Ez2]):
            ax.plot(y, comp, color=c, linestyle="--", label=f"air  {key}")
        color_idx += 1

    for ax, label in zip(axes, comp_labels):
        ax.set_xlabel("y (µm)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(f"DFT field components  —  λ = {lam_nm:.1f} nm", y=1.01)
    fig.tight_layout()
    out = os.path.join(fig_dir, "monitors_components.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_dft_monitors(sim_args, sim_dir, fig_dir, sim_dir_empty=None):
    lc_curves  = _load_dft_curves(sim_args, sim_dir)
    air_curves = _load_dft_curves(sim_args, sim_dir_empty) if sim_dir_empty else []
    if not lc_curves and not air_curves:
        return

    lam_nm = (lc_curves[0] if lc_curves else air_curves[0])[3]
    T_lc  = _compute_T(sim_args, sim_dir)
    T_air = _compute_T(sim_args, sim_dir_empty) if sim_dir_empty else None

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(7, 4))
    color_idx = 0
    for key, y, I, lam in lc_curves:
        ax.plot(y, I, color=colors[color_idx % len(colors)], label=f"LC  {key}")
        color_idx += 1
    for key, y, I, lam in air_curves:
        ax.plot(y, I, color=colors[color_idx % len(colors)],
                linestyle="--", label=f"air  {key}")
        color_idx += 1

    title_parts = [f"DFT intensity  —  λ = {lam_nm:.1f} nm"]
    if T_lc is not None:
        title_parts.append(f"T_LC = {T_lc:.1f}%")
    if T_air is not None:
        title_parts.append(f"T_air = {T_air:.1f}%")
    ax.set_title("  |  ".join(title_parts))
    ax.set_xlabel("y (µm)")
    ax.set_ylabel("Intensity |E|²")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(fig_dir, "monitors_intensity.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_2dsnap_grid(sim_args, sim_dir, fig_dir, n, suffix=""):
    for key in sim_args.get("object_order", []):
        mon = sim_args.get(key)
        if mon is None or mon.get("class") != "monitor":
            continue
        if mon.get("type") != "2Dsnap":
            continue

        npz_path = os.path.join(sim_dir, f"{key}.npz")
        if not os.path.exists(npz_path):
            print(f"No data for {key}, skipping")
            continue

        data = np.load(npz_path)
        t  = data["t"]
        Ez = data["Ez"]
        Ex = data["Ex"] if "Ex" in data else np.zeros_like(Ez)
        Ey = data["Ey"] if "Ey" in data else np.zeros_like(Ez)
        I  = Ex ** 2 + Ey ** 2 + Ez ** 2

        n_snap  = I.shape[0]
        n2      = min(n * n, n_snap)
        indices = np.round(np.linspace(0, n_snap - 1, n2)).astype(int)
        ncols   = min(n, n2)
        nrows   = (n2 + ncols - 1) // ncols

        is_1d = Ex.ndim == 2
        vmax  = float(np.percentile(I, 99)) if I.max() > 0 else 1.0

        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)

        for idx in range(nrows * ncols):
            i   = idx // ncols
            j   = idx % ncols
            ax  = axes[i, j]
            if idx < n2:
                snap_idx = indices[idx]
                if is_1d:
                    ax.plot(I[snap_idx])
                    ax.set_ylim(0, vmax)
                else:
                    ax.imshow(I[snap_idx].T, origin="lower", cmap="inferno",
                              vmin=0, vmax=vmax, aspect="auto")
                    ax.axis("off")
                ax.set_title(f"t={t[snap_idx]:.0f}", fontsize=6)
            else:
                ax.axis("off")

        fig.suptitle(f"{key}  —  I = |Ex|² + |Ey|² + |Ez|²", y=1.01)
        fig.tight_layout()
        out = os.path.join(fig_dir, f"{key}_snapshots{suffix}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


def _quiver_slice(ax, a0, a1, u, v, color, label0, label1, title, stride=1):
    aa, bb = np.meshgrid(a0[::stride], a1[::stride], indexing="ij")
    c = ax.quiver(aa, bb, u[::stride, ::stride], v[::stride, ::stride],
                  color[::stride, ::stride], cmap="hsv", clim=(-np.pi, np.pi),
                  pivot="mid", scale=None, headlength=0, headwidth=0, headaxislength=0)
    ax.set_xlabel(label0)
    ax.set_ylabel(label1)
    ax.set_title(title, fontsize=7)
    ax.set_aspect("equal")
    return c


def plot_lc_field_slices(sim_dir, fig_dir, n=3, stride=2):
    npz_path = os.path.join(sim_dir, "lc_fields.npz")
    if not os.path.exists(npz_path):
        print("No lc_fields.npz, skipping LC field plot")
        return

    data  = np.load(npz_path)
    phi   = data["phi"]    # (nx, ny, nz)
    theta = data["theta"]
    x, y, z = data["x"], data["y"], data["z"]

    nd = np.sin(theta) * np.cos(phi)  # nx component
    ne = np.sin(theta) * np.sin(phi)  # ny component
    nf = np.cos(theta)                # nz component

    is_2d = phi.shape[2] <= 5

    if is_2d:
        iz = phi.shape[2] // 2
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        # Panel 1: director field, all-black lines
        aa, bb = np.meshgrid(x[::stride], y[::stride], indexing="ij")
        axes[0].quiver(aa, bb,
                       nd[:, :, iz][::stride, ::stride],
                       ne[:, :, iz][::stride, ::stride],
                       color="black", pivot="mid", scale=None,
                       headlength=0, headwidth=0, headaxislength=0)
        axes[0].set_xlabel("x (µm)")
        axes[0].set_ylabel("y (µm)")
        axes[0].set_title(f"Director field — XY  z={z[iz]:.2f} µm")
        axes[0].set_aspect("equal")
        # Panel 2: phi imshow
        im_phi = axes[1].imshow(phi[:, :, iz].T, origin="lower",
                                extent=[x[0], x[-1], y[0], y[-1]],
                                cmap="hsv", vmin=-np.pi, vmax=np.pi, aspect="equal")
        axes[1].set_xlabel("x (µm)")
        axes[1].set_ylabel("y (µm)")
        axes[1].set_title(f"φ — XY  z={z[iz]:.2f} µm")
        plt.colorbar(im_phi, ax=axes[1], label="φ (rad)")
        # Panel 3: theta imshow
        im_th = axes[2].imshow(theta[:, :, iz].T, origin="lower",
                               extent=[x[0], x[-1], y[0], y[-1]],
                               cmap="plasma", vmin=0, vmax=np.pi / 2, aspect="equal")
        axes[2].set_xlabel("x (µm)")
        axes[2].set_ylabel("y (µm)")
        axes[2].set_title(f"θ — XY  z={z[iz]:.2f} µm")
        plt.colorbar(im_th, ax=axes[2], label="θ (rad)")
        fig.tight_layout()
        out = os.path.join(fig_dir, "lc_field_slices.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")
        return

    # 3D: rows = XY/XZ/YZ planes, 2 col-groups (phi quiver | theta imshow) × n slices
    planes = {
        "XY": (x, y, nd, ne, phi, theta, "x (µm)", "y (µm)", z, 2),
        "XZ": (x, z, nd, nf, phi, theta, "x (µm)", "z (µm)", y, 1),
        "YZ": (y, z, ne, nf, phi, theta, "y (µm)", "z (µm)", x, 0),
    }
    fig, axes = plt.subplots(3, 2 * n, figsize=(3 * 2 * n, 9))
    axes = np.array(axes).reshape(3, 2 * n)
    last_phi_c = last_th_im = None
    for row, (plane, (a0, a1, u3d, v3d, phi3d, th3d, la, lb, perp, ax_idx)) in enumerate(planes.items()):
        indices = np.round(np.linspace(0, perp.size - 1, n)).astype(int)
        for col, idx in enumerate(indices):
            sl = [slice(None)] * 3
            sl[ax_idx] = idx
            last_phi_c = _quiver_slice(axes[row, col],
                                       a0, a1,
                                       u3d[tuple(sl)], v3d[tuple(sl)], phi3d[tuple(sl)],
                                       la, lb,
                                       f"{plane} φ  {perp[idx]:.2f} µm", stride)
            last_th_im = axes[row, n + col].imshow(
                th3d[tuple(sl)].T, origin="lower",
                extent=[a0[0], a0[-1], a1[0], a1[-1]],
                cmap="plasma", vmin=0, vmax=np.pi / 2, aspect="equal")
            axes[row, n + col].set_xlabel(la)
            axes[row, n + col].set_ylabel(lb)
            axes[row, n + col].set_title(f"{plane} θ  {perp[idx]:.2f} µm", fontsize=7)
    if last_phi_c is not None:
        fig.colorbar(last_phi_c, ax=axes[:, :n], label="φ (rad)", shrink=0.6)
    if last_th_im is not None:
        fig.colorbar(last_th_im, ax=axes[:, n:], label="θ (rad)", shrink=0.6)
    fig.tight_layout()
    out = os.path.join(fig_dir, "lc_field_slices.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_snapshots(sim_dir, fig_dir):
    npz_path = os.path.join(sim_dir, "snapshots.npz")
    if not os.path.exists(npz_path):
        return
    data = np.load(npz_path)
    Ex = data["Ex"]
    Ey = data["Ey"]
    Ez = data["Ez"]
    t  = data["t"] if "t" in data else np.arange(Ex.shape[0], dtype=float)
    I  = Ex ** 2 + Ey ** 2 + Ez ** 2  # shape (n_snap, nx, ny) or (n_snap, nx, ny, nz)

    is_3d = I.ndim == 4
    if is_3d:
        iz = I.shape[3] // 2  # central z slice for display
        I_plot = I[:, :, :, iz]
    else:
        I_plot = I

    n_snap = I_plot.shape[0]
    ncols  = min(5, n_snap)
    nrows  = (n_snap + ncols - 1) // ncols
    vmax   = float(np.percentile(I_plot, 99.5)) if I_plot.max() > 0 else 1.0

    subtitle = "central z-slice" if is_3d else ""
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx in range(nrows * ncols):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        if idx < n_snap:
            ax.imshow(I_plot[idx].T, origin="lower", cmap="inferno",
                      vmin=0, vmax=vmax, aspect="auto")
            ax.set_title(f"t={t[idx]:.2f}", fontsize=7)
            ax.axis("off")
        else:
            ax.axis("off")

    title = f"Snapshots  —  I = |Ex|² + |Ey|² + |Ez|²"
    if subtitle:
        title += f"  ({subtitle})"
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    out = os.path.join(fig_dir, "snapshots_evolution.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main(folder, n):
    json_path     = os.path.join(folder, "simulation_data.json")
    sim_dir       = os.path.join(folder, "simulation")
    sim_dir_empty = os.path.join(folder, "simulation_empty")
    fig_dir       = os.path.join(folder, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    with open(json_path) as f:
        sim_args = json.load(f)

    empty_exists = os.path.isdir(sim_dir_empty)
    plot_dft_monitors(sim_args, sim_dir, fig_dir,
                      sim_dir_empty=sim_dir_empty if empty_exists else None)
    plot_dft_components(sim_args, sim_dir, fig_dir)
    plot_2dsnap_grid(sim_args, sim_dir, fig_dir, n)
    plot_lc_field_slices(sim_dir, fig_dir)
    plot_snapshots(sim_dir, fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/test")
    parser.add_argument("--n", type=int, default=5,
                        help="n×n grid for snapshot plots")
    args = parser.parse_args()
    main(args.path, args.n)
