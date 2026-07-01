import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # find root core modules
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ── helpers ───────────────────────────────────────────────────────────────────

def _compute_cell_x(sim_args):
    pml = 2.0 * float(sim_args.get("pml_size", 2.0))
    obj_sx = sum(
        float(sim_args[k]["sizes"][0])
        for k in sim_args.get("object_order", [])
        if isinstance(sim_args.get(k), dict)
        and sim_args[k].get("class") in ("guide", "reservoir")
        and "sizes" in sim_args[k]
    )
    return pml + obj_sx


def _monitor_size_y(mon_args, sim_args):
    pos = mon_args.get("position", {})
    req = float(pos.get("size", 0)) if isinstance(pos, dict) else 0.0
    return req if req > 0 else float(sim_args.get("cell_size_y", 10.0))


def _compute_T(sim_args, sim_dir):
    if not sim_dir:
        return None
    flux_keys = [k for k in sim_args.get("object_order", [])
                 if sim_args.get(k, {}).get("class") == "monitor"
                 and sim_args.get(k, {}).get("type") == "flux"
                 and os.path.exists(os.path.join(sim_dir, f"{k}.npz"))]
    if len(flux_keys) < 2:
        return None
    f0 = np.abs(np.load(os.path.join(sim_dir, f"{flux_keys[0]}.npz"))["fluxes"])
    f1 = np.abs(np.load(os.path.join(sim_dir, f"{flux_keys[1]}.npz"))["fluxes"])
    return float(np.mean(np.where(f0 > 0, f1 / f0 * 100.0, np.nan)))


def _iter_dft(sim_args, sim_dir):
    """Yield (key, y, I, Ex2, Ey2, Ez2, lam_nm) for all guide-face DFT monitors."""
    if not sim_dir:
        return
    for key in sim_args.get("object_order", []):
        mon = sim_args.get(key)
        if mon is None or mon.get("class") != "monitor":
            continue
        if mon.get("type") not in ("1Ddft", "2Ddft"):
            continue
        if "on_object" not in mon:
            continue
        npz = os.path.join(sim_dir, f"{key}.npz")
        if not os.path.exists(npz):
            continue
        d = np.load(npz)

        def _comp(c):
            a = np.atleast_1d(d[c][0])   # drop freq axis; scalar → 1-elem array
            if a.ndim == 2: a = a.sum(-1) # 3-D monitor: integrate over z
            return np.abs(a) ** 2

        Ex2, Ey2, Ez2 = _comp("Ex"), _comp("Ey"), _comp("Ez")
        # degenerate component (e.g. Ez in 2D TE sim stored as scalar) → zeros
        n = max(Ex2.size, Ey2.size, Ez2.size)
        if Ex2.size < n: Ex2 = np.zeros(n)
        if Ey2.size < n: Ey2 = np.zeros(n)
        if Ez2.size < n: Ez2 = np.zeros(n)

        I = Ex2 + Ey2 + Ez2
        size_y = _monitor_size_y(mon, sim_args)
        y = np.linspace(-size_y / 2, size_y / 2, n)
        lam_nm = 1e3 / d["freqs"][0]
        yield key, y, I, Ex2, Ey2, Ez2, lam_nm


# ── 1-D intensity (2D sims) ───────────────────────────────────────────────────

def plot_intensity(sim_args, sim_dir, fig_dir, sim_dir_empty=None):
    """I(y) for all guide-face DFT monitors; LC solid, air dashed."""
    lc  = list(_iter_dft(sim_args, sim_dir))
    air = list(_iter_dft(sim_args, sim_dir_empty)) if sim_dir_empty else []
    if not lc and not air:
        return
    lam_nm = (lc[0] if lc else air[0])[6]
    T_lc   = _compute_T(sim_args, sim_dir)
    T_air  = _compute_T(sim_args, sim_dir_empty)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(7, 4))
    for ci, (key, y, I, *_) in enumerate(lc):
        ax.plot(y, I, color=colors[ci % len(colors)], label=f"LC  {key}")
    for ci, (key, y, I, *_) in enumerate(air):
        ax.plot(y, I, color=colors[ci % len(colors)], linestyle="--", label=f"air  {key}")
    parts = [f"I(y)  —  λ = {lam_nm:.1f} nm"]
    if T_lc  is not None: parts.append(f"T_LC = {T_lc:.1f}%")
    if T_air is not None: parts.append(f"T_air = {T_air:.1f}%")
    ax.set_title("  |  ".join(parts))
    ax.set_xlabel("y (µm)"); ax.set_ylabel("|E|²")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(fig_dir, "intensity.png")
    fig.savefig(out, dpi=150); plt.close(fig); print(f"Saved {out}")


def plot_components(sim_args, sim_dir, fig_dir, sim_dir_empty=None):
    """|Ex|², |Ey|², |Ez|² for all guide-face DFT monitors."""
    lc  = list(_iter_dft(sim_args, sim_dir))
    air = list(_iter_dft(sim_args, sim_dir_empty)) if sim_dir_empty else []
    if not lc and not air:
        return
    lam_nm = (lc[0] if lc else air[0])[6]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ci, (key, y, _, Ex2, Ey2, Ez2, _) in enumerate(lc):
        c = colors[ci % len(colors)]
        for ax, comp in zip(axes, [Ex2, Ey2, Ez2]):
            ax.plot(y, comp, color=c, label=f"LC  {key}")
    for ci, (key, y, _, Ex2, Ey2, Ez2, _) in enumerate(air):
        c = colors[ci % len(colors)]
        for ax, comp in zip(axes, [Ex2, Ey2, Ez2]):
            ax.plot(y, comp, color=c, linestyle="--", label=f"air  {key}")
    for ax, label in zip(axes, ["|Ex|²", "|Ey|²", "|Ez|²"]):
        ax.set_xlabel("y (µm)"); ax.set_ylabel(label); ax.set_title(label)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f"Field components  —  λ = {lam_nm:.1f} nm", y=1.01)
    fig.tight_layout()
    out = os.path.join(fig_dir, "components.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"Saved {out}")


# ── 2-D field imshow ──────────────────────────────────────────────────────────

def plot_field_2d(sim_args, sim_dir, fig_dir, sim_dir_empty=None):
    """2D imshow for all 2Ddft monitors (snap_full and 3D guide-face monitors)."""
    dft_keys = [k for k in sim_args.get("object_order", [])
                if sim_args.get(k, {}).get("class") == "monitor"
                and sim_args.get(k, {}).get("type") == "2Ddft"
                and os.path.exists(os.path.join(sim_dir, f"{k}.npz"))]
    if not dft_keys:
        return
    cell_z = float(sim_args.get("cell_size_z", 0.0))
    cell_y = float(sim_args.get("cell_size_y", 10.0))
    cell_x = _compute_cell_x(sim_args)
    T_lc   = _compute_T(sim_args, sim_dir)
    T_air  = _compute_T(sim_args, sim_dir_empty)

    fig, axes = plt.subplots(len(dft_keys), 3,
                             figsize=(12, 4 * len(dft_keys)), squeeze=False)
    for row, key in enumerate(dft_keys):
        mon  = sim_args[key]
        data = np.load(os.path.join(sim_dir, f"{key}.npz"))
        if "on_object" not in mon and cell_z == 0.0:
            extent = [-cell_x / 2, cell_x / 2, -cell_y / 2, cell_y / 2]
            xlabel, ylabel = "x (µm)", "y (µm)"
        else:
            size_y = _monitor_size_y(mon, sim_args)
            extent = [-size_y / 2, size_y / 2, -cell_z / 2, cell_z / 2]
            xlabel, ylabel = "y (µm)", "z (µm)"
        comps = [np.abs(data["Ex"][0]) ** 2,
                 np.abs(data["Ey"][0]) ** 2,
                 np.abs(data["Ez"][0]) ** 2]
        vmax  = max(float(c.max()) for c in comps) or 1.0
        for col, (comp, label) in enumerate(zip(comps, ["|Ex|²", "|Ey|²", "|Ez|²"])):
            ax = axes[row][col]
            im = ax.imshow(comp.T, origin="lower", cmap="inferno",
                           vmin=0, vmax=vmax, aspect="auto", extent=extent)
            ax.set_title(f"{key}  {label}", fontsize=9)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            plt.colorbar(im, ax=ax, shrink=0.8)

    lam_nm = 1e3 / np.load(os.path.join(sim_dir, f"{dft_keys[0]}.npz"))["freqs"][0]
    parts  = [f"DFT fields  —  λ = {lam_nm:.1f} nm"]
    if T_lc  is not None: parts.append(f"T_LC = {T_lc:.1f}%")
    if T_air is not None: parts.append(f"T_air = {T_air:.1f}%")
    fig.suptitle("  |  ".join(parts), y=1.01)
    fig.tight_layout()
    out = os.path.join(fig_dir, "field_2d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"Saved {out}")


# ── snapshots ─────────────────────────────────────────────────────────────────

def plot_snapshots(sim_dir, fig_dir, sim_args=None):
    npz = os.path.join(sim_dir, "snapshots.npz")
    if not os.path.exists(npz):
        return
    d  = np.load(npz)
    t  = d["t"] if "t" in d else np.arange(d["Ex"].shape[0], dtype=float)
    I  = d["Ex"] ** 2 + d["Ey"] ** 2 + d["Ez"] ** 2
    if I.ndim == 4:
        I = I[:, :, :, I.shape[3] // 2]

    extent = None
    if sim_args is not None:
        cx = _compute_cell_x(sim_args)
        cy = float(sim_args.get("cell_size_y", 0.0))
        extent = [-cx / 2, cx / 2, -cy / 2, cy / 2]

    n_snap = I.shape[0]
    ncols  = min(5, n_snap)
    nrows  = (n_snap + ncols - 1) // ncols
    vmax   = float(np.percentile(I, 99.5)) if I.max() > 0 else 1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for idx in range(nrows * ncols):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        if idx < n_snap:
            ax.imshow(I[idx].T, origin="lower", cmap="inferno",
                      vmin=0, vmax=vmax, aspect="auto", extent=extent)
            ax.set_title(f"t={t[idx]:.2f}", fontsize=7)
            if extent is not None:
                ax.set_xlabel("x (µm)", fontsize=5)
                ax.set_ylabel("y (µm)", fontsize=5)
                ax.tick_params(labelsize=5)
            else:
                ax.axis("off")
        else:
            ax.axis("off")
    fig.suptitle("Snapshots — I = |Ex|²+|Ey|²+|Ez|²", y=1.01)
    fig.tight_layout()
    out = os.path.join(fig_dir, "snapshots.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig); print(f"Saved {out}")


# ── director structure ────────────────────────────────────────────────────────

def _quiver_slice(ax, a0, a1, u, v, color, la, lb, title, stride=1):
    aa, bb = np.meshgrid(a0[::stride], a1[::stride], indexing="ij")
    c = ax.quiver(aa, bb, u[::stride, ::stride], v[::stride, ::stride],
                  color[::stride, ::stride], cmap="hsv", clim=(-np.pi, np.pi),
                  pivot="mid", scale=None, headlength=0, headwidth=0, headaxislength=0)
    ax.set_xlabel(la); ax.set_ylabel(lb)
    ax.set_title(title, fontsize=7); ax.set_aspect("equal")
    return c


def plot_director(sim_dir, fig_dir, n=3, stride=2):
    """Director phi/theta from lc_fields.npz; 2D=3-panel, 3D=slice grid."""
    npz = os.path.join(sim_dir, "lc_fields.npz")
    if not os.path.exists(npz):
        return
    d = np.load(npz)
    phi, theta = d["phi"] % np.pi, d["theta"]
    x, y, z = d["x"], d["y"], d["z"]
    nd = np.sin(theta) * np.cos(phi)
    ne = np.sin(theta) * np.sin(phi)
    nf = np.cos(theta)
    is_2d = phi.shape[2] <= 5

    if is_2d:
        iz  = phi.shape[2] // 2
        fig, axes = plt.subplots(3, 1, figsize=(8, 14))
        aa, bb = np.meshgrid(x[::stride], y[::stride], indexing="ij")
        axes[0].quiver(aa, bb,
                       nd[:, :, iz][::stride, ::stride],
                       ne[:, :, iz][::stride, ::stride],
                       color="black", pivot="mid", scale=None,
                       headlength=0, headwidth=0, headaxislength=0)
        axes[0].set_xlabel("x (µm)"); axes[0].set_ylabel("y (µm)")
        axes[0].set_title(f"Director XY  z={z[iz]:.2f} µm")
        axes[0].set_aspect("equal")
        im0 = axes[1].imshow(phi[:, :, iz].T, origin="lower",
                             extent=[x[0], x[-1], y[0], y[-1]],
                             cmap="hsv", vmin=0, vmax=np.pi, aspect="equal")
        axes[1].set_xlabel("x (µm)"); axes[1].set_ylabel("y (µm)"); axes[1].set_title("φ")
        div0 = make_axes_locatable(axes[1]); cax0 = div0.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im0, cax=cax0, label="φ (rad)")
        im1 = axes[2].imshow(theta[:, :, iz].T, origin="lower",
                             extent=[x[0], x[-1], y[0], y[-1]],
                             cmap="plasma", vmin=0, vmax=np.pi / 2, aspect="equal")
        axes[2].set_xlabel("x (µm)"); axes[2].set_ylabel("y (µm)"); axes[2].set_title("θ")
        div1 = make_axes_locatable(axes[2]); cax1 = div1.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im1, cax=cax1, label="θ (rad)")
    else:
        planes = {
            "XY": (x, y, nd, ne, phi, theta, "x (µm)", "y (µm)", z, 2),
            "XZ": (x, z, nd, nf, phi, theta, "x (µm)", "z (µm)", y, 1),
            "YZ": (y, z, ne, nf, phi, theta, "y (µm)", "z (µm)", x, 0),
        }
        fig, axes = plt.subplots(3, 2 * n, figsize=(3 * 2 * n, 9))
        axes = np.array(axes).reshape(3, 2 * n)
        last_c = last_im = None
        for row, (plane, (a0, a1, u3, v3, p3, t3, la, lb, perp, ax_i)) in enumerate(planes.items()):
            idxs = np.round(np.linspace(0, perp.size - 1, n)).astype(int)
            for col, idx in enumerate(idxs):
                sl = [slice(None)] * 3; sl[ax_i] = idx
                last_c  = _quiver_slice(axes[row, col], a0, a1,
                                        u3[tuple(sl)], v3[tuple(sl)], p3[tuple(sl)],
                                        la, lb, f"{plane} φ  {perp[idx]:.2f} µm", stride)
                last_im = axes[row, n + col].imshow(
                    t3[tuple(sl)].T, origin="lower",
                    extent=[a0[0], a0[-1], a1[0], a1[-1]],
                    cmap="plasma", vmin=0, vmax=np.pi / 2, aspect="equal")
                axes[row, n + col].set_xlabel(la); axes[row, n + col].set_ylabel(lb)
                axes[row, n + col].set_title(f"{plane} θ  {perp[idx]:.2f} µm", fontsize=7)
        if last_c:  fig.colorbar(last_c,  ax=axes[:, :n], label="φ (rad)", shrink=0.6)
        if last_im: fig.colorbar(last_im, ax=axes[:, n:], label="θ (rad)", shrink=0.6)

    fig.tight_layout()
    out = os.path.join(fig_dir, "director.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"Saved {out}")


# ── boundary conditions ───────────────────────────────────────────────────────

def plot_boundary_conditions(sim_dir, fig_dir):
    """Phi and theta at reservoir boundaries from lc_fields.npz."""
    npz = os.path.join(sim_dir, "lc_fields.npz")
    if not os.path.exists(npz):
        return
    d = np.load(npz)
    phi, theta = d["phi"] % np.pi, d["theta"]
    x, y, z = d["x"], d["y"], d["z"]
    is_2d = phi.shape[2] <= 5

    if is_2d:
        iz  = phi.shape[2] // 2
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        for name, vals_phi, vals_theta in [("y_min", phi[:, 0, iz], theta[:, 0, iz]),
                                           ("y_max", phi[:, -1, iz], theta[:, -1, iz])]:
            axes[0].plot(x, vals_phi,  label=name)
            axes[1].plot(x, vals_theta, label=name)
        axes[0].set_ylim(0, np.pi)
        axes[0].set_xlabel("x (µm)"); axes[0].set_ylabel("φ (rad)")
        axes[0].set_title("Boundary φ"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("x (µm)"); axes[1].set_ylabel("θ (rad)")
        axes[1].set_title("Boundary θ"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    else:
        faces = [
            ("x_min", phi[0, :, :],   theta[0, :, :],   y, z, "y (µm)", "z (µm)"),
            ("x_max", phi[-1, :, :],  theta[-1, :, :],  y, z, "y (µm)", "z (µm)"),
            ("y_min", phi[:, 0, :],   theta[:, 0, :],   x, z, "x (µm)", "z (µm)"),
            ("y_max", phi[:, -1, :],  theta[:, -1, :],  x, z, "x (µm)", "z (µm)"),
            ("z_min", phi[:, :, 0],   theta[:, :, 0],   x, y, "x (µm)", "y (µm)"),
            ("z_max", phi[:, :, -1],  theta[:, :, -1],  x, y, "x (µm)", "y (µm)"),
        ]
        fig, axes = plt.subplots(6, 2, figsize=(8, 18))
        for row, (name, pf, tf, a0, a1, la, lb) in enumerate(faces):
            ext = [a0[0], a0[-1], a1[0], a1[-1]]
            im0 = axes[row, 0].imshow(pf.T, origin="lower", extent=ext,
                                      cmap="hsv", vmin=0, vmax=np.pi, aspect="auto")
            axes[row, 0].set_title(f"{name} φ", fontsize=8)
            axes[row, 0].set_xlabel(la); axes[row, 0].set_ylabel(lb)
            plt.colorbar(im0, ax=axes[row, 0], shrink=0.8)
            im1 = axes[row, 1].imshow(tf.T, origin="lower", extent=ext,
                                      cmap="plasma", vmin=0, vmax=np.pi / 2, aspect="auto")
            axes[row, 1].set_title(f"{name} θ", fontsize=8)
            axes[row, 1].set_xlabel(la); axes[row, 1].set_ylabel(lb)
            plt.colorbar(im1, ax=axes[row, 1], shrink=0.8)

    fig.tight_layout()
    out = os.path.join(fig_dir, "boundary_conditions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(folder, n=3):
    sim_dir = os.path.join(folder, "simulation")
    empty   = os.path.join(folder, "simulation_empty")
    fig_dir = os.path.join(folder, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    with open(os.path.join(folder, "simulation_data.json")) as f:
        sim_args = json.load(f)

    # Dispatch to voltage-reservoir plotter for that reservoir class — the old
    # plot.py functions (boundary_conditions / director / etc.) all assume the
    # legacy MEEP-relaxed-LC pipeline and produce nothing useful here.
    res_class = sim_args.get("reservoir", {}).get("class")
    if res_class == "voltage_reservoir":
        from plot_voltage_reservoir import plot_all as _plot_voltage_all
        for p in _plot_voltage_all(folder):
            print(f"saved {p}")
        return

    empty_dir = empty if os.path.isdir(empty) else None
    is_3d = int(sim_args.get("dimention", 1)) == 3

    if is_3d:
        plot_field_2d(sim_args, sim_dir, fig_dir, empty_dir)
    else:
        plot_intensity(sim_args, sim_dir, fig_dir, empty_dir)
        plot_components(sim_args, sim_dir, fig_dir, empty_dir)
        plot_field_2d(sim_args, sim_dir, fig_dir, empty_dir)

    plot_snapshots(sim_dir, fig_dir, sim_args)
    plot_director(sim_dir, fig_dir, n=n)
    plot_boundary_conditions(sim_dir, fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/test")
    parser.add_argument("--n", type=int, default=3,
                        help="director slices shown for 3D plots")
    args = parser.parse_args()
    main(args.path, args.n)
