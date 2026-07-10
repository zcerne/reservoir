"""Compute and save the full 3x3 epsilon tensor on the FDTD grid from a relaxed
LC field. Both engines then read this identical grid — no per-engine
interpolation differences.

Produces: <design>/simulation/epsilon.npz with keys:
  ixx_Ex, ixy_Ex, ixz_Ex   — ε_inv at Ex-face (i+1/2, j)
  ixy_Ey, iyy_Ey, iyz_Ey   — ε_inv at Ey-face (i, j+1/2)
  ixz_nd, iyz_nd, izz_nd   — ε_inv at node (i, j)
  n_max                    — max refractive index (for Courant dt)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_eps_grid(design_path, Nx, Ny, dx, cell_x, cell_y):
    """Compute the full 3x3 epsilon-inverse on the FDTD grid at all three Yee
    locations. Returns a dict ready for np.savez."""
    import json
    with open(os.path.join(design_path, "simulation_data.json")) as f:
        cfg = json.load(f)

    lc = np.load(os.path.join(design_path, "simulation", "lc_fields.npz"))
    mid = lc["phi"].shape[2] // 2
    phi_lc = np.asarray(lc["phi"])[:, :, mid]
    theta_lc = np.asarray(lc["theta"])[:, :, mid]
    lc_x = np.asarray(lc["x"]); lc_y = np.asarray(lc["y"])

    # Find reservoir object
    res = next((cfg[k] for k in cfg["object_order"]
                if cfg[k].get("class") in ("reservoir", "voltage_reservoir")), None)
    n_o = float(res.get("n_o", 1.52))
    n_e = float(res.get("n_e", 1.71))

    # Reservoir position (MEEP coords, like _update_all_args computes)
    pml = float(cfg.get("pml_size", 2.0))
    sizes = res.get("sizes", [5.0, 6.0])
    res_sx = float(sizes[0]); res_sy = float(sizes[1])

    # Walk objects to find reservoir x position
    current_x = 0.0
    res_x_local = None
    for key in cfg["object_order"]:
        obj = cfg[key]
        sx = float(obj.get("sizes", [0])[0]) if "sizes" in obj else 0.0
        if key == "mirror_1" or key == "mirror_2":
            lam = float(obj["lam"]); indices = obj.get("n_indexes", [1.0, 1.0])
            T = float(obj.get("transmission", 0.1))
            n_lays = _mirror_n_layers(T, indices)
            sx = sum(lam / 4.0 / float(indices[i % 2]) for i in range(n_lays))
        if key == next((k for k in cfg["object_order"]
                        if cfg[k].get("class") in ("reservoir", "voltage_reservoir")), None):
            res_x_local = current_x
        current_x += float(sx)

    total_x = current_x + 2 * pml
    x0 = -total_x / 2 + pml
    res_x_lo = res_x_local + x0
    res_x_hi = res_x_lo + res_sx
    res_y_lo = -res_sy / 2; res_y_hi = res_sy / 2

    # Interpolators for phi and theta
    from scipy.interpolate import RectBivariateSpline
    # LC local coords shifted to MEEP coords
    lc_x_m = (lc_x - lc_x.min()) + res_x_lo
    lc_y_m = (lc_y - lc_y.min()) + res_y_lo
    interp_phi = RectBivariateSpline(lc_x_m, lc_y_m, phi_lc, kx=3, ky=3)
    interp_theta = RectBivariateSpline(lc_x_m, lc_y_m, theta_lc, kx=3, ky=3)

    eps_perp = n_o ** 2; delta = n_e ** 2 - n_o ** 2

    def eps_at(x_pos, y_pos, frac):
        """Full 3x3 epsilon from director, blended by fill-fraction at this Yee position."""
        phi = interp_phi(x_pos, y_pos, grid=False)
        theta = interp_theta(x_pos, y_pos, grid=False)
        nx = np.sin(theta) * np.cos(phi); ny = np.sin(theta) * np.sin(phi); nz = np.cos(theta)
        exx = eps_perp + delta * nx * nx; eyy = eps_perp + delta * ny * ny
        ezz = eps_perp + delta * nz * nz
        exy = delta * nx * ny; exz = delta * nx * nz; eyz = delta * ny * nz
        # blend with vacuum by fill-fraction
        return (frac * exx + (1 - frac), frac * eyy + (1 - frac), frac * ezz + (1 - frac),
                frac * exy, frac * exz, frac * eyz)

    def sample(y_off, x_off=None, use_x=True):
        """Sample epsilon at Yee positions. x_off = half-cell offset in x; y_off in y."""
        i = np.arange(Nx); j = np.arange(Ny)
        xp = (i + (x_off if x_off else 0)) * dx - cell_x / 2
        yp = (j + y_off) * dx - cell_y / 2
        # area-fraction overlap with reservoir
        cxlo = xp - 0.5 * dx; cxhi = xp + 0.5 * dx
        cylo = yp - 0.5 * dx; cyhi = yp + 0.5 * dx
        ox = np.clip(np.minimum(cxhi, res_x_hi) - np.maximum(cxlo, res_x_lo), 0, dx) / dx
        oy = np.clip(np.minimum(cyhi, res_y_hi) - np.maximum(cylo, res_y_lo), 0, dx) / dx
        frac = (ox[:, None] * oy[None, :]).astype(np.float64)
        xe = np.clip(xp, lc_x_m[0], lc_x_m[-1])
        ye = np.clip(yp, lc_y_m[0], lc_y_m[-1])
        return eps_at(xe[:, None], ye[None, :], frac)

    # Sample at Ex-face (i+1/2, j): x_off=0.5, y_off=0
    eEx = sample(0.0, 0.5)
    # Ey-face (i, j+1/2): x_off=0, y_off=0.5
    eEy = sample(0.5, 0.0)
    # Node (i, j): x_off=0, y_off=0
    end = sample(0.0, 0.0)

    def inv3(e6):
        xx, yy, zz, xy, xz, yz = e6
        det = xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)
        return ((yy * zz - yz * yz) / det, (xx * zz - xz * xz) / det,
                (xx * yy - xy * xy) / det, (xz * yz - xy * zz) / det,
                (xy * yz - xz * yy) / det, (xz * xy - xx * yz) / det)

    ixx_Ex, _, _, ixy_Ex, ixz_Ex, _ = inv3(eEx)
    _, iyy_Ey, _, ixy_Ey, _, iyz_Ey = inv3(eEy)
    _, _, izz_nd, _, ixz_nd, iyz_nd = inv3(end)

    n_max = float(np.sqrt(max(max(a.max() for a in e) for e in (eEx, eEy, end))))

    return dict(ixx_Ex=ixx_Ex, ixy_Ex=ixy_Ex, ixz_Ex=ixz_Ex,
                ixy_Ey=ixy_Ey, iyy_Ey=iyy_Ey, iyz_Ey=iyz_Ey,
                ixz_nd=ixz_nd, iyz_nd=iyz_nd, izz_nd=izz_nd, n_max=n_max)


def _mirror_n_layers(T, indices):
    n_H = float(max(indices)); n_L = float(min(indices))
    if n_H <= n_L: return 2
    sqrtR = np.sqrt(1.0 - T); rho = (1 + sqrtR) / (1 - sqrtR)
    return 2 * max(1, int(np.ceil(np.log(rho) / (2 * np.log(n_H / n_L)))))
