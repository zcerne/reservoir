import numpy as np
from scipy.ndimage import gaussian_filter


def _smooth_face(rng, shape, sigma):
    """Gaussian-smoothed random field in [0, 1] on a 2D face grid."""
    raw = rng.uniform(0, 1, shape)
    if sigma > 0:
        raw = gaussian_filter(raw, sigma=sigma, mode="wrap")
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo + 1e-12)


def _perlin_1d(rng, n, wavelength):
    """1D Perlin (gradient) noise of length n, in [0,1]. `wavelength` = lattice
    spacing in samples (larger = smoother). Gradient noise → C¹-smooth, no jaggies."""
    L = max(2, int(round(wavelength)))
    n_nodes = n // L + 2
    grad = rng.uniform(-1.0, 1.0, n_nodes)          # random slope at each lattice node
    x = np.arange(n) / L
    i0 = np.floor(x).astype(int); t = x - i0
    def fade(u):                                     # 6u⁵−15u⁴+10u³ (Perlin smootherstep)
        return u * u * u * (u * (u * 6 - 15) + 10)
    g0 = grad[i0]; g1 = grad[i0 + 1]
    d0 = g0 * t; d1 = g1 * (t - 1.0)                 # dot(gradient, distance)
    val = d0 + fade(t) * (d1 - d0)
    lo, hi = val.min(), val.max()
    return (val - lo) / (hi - lo + 1e-12)


def perlin_2d_boundaries(resolution, dimensions, seed=None, ignore_faces=None, scale=3.0):
    """Perlin-noise director anchoring on the 4 side faces of a 2D cell. Smooth,
    continuous, random-but-not-noisy (gradient noise, correlation length `scale` µm).
    Faces: x_min/x_max vary along y (length n_y); y_min/y_max vary along x (n_x)."""
    rng = np.random.default_rng(seed)
    sx, sy = float(dimensions[0]), float(dimensions[1])
    n_x = int(sx * resolution) + 1
    n_y = int(sy * resolution) + 1
    ign = ignore_faces if ignore_faces is not None else [False] * 6
    wl = scale * resolution                          # samples per Perlin lattice cell
    keys = ("x_min", "x_max", "y_min", "y_max")
    sizes = (n_y, n_y, n_x, n_x)
    face_phi, face_theta = {}, {}
    for key, sz, ignored in zip(keys, sizes, ign):
        if ignored:
            face_phi[key] = None; face_theta[key] = None
        else:
            face_phi[key] = _perlin_1d(rng, sz, wl) * np.pi
            # θ anchored IN-PLANE (π/2) — 2D-planar convention; only φ is random.
            face_theta[key] = np.full(sz, np.pi / 2)
    return face_phi, face_theta


def perlin_3d_boundaries(resolution, dimensions, seed=None, ignore_faces=None, scale=2.0,
                         same_opposite_faces=None):
    """Smooth (Perlin-like) boundary conditions on all 6 faces of a 3D LC cube.

    scale: spatial smoothness in µm — larger = smoother director pattern.
    same_opposite_faces: 3-element bool list [same_x, same_y, same_z].
        If True for a pair, the max face copies the min face (identical BCs on both sides).
    """
    rng = np.random.default_rng(seed)
    sx, sy, sz = float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
    n_x = int(sx * resolution) + 1
    n_y = int(sy * resolution) + 1
    n_z = int(sz * resolution) + 1
    ign  = ignore_faces if ignore_faces is not None else [False] * 6
    same = same_opposite_faces if same_opposite_faces is not None else [False, False, False]

    sigma = scale * resolution  # pixels per smoothing length

    # pairs: (min_key, max_key, shape, same_index)
    pairs = [
        ("x_min", "x_max", (n_y, n_z), 0),
        ("y_min", "y_max", (n_x, n_z), 1),
        ("z_min", "z_max", (n_x, n_y), 2),
    ]
    # ign order: x_min, x_max, y_min, y_max, z_min, z_max
    ign_by_pair = [(ign[0], ign[1]), (ign[2], ign[3]), (ign[4], ign[5])]

    face_phi   = {}
    face_theta = {}
    for (k_min, k_max, shape, si), (ign_min, ign_max) in zip(pairs, ign_by_pair):
        # generate min face
        if ign_min:
            phi_min, theta_min = None, None
        else:
            phi_min   = _smooth_face(rng, shape, sigma).ravel() * np.pi
            theta_min = _smooth_face(rng, shape, sigma).ravel() * (np.pi / 2)

        # generate or copy max face
        if ign_max:
            phi_max, theta_max = None, None
        elif same[si] and phi_min is not None:
            phi_max, theta_max = phi_min.copy(), theta_min.copy()
        else:
            phi_max   = _smooth_face(rng, shape, sigma).ravel() * np.pi
            theta_max = _smooth_face(rng, shape, sigma).ravel() * (np.pi / 2)

        face_phi[k_min],   face_theta[k_min]   = phi_min,   theta_min
        face_phi[k_max],   face_theta[k_max]   = phi_max,   theta_max

    return face_phi, face_theta


def sinus_3d_boundaries(resolution, dimensions, seed=None, ignore_faces=None, n_periods=1):
    """Sinusoidal boundary conditions varying along x for each face group.

    xz faces (y_min, y_max): sinusoidal phi, theta free.
    xy faces (z_min, z_max): sinusoidal theta, phi free.
    yz faces (x_min, x_max): always free (set ignore_faces=[T,T,F,F,F,F] in JSON).

    phi  = π/2 + π/2 * sin(2π * n_periods * x/sx)  →  range [0, π]
    theta= π/4 + π/4 * sin(2π * n_periods * x/sx)  →  range [0, π/2]
    """
    sx, sy, sz = float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
    n_x = int(sx * resolution) + 1
    n_y = int(sy * resolution) + 1
    n_z = int(sz * resolution) + 1
    ign = ignore_faces if ignore_faces is not None else [False] * 6

    x_norm = np.linspace(0, 1, n_x)
    phi_along_x   = np.pi / 2 + np.pi / 2 * np.sin(2 * np.pi * n_periods * x_norm)
    theta_along_x = np.pi / 4 + np.pi / 4 * np.sin(2 * np.pi * n_periods * x_norm)

    def _tile_x_along_z(arr_x):
        return np.tile(arr_x[:, None], (1, n_z)).ravel()

    def _tile_x_along_y(arr_x):
        return np.tile(arr_x[:, None], (1, n_y)).ravel()

    keys_ign = list(zip(
        ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], ign))

    face_phi   = {}
    face_theta = {}
    for key, ignored in keys_ign:
        face_phi[key]   = None
        face_theta[key] = None

    # xz faces (y_min, y_max): sinusoidal phi
    if not ign[2]: face_phi["y_min"] = _tile_x_along_z(phi_along_x)
    if not ign[3]: face_phi["y_max"] = _tile_x_along_z(phi_along_x)
    # xy faces (z_min, z_max): sinusoidal theta
    if not ign[4]: face_theta["z_min"] = _tile_x_along_y(theta_along_x)
    if not ign[5]: face_theta["z_max"] = _tile_x_along_y(theta_along_x)

    return face_phi, face_theta


def sinus_2d_boundaries(resolution, dimensions, seed=None, ignore_faces=None,
                        n_periods=1, phase_shift=0.0):
    """Sinusoidal boundary conditions for 2D LC (x_min/x_max free, y_min/y_max sinusoidal phi).

    y_min: phi = π/2 + π/2 * sin(2π * n_periods * x/sx)           → range [0, π]
    y_max: phi = π/2 + π/2 * sin(2π * n_periods * x/sx + phase_shift)

    phase_shift=0   → same pattern top/bottom → pure x-variation in bulk
    phase_shift=π   → opposite patterns       → variation in both x and y
    """
    sx, sy = float(dimensions[0]), float(dimensions[1])
    n_x = int(sx * resolution) + 1
    ign = ignore_faces if ignore_faces is not None else [False] * 6

    x_norm = np.linspace(0, 1, n_x)
    phi_ymin = np.pi / 2 + np.pi / 2 * np.sin(2 * np.pi * n_periods * x_norm)
    phi_ymax = np.pi / 2 + np.pi / 2 * np.sin(2 * np.pi * n_periods * x_norm + phase_shift)

    face_phi   = {"x_min": None, "x_max": None, "y_min": None, "y_max": None}
    face_theta = {"x_min": None, "x_max": None, "y_min": None, "y_max": None}

    if not ign[2]: face_phi["y_min"] = phi_ymin
    if not ign[3]: face_phi["y_max"] = phi_ymax

    return face_phi, face_theta


def sinus_random_2d_boundaries(resolution, dimensions, seed=None, ignore_faces=None,
                               n_periods=1, phase_shift=0.0, noise_level=0.5, scale=10.0):
    """Sinusoidal BCs with superimposed smooth random noise ('defects').

    phi_ymin = clip(pi/2 + pi/2*sin(2pi*n*x/sx) + noise_level*pi*noise(x), 0, pi)
    phi_ymax = same with phase_shift and independent noise

    noise_level: amplitude of noise relative to pi (0=pure sinus, 1=full-range noise)
    scale: smoothness of noise in µm (larger = smoother defects)
    """
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(seed)
    sx, sy = float(dimensions[0]), float(dimensions[1])
    n_x = int(sx * resolution) + 1
    ign = ignore_faces if ignore_faces is not None else [False] * 6

    x_norm = np.linspace(0, 1, n_x)
    phi_ymin_sin = np.pi / 2 + np.pi / 2 * np.sin(2 * np.pi * n_periods * x_norm)
    phi_ymax_sin = np.pi / 2 + np.pi / 2 * np.sin(2 * np.pi * n_periods * x_norm + phase_shift)

    sigma = scale * resolution
    def _smooth_noise_1d(size):
        raw = rng.uniform(0, 1, size)
        if sigma > 0:
            raw = gaussian_filter1d(raw, sigma=sigma, mode='wrap')
        lo, hi = raw.min(), raw.max()
        return (raw - lo) / (hi - lo + 1e-12) * 2 - 1  # normalise to [-1, 1]

    noise_ymin = _smooth_noise_1d(n_x)
    noise_ymax = _smooth_noise_1d(n_x)

    phi_ymin = np.clip(phi_ymin_sin + noise_level * np.pi * noise_ymin, 0, np.pi)
    phi_ymax = np.clip(phi_ymax_sin + noise_level * np.pi * noise_ymax, 0, np.pi)

    face_phi   = {"x_min": None, "x_max": None, "y_min": None, "y_max": None}
    face_theta = {"x_min": None, "x_max": None, "y_min": None, "y_max": None}

    if not ign[2]: face_phi["y_min"] = phi_ymin
    if not ign[3]: face_phi["y_max"] = phi_ymax

    return face_phi, face_theta


def random_3d_boundaries(resolution, dimensions, seed=None, ignore_faces=None):
    rng = np.random.default_rng(seed)
    sx, sy, sz = float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
    n_x = int(sx * resolution) + 1
    n_y = int(sy * resolution) + 1
    n_z = int(sz * resolution) + 1
    ign = ignore_faces if ignore_faces is not None else [False] * 6

    keys = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    sizes = (n_y * n_z, n_y * n_z, n_x * n_z, n_x * n_z, n_x * n_y, n_x * n_y)
    face_phi   = {}
    face_theta = {}
    for key, sz_face, ignored in zip(keys, sizes, ign):
        if ignored:
            face_phi[key]   = None
            face_theta[key] = None
        else:
            face_phi[key]   = rng.uniform(0, np.pi,     sz_face)
            face_theta[key] = rng.uniform(0, np.pi / 2, sz_face)
    return face_phi, face_theta


def competing_3d_boundaries(resolution, dimensions, seed=None, ignore_faces=None, scale=1.0):
    """3D boundaries with competing theta anchoring on opposing face pairs.

    Creates 3D bulk frustration by combining:
    - y pair: y_min biased homeotropic (theta ∈ [0, π/4]),
              y_max biased planar      (theta ∈ [π/4, π/2])
    - z pair: z_min biased planar      (theta ∈ [π/4, π/2]),
              z_max biased homeotropic (theta ∈ [0, π/4])
    - x pair: full theta range [0, π/2] (unbiased)
    - All faces: smooth random phi ∈ [0, π]

    Combined with q0 (cholesteric twist along x), this creates orthogonal theta
    gradients in both y and z, forcing maximally complex 3D bulk director field.

    scale: spatial smoothness of phi/theta variation in µm
    """
    rng = np.random.default_rng(seed)
    sx, sy, sz = float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
    n_x = int(sx * resolution) + 1
    n_y = int(sy * resolution) + 1
    n_z = int(sz * resolution) + 1
    ign = ignore_faces if ignore_faces is not None else [False] * 6
    sigma = scale * resolution

    # face specs: (key, shape, theta_lo, theta_hi)
    specs = [
        ("x_min", (n_y, n_z), 0.0,        np.pi / 2),   # unbiased
        ("x_max", (n_y, n_z), 0.0,        np.pi / 2),   # unbiased
        ("y_min", (n_x, n_z), 0.0,        np.pi / 4),   # homeotropic-biased
        ("y_max", (n_x, n_z), np.pi / 4,  np.pi / 2),   # planar-biased
        ("z_min", (n_x, n_y), np.pi / 4,  np.pi / 2),   # planar-biased
        ("z_max", (n_x, n_y), 0.0,        np.pi / 4),   # homeotropic-biased
    ]
    face_phi   = {}
    face_theta = {}
    for (key, shape), ignored in zip([(s[0], s[1]) for s in specs], ign):
        face_phi[key]   = None
        face_theta[key] = None
    for (key, shape, th_lo, th_hi), ignored in zip(specs, ign):
        if ignored:
            continue
        face_phi[key]   = _smooth_face(rng, shape, sigma).ravel() * np.pi
        face_theta[key] = (th_lo + _smooth_face(rng, shape, sigma).ravel() * (th_hi - th_lo))
    return face_phi, face_theta


def random_2d_boundaries(resolution, dimensions, seed=None, ignore_faces=None):
    rng = np.random.default_rng(seed)
    sx, sy = float(dimensions[0]), float(dimensions[1])
    n_x = int(sx * resolution) + 1  # points along x (used by y-faces)
    n_y = int(sy * resolution) + 1  # points along y (used by x-faces)
    ign = ignore_faces if ignore_faces is not None else [False] * 6

    keys  = ("x_min", "x_max", "y_min", "y_max")
    sizes = (n_y, n_y, n_x, n_x)
    face_phi   = {}
    face_theta = {}
    for key, sz_face, ignored in zip(keys, sizes, ign):
        if ignored:
            face_phi[key]   = None
            face_theta[key] = None
        else:
            face_phi[key]   = rng.uniform(0, np.pi,     sz_face)
            face_theta[key] = rng.uniform(0, np.pi / 2, sz_face)
    return face_phi, face_theta


def smooth_random_2d_boundaries(resolution, dimensions, seed=None, ignore_faces=None,
                                scale=3.0):
    """Large-scale SMOOTH random boundary anchoring on the 4 side faces of a 2D cell.

    Per-pixel white noise (`random_2d`) is essentially uniform for an LC — the
    high-frequency anchoring averages out / can't be resolved by the elastic
    field. This Gaussian-smooths the random face values to a correlation length
    `scale` (µm), giving big smooth director fluctuations the bulk can follow.
    `scale` larger = bigger, smoother patterns.
    """
    rng = np.random.default_rng(seed)
    sx, sy = float(dimensions[0]), float(dimensions[1])
    n_x = int(sx * resolution) + 1
    n_y = int(sy * resolution) + 1
    ign = ignore_faces if ignore_faces is not None else [False] * 6
    sigma = scale * resolution                    # pixels per smoothing length
    keys  = ("x_min", "x_max", "y_min", "y_max")
    sizes = (n_y, n_y, n_x, n_x)
    face_phi, face_theta = {}, {}
    for key, sz_face, ignored in zip(keys, sizes, ign):
        if ignored:
            face_phi[key] = None; face_theta[key] = None
        else:
            face_phi[key]   = _smooth_face(rng, (sz_face,), sigma).ravel() * np.pi
            face_theta[key] = _smooth_face(rng, (sz_face,), sigma).ravel() * (np.pi / 2)
    return face_phi, face_theta


def defect_2d_boundaries(resolution, dimensions, seed=None, ignore_faces=None,
                         scale=1.0, n_periods=0.0):
    """Boundary anchoring with a topological WINDING -> forces a defect inside.

    Sets the in-plane director on the 4 side faces to the azimuthal/spiral pattern
    φ(r) = m·atan2(y−yc, x−xc) + ψ0 about the cell centre (xc,yc). Going once around
    the perimeter the director winds by 2π·m, so a defect of charge m sits inside
    (m=1 → one +1 defect in the middle; Q-tensor melts its S→0 core, the director
    model shows a phase singularity). `scale` = winding m (use 1.0 for one defect),
    `n_periods` = constant spiral offset ψ0/π (0 = radial, 0.5 = azimuthal).
    theta is planar (π/2) everywhere.
    """
    sx, sy = float(dimensions[0]), float(dimensions[1])
    xc, yc = sx / 2.0, sy / 2.0
    m = float(scale); psi0 = float(n_periods) * np.pi
    n_x = int(sx * resolution) + 1
    n_y = int(sy * resolution) + 1
    xs = np.linspace(0.0, sx, n_x)
    ys = np.linspace(0.0, sy, n_y)
    ign = ignore_faces if ignore_faces is not None else [False] * 6

    def az(xpts, ypts):
        return (m * np.arctan2(ypts - yc, xpts - xc) + psi0) % np.pi

    faces = {
        "x_min": (np.full(n_y, 0.0), ys),
        "x_max": (np.full(n_y, sx), ys),
        "y_min": (xs, np.full(n_x, 0.0)),
        "y_max": (xs, np.full(n_x, sy)),
    }
    face_phi, face_theta = {}, {}
    for key, ignored in zip(("x_min", "x_max", "y_min", "y_max"), ign):
        if ignored:
            face_phi[key] = None; face_theta[key] = None
        else:
            xp, yp = faces[key]
            face_phi[key]   = az(xp, yp)
            face_theta[key] = np.full(xp.shape if xp.ndim else yp.shape, np.pi / 2)
    return face_phi, face_theta


if __name__ == "__main__":
    import argparse
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None,
                        help="Simulation folder with simulation_data.json (overrides --dimensions/--resolution)")
    parser.add_argument("--dimensions", nargs=3, type=float, default=[10, 10, 10], metavar=("sx", "sy", "sz"))
    parser.add_argument("--resolution", type=int, default=5)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="boundary_perlin.png")
    args = parser.parse_args()

    if args.path is not None:
        from pathlib import Path
        sim_path = Path(args.path)
        with open(sim_path / "simulation_data.json") as f:
            cfg = json.load(f)["reservoir"]
        dims = list(cfg["sizes"])
        res  = cfg["resolution"]
        if args.scale == 2.0 and "boundary_scale" in cfg:
            args.scale = cfg["boundary_scale"]
        if args.seed is None and cfg.get("boundary_seed") is not None:
            args.seed = cfg["boundary_seed"]
        fig_dir = sim_path / "figures"
        fig_dir.mkdir(exist_ok=True)
        if args.out == "boundary_perlin.png":
            args.out = str(fig_dir / "boundary_perlin.png")
    else:
        dims = args.dimensions
        res  = args.resolution
    sx, sy, sz = float(dims[0]), float(dims[1]), float(dims[2])
    n_x = int(sx * res) + 1
    n_y = int(sy * res) + 1
    n_z = int(sz * res) + 1

    face_phi, face_theta = perlin_3d_boundaries(
        res, dims, seed=args.seed, scale=args.scale)

    # Shape of each face's 2D grid for reshaping the raveled arrays
    face_shapes = {
        "x_min": (n_y, n_z), "x_max": (n_y, n_z),
        "y_min": (n_x, n_z), "y_max": (n_x, n_z),
        "z_min": (n_x, n_y), "z_max": (n_x, n_y),
    }
    # Axis labels per face: (horizontal axis name, vertical axis name)
    face_axes = {
        "x_min": ("z", "y"), "x_max": ("z", "y"),
        "y_min": ("z", "x"), "y_max": ("z", "x"),
        "z_min": ("y", "x"), "z_max": ("y", "x"),
    }

    keys = list(face_shapes.keys())
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    row_data  = [(face_phi, "phi  (0 → π)", "hsv"), (face_theta, "theta  (0 → π/2)", "plasma")]

    for row, (data, row_label, cmap) in enumerate(row_data):
        for col, key in enumerate(keys):
            ax  = axes[row][col]
            arr = data.get(key)
            if arr is None:
                ax.set_facecolor("0.2")
                ax.text(0.5, 0.5, "free", ha="center", va="center",
                        transform=ax.transAxes, color="white", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                shape = face_shapes[key]
                img   = arr.reshape(shape)
                vmax  = np.pi if row == 0 else np.pi / 2
                im = ax.imshow(img, origin="lower", cmap=cmap,
                               vmin=0, vmax=vmax, aspect="auto")
                plt.colorbar(im, ax=ax, shrink=0.8)
                hax, vax = face_axes[key]
                ax.set_xlabel(hax); ax.set_ylabel(vax)
            if row == 0:
                ax.set_title(key, fontsize=9)
        axes[row][0].set_ylabel(f"{row_label}\n{face_axes[keys[0]][1]}")

    fig.suptitle(
        f"perlin_3d_boundaries — dims={dims}, res={res}, scale={args.scale}, seed={args.seed}",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved {args.out}")
