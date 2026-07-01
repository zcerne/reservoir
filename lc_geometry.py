import numpy as np
try:
    import meep as mp          # only needed for the MEEP geometry builders below
except ImportError:
    mp = None                  # relax / characterization paths don't need meep


def get_dielectric_3d(n_o_sq, n_e_sq, phi, theta, S=1.0):
    """
    General 3D LC dielectric tensor.

    Director: n = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
        phi   : azimuthal angle in xy plane (radians)
        theta : polar angle from z-axis (radians); pi/2 = director in xy plane

    eps_ij = eps_perp*delta_ij + delta_eps * n_i * n_j
    MEEP epsilon_offdiag = (eps_xy, eps_xz, eps_yz)
    """
    delta_eps0 = n_e_sq - n_o_sq
    eps_avg   = (n_e_sq + 2.0 * n_o_sq) / 3.0
    eps_perp  = eps_avg - (1.0 / 3.0) * delta_eps0 * S
    delta_eps = delta_eps0 * S

    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)

    main_diag = mp.Vector3(
        float(eps_perp + delta_eps * nx * nx),
        float(eps_perp + delta_eps * ny * ny),
        float(eps_perp + delta_eps * nz * nz),
    )
    off_diag = mp.Vector3(
        float(delta_eps * nx * ny),  # eps_xy
        float(delta_eps * nx * nz),  # eps_xz
        float(delta_eps * ny * nz),  # eps_yz
    )
    return main_diag, off_diag


def get_dielectric_from_S_theta_yz(n_o_sq, n_e_sq, theta, S):
    """
    Dielectric tensor for director rotating in the yz plane.

    Director: n = (0, sin(theta), cos(theta))

        theta=0    -> n along z -> Ez sees n_e (extraordinary)
        theta=pi/2 -> n along y -> Ez sees n_o (ordinary)

    Tensor components (xy-plane MEEP sim, Ez polarisation):
        eps_xx = eps_perp                       (ordinary, independent of theta)
        eps_yy = eps_perp + delta_eps * sin^2
        eps_zz = eps_perp + delta_eps * cos^2   <- what Ez sees
        eps_yz = delta_eps * sin * cos
        eps_xy = eps_xz = 0

    MEEP epsilon_offdiag convention: (eps_xy, eps_xz, eps_yz)
    """
    delta_eps0 = n_e_sq - n_o_sq
    eps_avg    = (n_e_sq + 2.0 * n_o_sq) / 3.0
    eps_perp   = eps_avg - (1.0 / 3.0) * delta_eps0 * S
    delta_eps  = delta_eps0 * S

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    t11 = float(eps_perp)
    t22 = float(eps_perp + delta_eps * sin_t ** 2)
    t33 = float(eps_perp + delta_eps * cos_t ** 2)
    t23 = float(delta_eps * sin_t * cos_t)

    main_diag = mp.Vector3(t11, t22, t33)
    off_diag  = mp.Vector3(0.0, 0.0, t23)   # (eps_xy, eps_xz, eps_yz)
    return main_diag, off_diag


def get_dielectric_from_S_theta(n_o_sq, n_e_sq, theta, S):
    """
    Dielectric tensor from Q-tensor for a director confined to the xy plane.

    Physics:
        Q_ij = S * (n_i*n_j - delta_ij/3)
        Director: n = (cos(theta), sin(theta), 0)

        eps_avg   = (n_e_sq + 2*n_o_sq) / 3
        delta_eps0 = n_e_sq - n_o_sq          (full-order birefringence)

        eps_par(S)  = eps_avg + (2/3)*delta_eps0*S   -> n_e_sq when S=1
        eps_perp(S) = eps_avg - (1/3)*delta_eps0*S   -> n_o_sq when S=1

        eps_ij = eps_perp(S)*delta_ij + delta_eps(S)*n_i*n_j
               where delta_eps(S) = eps_par(S) - eps_perp(S) = delta_eps0*S

    At S=0: isotropic eps_ij = eps_avg * delta_ij.
    At S=1: uniaxial eps_|| = n_e_sq, eps_perp = n_o_sq.

    Args:
        n_o_sq  : ordinary permittivity (n_o**2)
        n_e_sq  : extraordinary permittivity (n_e**2)
        theta   : director angle in xy plane (radians)
        S       : scalar order parameter in [0, 1]

    Returns:
        (main_diag, off_diag) as mp.Vector3 for mp.Medium
        MEEP convention: epsilon_offdiag = (eps_xy, eps_xz, eps_yz)
    """
    delta_eps0 = n_e_sq - n_o_sq
    eps_avg    = (n_e_sq + 2.0 * n_o_sq) / 3.0

    eps_perp = eps_avg - (1.0 / 3.0) * delta_eps0 * S
    delta_eps = delta_eps0 * S  # = eps_par - eps_perp

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # n = (cos_t, sin_t, 0)
    t11 = eps_perp + delta_eps * cos_t ** 2   # eps_xx
    t22 = eps_perp + delta_eps * sin_t ** 2   # eps_yy
    t33 = float(eps_perp)                      # eps_zz (out-of-plane, isotropic)
    t12 = delta_eps * cos_t * sin_t            # eps_xy
    # xz and yz are zero (director has no z component)

    main_diag = mp.Vector3(float(t11), float(t22), t33)
    off_diag  = mp.Vector3(float(t12), 0.0, 0.0)
    return main_diag, off_diag


def get_layer_id(v, layer_widths):
    L = np.sum(layer_widths)
    x = v.x + L / 2
    cum_sum = np.cumsum(layer_widths)
    layer_id = int(np.clip(np.searchsorted(cum_sum, x), 0, len(layer_widths) - 1))
    return x, layer_id


def get_material(v, design_weights, data, design_field_functs=None):
    """
    MEEP material function for LC layers.

    design_field_functs : list, one entry per layer.
        Each entry is either None (uniform layer) or [S_func, theta_func]
        where:
            S_func(x_local, y)     -> order parameter S in [0, 1]
            theta_func(x_local, y) -> director angle in xy plane (radians)
        Both are RectBivariateSpline interpolators.

    Isotropic layers (design_weights.shape[1] == 3, third column != 0):
        returned as mp.Medium(index=n).

    LC layers without field functions:
        Uniform dielectric at full order (S=1) and angle from design_weights[:,1].

    LC layers with field functions:
        Spatially varying dielectric from Q-tensor (S, theta) fields.
    """
    cell_type = data.get("cell_type", "basic")
    layer_widths = design_weights[:, 0]
    layer_angles = design_weights[:, 1]

    x_positive, layer_id = get_layer_id(v, layer_widths)
    x_local = x_positive - np.sum(layer_widths[:layer_id])

    # --- Isotropic layer (third column stores refractive index) ---
    if design_weights.shape[1] == 3:
        isotropic_n = design_weights[:, 2]
        if isotropic_n[layer_id] > 0:
            return mp.Medium(index=float(isotropic_n[layer_id]))

    main_diag, off_diag = None, None

    if cell_type == "basic":
        n_o_sq = data["medium_low"]  ** 2
        n_e_sq = data["medium_high"] ** 2

        if design_field_functs is None or design_field_functs[layer_id] is None:
            # Uniform layer at full order (S=1)
            theta = float(layer_angles[layer_id])
            main_diag, off_diag = get_dielectric_from_S_theta(n_o_sq, n_e_sq, theta, S=1.0)
        else:
            # Spatially varying: [S_func, theta_func] from _set_material_fields_functions
            functs     = design_field_functs[layer_id]
            S_func     = functs[0]   # phi slot repurposed as order-parameter S
            theta_func = functs[1]   # director angle in xy plane

            S     = float(np.asarray(S_func(x_local, v.y)).flat[0])
            theta = float(np.asarray(theta_func(x_local, v.y)).flat[0])

            # Clamp S to physical range
            S = float(np.clip(S, 0.0, 1.0))

            main_diag, off_diag = get_dielectric_from_S_theta(n_o_sq, n_e_sq, theta, S)

    assert main_diag is not None
    assert off_diag  is not None
    return mp.Medium(epsilon_diag=main_diag, epsilon_offdiag=off_diag)
