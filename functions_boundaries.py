import numpy as np


def random_3d_boundaries(resolution, dimensions, seed=None):
    """
    Random boundary conditions for a 3D LC reservoir — all 6 faces.

    Returns (face_phi, face_theta), each a dict with keys:
        'x_min', 'x_max'  — 1D arrays of length n_y * n_z  (y-major, z-fast)
        'y_min', 'y_max'  — 1D arrays of length n_x * n_z  (x-major, z-fast)
        'z_min', 'z_max'  — 1D arrays of length n_x * n_y  (x-major, y-fast)

    Ordering matches the C-order raveling of the (n_x, n_y, n_z) grid used
    by class_reservoir, so phi0[face_mask] = face_array assigns values correctly.

    phi  sampled uniformly in [0, π]   (LC head-tail symmetry)
    theta sampled uniformly in [0, π/2] (homeotropic → planar)
    """
    rng = np.random.default_rng(seed)
    sx, sy, sz = float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
    n_x = int(sx * resolution) + 1
    n_y = int(sy * resolution) + 1
    n_z = int(sz * resolution) + 1

    face_phi = {
        "x_min": rng.uniform(0, np.pi, n_y * n_z),
        "x_max": rng.uniform(0, np.pi, n_y * n_z),
        "y_min": rng.uniform(0, np.pi, n_x * n_z),
        "y_max": rng.uniform(0, np.pi, n_x * n_z),
        "z_min": rng.uniform(0, np.pi, n_x * n_y),
        "z_max": rng.uniform(0, np.pi, n_x * n_y),
    }
    face_theta = {
        "x_min": rng.uniform(0, np.pi / 2, n_y * n_z),
        "x_max": rng.uniform(0, np.pi / 2, n_y * n_z),
        "y_min": rng.uniform(0, np.pi / 2, n_x * n_z),
        "y_max": rng.uniform(0, np.pi / 2, n_x * n_z),
        "z_min": rng.uniform(0, np.pi / 2, n_x * n_y),
        "z_max": rng.uniform(0, np.pi / 2, n_x * n_y),
    }
    return face_phi, face_theta


def random_2d_boundaries(resolution, dimensions, seed=None):
    """
    Random planar boundary conditions for a 2D LC reservoir.

    Returns (face_phi, face_theta), each a dict with keys:
        'x_min', 'x_max'  — 1D arrays of length n_y (left/right faces)
        'y_min', 'y_max'  — 1D arrays of length n_x (bottom/top faces)

    Values are ordered from the negative to positive coordinate along
    the face (e.g. x_min goes from y_min to y_max).

    phi  sampled uniformly in [0, π]  (LC head-tail symmetry)
    theta sampled uniformly in [0, π/2]  (homeotropic → planar)

    Args:
        resolution  : LC grid resolution (grid points per µm)
        dimensions  : [sx, sy] reservoir physical size in µm
        seed        : integer seed for reproducibility (None = random)
    """
    rng = np.random.default_rng(seed)
    sx, sy = float(dimensions[0]), float(dimensions[1])
    n_x = int(sx * resolution) + 1  # points along x (used by y-faces)
    n_y = int(sy * resolution) + 1  # points along y (used by x-faces)

    face_phi = {
        "x_min": rng.uniform(0, np.pi, n_y),
        "x_max": rng.uniform(0, np.pi, n_y),
        "y_min": rng.uniform(0, np.pi, n_x),
        "y_max": rng.uniform(0, np.pi, n_x),
    }
    face_theta = {
        "x_min": rng.uniform(0, np.pi / 2, n_y),
        "x_max": rng.uniform(0, np.pi / 2, n_y),
        "y_min": rng.uniform(0, np.pi / 2, n_x),
        "y_max": rng.uniform(0, np.pi / 2, n_x),
    }
    return face_phi, face_theta
