import json
import numpy as np
from pathlib import Path
from alcs_class import XaLCS
from lc_geometry import get_dielectric_3d


class Reservoir:
    """
    LC rectangle/cube that relaxes to minimal Frank free energy under planar
    boundary conditions. Reads parameters from simulation_data.json in the
    given folder (lc key).

    face_phi: 6-tuple (x_min, x_max, y_min, y_max, z_min, z_max) in radians.
              None = free (no anchoring). Default: all faces planar at phi=0.
    """

    def __init__(self, folder: str | Path):
        folder = Path(folder)
        with open(folder / "simulation_data.json") as f:
            data = json.load(f)
        cfg = data["lc"]

        ec = cfg["elastic_constants"]
        self.dimensions = tuple(cfg["dimensions"])
        self.resolution = cfg["resolution"]
        self.boundary_conditions = tuple(cfg["boundary_conditions"])
        self.elastic_constants = (ec["K1"], ec["K2"], ec["K3"], ec["q0"])
        self.face_phi = tuple(x if x is not None else None for x in cfg["face_phi"])
        self.maxeval = cfg.get("maxeval", 2000)
        self.f_tolerance = cfg.get("f_tolerance", 1e-6)
        self.n_o = float(cfg.get("n_o", 1.52))
        self.n_e = float(cfg.get("n_e", 1.71))
        self.S   = float(cfg.get("S", 1.0))
        self.n_background = float(data.get("background_index", 1.0))
        self._sim = None
        self._meep_center_x: float = 0.0

    def _cell_size(self):
        if len(self.dimensions) == 2:
            sx, sy = self.dimensions
            sz = 4.0 / self.resolution  # 5 z-points for quasi-2D
            return (sx, sy, sz)
        return tuple(self.dimensions)

    def run_minimization(self):
        cell = self._cell_size()
        sim = XaLCS(cell_size=cell, resolution=self.resolution,
                    elastic_constants=self.elastic_constants,
                    boundary_conditions=self.boundary_conditions,
                    phi_only=len(self.dimensions) == 2)
        sim.maxeval = self.maxeval
        sim.f_tolerance = self.f_tolerance

        cell_size = np.asarray(cell)
        res = (cell_size * self.resolution + 1).astype(int)
        x = np.linspace(-cell_size[0]/2, cell_size[0]/2, res[0])
        y = np.linspace(-cell_size[1]/2, cell_size[1]/2, res[1])
        z = np.linspace(-cell_size[2]/2, cell_size[2]/2, res[2])
        pos_x, pos_y, pos_z = np.meshgrid(x, y, z)
        pos = np.swapaxes(np.asarray((pos_x, pos_y, pos_z)), 1, 2)
        x_coords, y_coords, z_coords = pos[0].ravel(), pos[1].ravel(), pos[2].ravel()
        n = x_coords.size

        # Boolean masks selecting the outermost grid points on each face.
        # Pinning their bounds enforces anchoring.
        face_masks = [
            x_coords == -cell_size[0]/2,  # x_min
            x_coords ==  cell_size[0]/2,  # x_max
            y_coords == -cell_size[1]/2,  # y_min
            y_coords ==  cell_size[1]/2,  # y_max
            z_coords == -cell_size[2]/2,  # z_min
            z_coords ==  cell_size[2]/2,  # z_max
        ]

        phi0 = np.zeros(n)
        theta0 = np.full(n, np.pi / 2)

        lb_phi = np.full(n, -np.pi)
        ub_phi = np.full(n,  np.pi)
        lb_theta = np.zeros(n)
        ub_theta = np.full(n, np.pi)

        for mask, phi_val in zip(face_masks, self.face_phi):
            if phi_val is None:
                continue
            phi0[mask] = phi_val
            lb_phi[mask] = phi_val
            ub_phi[mask] = phi_val
            theta0[mask] = np.pi / 2
            lb_theta[mask] = np.pi / 2
            ub_theta[mask] = np.pi / 2

        sim.initial_state_phi = phi0
        sim.initial_state_theta = theta0
        sim.lower_bounds_phi = lb_phi
        sim.upper_bounds_phi = ub_phi
        sim.lower_bounds_theta = lb_theta
        sim.upper_bounds_theta = ub_theta

        sim.setup()
        sim.minimize()
        self._sim = sim

    def get_results(self):
        """Returns (phi, theta, nx, ny, nz) arrays on the 3D grid."""
        if self._sim is None:
            raise RuntimeError("Run run_minimization() first.")
        return self._sim.get_results()

    def get_geometry_blocks(self):
        from scipy.interpolate import RectBivariateSpline
        import meep as mp

        phi, theta, *_ = self.get_results_2d()  # shape (nx, ny)

        cell = self._cell_size()
        sx, sy = float(cell[0]), float(cell[1])
        nx_pts, ny_pts = phi.shape
        x_lc = np.linspace(-sx / 2, sx / 2, nx_pts)
        y_lc = np.linspace(-sy / 2, sy / 2, ny_pts)

        phi_interp   = RectBivariateSpline(x_lc, y_lc, phi)
        theta_interp = RectBivariateSpline(x_lc, y_lc, theta)

        n_o_sq = self.n_o ** 2
        n_e_sq = self.n_e ** 2
        S  = self.S
        cx = self._meep_center_x

        def _mat(v):
            lc_x = float(v.x) - cx
            phi_v   = float(np.asarray(phi_interp(lc_x, float(v.y))).flat[0])
            theta_v = float(np.asarray(theta_interp(lc_x, float(v.y))).flat[0])
            d, od = get_dielectric_3d(n_o_sq, n_e_sq, phi_v, theta_v, S)
            return mp.Medium(epsilon_diag=d, epsilon_offdiag=od)

        return [mp.Block(
            center=mp.Vector3(cx, 0, 0),
            size=mp.Vector3(sx, sy, mp.inf),
            material=lambda v: _mat(v),
        )]

    def get_results_2d(self, z_slice=None):
        """Returns (phi, theta, nx, ny, nz) for a single z-slice (default: middle)."""
        phi, theta, nx, ny, nz = self.get_results()
        iz = phi.shape[2] // 2 if z_slice is None else z_slice  # type: ignore[misc]
        return phi[:, :, iz], theta[:, :, iz], nx[:, :, iz], ny[:, :, iz], nz[:, :, iz]
