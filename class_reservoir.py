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
        self.folder = folder
        with open(folder / "simulation_data.json") as f:
            data = json.load(f)
        cfg = data["reservoir"]

        ec = cfg["elastic_constants"]
        self.dimensions = tuple(cfg["sizes"])
        self.resolution = cfg["resolution"]
        self.boundary_conditions = tuple(cfg["boundary_conditions"])
        self.elastic_constants = (ec["K1"], ec["K2"], ec["K3"], ec["q0"])
        self.face_phi   = tuple(x if x is not None else None for x in cfg["face_phi"])
        _fp = cfg.get("face_theta")
        if _fp is not None:
            self.face_theta = tuple(x if x is not None else None for x in _fp)
        else:
            # backward compat: pin theta=π/2 wherever phi is pinned
            self.face_theta = tuple(np.pi / 2 if p is not None else None for p in self.face_phi)
        self.optimize_phi_theta = tuple(cfg.get("optimize_phi_theta", [True, True]))
        self.boundary_function = cfg.get("boundary_function", None)
        self.boundary_seed     = cfg.get("boundary_seed", None)
        self.maxeval = cfg.get("maxeval", 2000)
        self.f_tolerance = cfg.get("f_tolerance", 1e-6)
        self.n_o = float(cfg.get("n_o", 1.52))
        self.n_e = float(cfg.get("n_e", 1.71))
        self.S   = float(cfg.get("S", 1.0))
        self.n_background = float(data.get("background_index", 1.0))
        self._sim = None
        self._phi_cache: np.ndarray | None = None
        self._theta_cache: np.ndarray | None = None
        self._meep_center_x: float = 0.0

    def _cell_size(self) -> tuple[float, float, float]:
        if len(self.dimensions) == 2:
            sx, sy = self.dimensions
            return (float(sx), float(sy), 4.0 / self.resolution)
        sx, sy, sz = self.dimensions
        return (float(sx), float(sy), float(sz))

    def run_minimization(self):
        cell = self._cell_size()
        sim = XaLCS(cell_size=cell, resolution=self.resolution,
                    elastic_constants=self.elastic_constants,
                    boundary_conditions=self.boundary_conditions,
                    optimize_phi_theta=self.optimize_phi_theta)
        sim.maxeval = self.maxeval
        sim.f_tolerance = self.f_tolerance

        cell_size = np.asarray(cell)
        res = (cell_size * self.resolution + 1).astype(int)
        nz_pts = int(res[2])

        if self.boundary_function is not None:
            from functions_boundaries import random_2d_boundaries, random_3d_boundaries
            _fn_map = {"random": random_2d_boundaries, "random_3d": random_3d_boundaries}
            fn = _fn_map[self.boundary_function]
            dims = self.dimensions if self.boundary_function == "random_3d" else self.dimensions[:2]
            fp_arr, ft_arr = fn(self.resolution, dims, seed=self.boundary_seed)
            if "z_min" in fp_arr:
                # 3D function: all 6 faces have per-pixel arrays sized to match face_mask counts
                active_face_phi = [fp_arr[k] for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")]
                active_face_theta = [ft_arr[k] for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")]
            else:
                # 2D function: repeat each 1D edge array across the z dimension
                active_face_phi = [
                    np.repeat(fp_arr["x_min"], nz_pts),
                    np.repeat(fp_arr["x_max"], nz_pts),
                    np.repeat(fp_arr["y_min"], nz_pts),
                    np.repeat(fp_arr["y_max"], nz_pts),
                    None, None,
                ]
                active_face_theta = [
                    np.repeat(ft_arr["x_min"], nz_pts),
                    np.repeat(ft_arr["x_max"], nz_pts),
                    np.repeat(ft_arr["y_min"], nz_pts),
                    np.repeat(ft_arr["y_max"], nz_pts),
                    None, None,
                ]
        else:
            active_face_phi   = list(self.face_phi)
            active_face_theta = list(self.face_theta)
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

        for mask, phi_val, theta_val in zip(face_masks, active_face_phi, active_face_theta):
            if phi_val is not None:
                phi0[mask] = phi_val
                lb_phi[mask] = phi_val
                ub_phi[mask] = phi_val
            if theta_val is not None:
                theta0[mask] = theta_val
                lb_theta[mask] = theta_val
                ub_theta[mask] = theta_val

        sim.initial_state_phi = phi0
        sim.initial_state_theta = theta0
        sim.lower_bounds_phi = lb_phi
        sim.upper_bounds_phi = ub_phi
        sim.lower_bounds_theta = lb_theta
        sim.upper_bounds_theta = ub_theta

        sim.setup()
        sim.minimize()
        self._sim = sim

    def load_fields(self):
        """Load pre-computed director field from lc_fields.npz (skips minimization)."""
        npz = self.folder / "simulation" / "lc_fields.npz"
        data = np.load(npz)
        self._phi_cache   = data["phi"]
        self._theta_cache = data["theta"]

    def get_results(self):
        """Returns (phi, theta, nx, ny, nz) arrays on the 3D grid."""
        if self._phi_cache is not None and self._theta_cache is not None:
            phi, theta = self._phi_cache, self._theta_cache
            nx = np.sin(theta) * np.cos(phi)
            ny = np.sin(theta) * np.sin(phi)
            nz = np.cos(theta)
            return phi, theta, nx, ny, nz
        if self._sim is None:
            raise RuntimeError("Run run_minimization() or load_fields() first.")
        return self._sim.get_results()

    def get_geometry_blocks(self):
        import meep as mp

        cell = self._cell_size()
        sx, sy, sz = float(cell[0]), float(cell[1]), float(cell[2])
        n_o_sq = self.n_o ** 2
        n_e_sq = self.n_e ** 2
        S  = self.S
        cx = self._meep_center_x

        if len(self.dimensions) == 2:
            from scipy.interpolate import RectBivariateSpline
            phi, theta, *_ = self.get_results_2d()
            nx_pts, ny_pts = phi.shape
            x_lc = np.linspace(-sx / 2, sx / 2, nx_pts)
            y_lc = np.linspace(-sy / 2, sy / 2, ny_pts)
            phi_interp   = RectBivariateSpline(x_lc, y_lc, phi)
            theta_interp = RectBivariateSpline(x_lc, y_lc, theta)

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
        else:
            from scipy.interpolate import RegularGridInterpolator
            phi, theta, *_ = self.get_results()
            nx_pts = int(phi.shape[0])  # type: ignore[index]
            ny_pts = int(phi.shape[1])  # type: ignore[index]
            nz_pts = int(phi.shape[2])  # type: ignore[index]
            x_lc = np.linspace(-sx / 2, sx / 2, nx_pts)
            y_lc = np.linspace(-sy / 2, sy / 2, ny_pts)
            z_lc = np.linspace(-sz / 2, sz / 2, nz_pts)
            phi_interp   = RegularGridInterpolator((x_lc, y_lc, z_lc), phi,
                                                   bounds_error=False, fill_value=np.nan)  # type: ignore[arg-type]
            theta_interp = RegularGridInterpolator((x_lc, y_lc, z_lc), theta,
                                                   bounds_error=False, fill_value=np.nan)  # type: ignore[arg-type]

            def _mat3(v):
                pt = np.array([[float(v.x) - cx, float(v.y), float(v.z)]])
                phi_v   = float(phi_interp(pt)[0])
                theta_v = float(theta_interp(pt)[0])
                d, od = get_dielectric_3d(n_o_sq, n_e_sq, phi_v, theta_v, S)
                return mp.Medium(epsilon_diag=d, epsilon_offdiag=od)

            return [mp.Block(
                center=mp.Vector3(cx, 0, 0),
                size=mp.Vector3(sx, sy, sz),
                material=lambda v: _mat3(v),
            )]

    def get_results_2d(self, z_slice=None):
        """Returns (phi, theta, nx, ny, nz) for a single z-slice (default: middle)."""
        phi, theta, nx, ny, nz = self.get_results()
        iz = phi.shape[2] // 2 if z_slice is None else z_slice  # type: ignore[misc]
        return phi[:, :, iz], theta[:, :, iz], nx[:, :, iz], ny[:, :, iz], nz[:, :, iz]

    def save_fields(self):
        """Save director field arrays + grid coordinates to lc_fields.npz."""
        phi, theta, *_ = self.get_results()
        sx, sy, sz = self._cell_size()
        nx, ny, nz = phi.shape[0], phi.shape[1], phi.shape[2]  # type: ignore[misc]
        x = np.linspace(-sx / 2, sx / 2, nx)
        y = np.linspace(-sy / 2, sy / 2, ny)
        z = np.linspace(-sz / 2, sz / 2, nz)
        out = self.folder / "simulation"
        out.mkdir(exist_ok=True)
        np.savez(out / "lc_fields.npz", phi=phi, theta=theta, x=x, y=y, z=z)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/test")
    args = parser.parse_args()
    r = Reservoir(args.path)
    r.run_minimization()
    r.save_fields()
    print(f"Done. Fields saved to {r.folder / 'simulation' / 'lc_fields.npz'}")

