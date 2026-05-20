import meep as mp
import numpy as np

from scipy.interpolate import RectBivariateSpline
from lc_geometry import get_dielectric_from_S_theta_yz


class Layer:
    def __init__(self, args):
        self.size = args["size"]
        self.center = args["center"]
        self.layer_type = args.get("layer_type", "lc")  # "lc" or "isotropic"
        self.n_indices = args["n_indices"]  # [n_o, n_e] for LC; [n, n] for isotropic
        self.n_field: np.ndarray | None = None  # 2D theta field (ny, nx), fixed
        self.resolution = args["resolution"]
        self.theta0 = args["theta_0"]
        # cell_y overrides size.y for grid allocation when size.y is mp.inf
        self._cell_y: float = float(args.get("cell_y", self.size.y))
        self.initialize_fields()

    def initialize_fields(self) -> None:
        """Allocate theta field and build spline interpolators."""
        nx = max(2, int(round(self.resolution * self.size.x)))
        ny = max(2, int(round(self.resolution * self._cell_y)))
        self.n_field = np.full((ny, nx), self.theta0, dtype=np.float64)
        if self.layer_type == "lc":
            self.update_interpolation_functions()

    def get_geometry_block(self):
        if self.layer_type == "isotropic":
            return mp.Block(
                center=self.center,
                size=self.size,
                material=mp.Medium(index=self.n_indices[0]),
            )
        return mp.Block(
            center=self.center,
            size=self.size,
            material=lambda v: self.get_material(v),
        )

    def get_material(self, v):
        theta_value = float(np.asarray(self.interpolation_function_theta(v.x, v.y)).flat[0])
        n_o_sq = self.n_indices[0] ** 2
        n_e_sq = self.n_indices[1] ** 2
        # Director in yz plane: n=(0, sinθ, cosθ) — correct for Ez polarisation
        main_diag, off_diag = get_dielectric_from_S_theta_yz(n_o_sq, n_e_sq, theta_value, 1.0)
        assert main_diag is not None
        assert off_diag is not None
        return mp.Medium(epsilon_diag=main_diag, epsilon_offdiag=off_diag)

    def update_interpolation_functions(self) -> None:
        """Rebuild spline interpolator for theta from current n_field."""
        assert self.n_field is not None
        ny, nx = self.n_field.shape
        x_axis = np.linspace(self.center.x - self.size.x / 2,
                             self.center.x + self.size.x / 2, nx)
        y_axis = np.linspace(-self._cell_y / 2, self._cell_y / 2, ny)
        kx = min(3, max(1, nx - 1))
        ky = min(3, max(1, ny - 1))
        self.interpolation_function_theta = RectBivariateSpline(
            x_axis, y_axis, self.n_field.T, kx=kx, ky=ky)
