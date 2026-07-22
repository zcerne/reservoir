import warnings
import meep as mp
import numpy as np
from lc_geometry import get_dielectric_from_S_theta_yz


class SLM:
    """LC spatial light modulator — fixed phase mask.

    Divides the aperture into areas, each with a constant LC director angle
    that produces a target phase shift.

    area_values: list of floats in [0, 1]
        0 → zero relative phase (theta = π/2, ordinary index n_o)
        1 → π phase shift (theta = 0, extraordinary index n_e)

    Width is computed from physics: d = λ / (2·(n_e − n_o)) — thickness
    needed for a π phase excursion (half-wave retarder at val=1).

    2D simulation: n_areas equal strips along y.
    3D simulation: 2×2 grid only (n_areas must be 4).
        Order: [0]=y− z+, [1]=y+ z+, [2]=y− z−, [3]=y+ z−
    """

    def __init__(self, args):
        self.wavelength = float(args["lam"])
        self.n_o = float(args["no_ne"][0])
        self.n_e = float(args["no_ne"][1])
        self.n_areas = int(args["number_of_areas"])
        self.area_values = [float(v) for v in args["area_values"]]
        self.center = args.get("center", mp.Vector3(0, 0, 0))
        self.size_y = float(args.get("size_y", 10.0))
        self.size_z = float(args.get("size_z", 0.0))
        self._is_3d = self.size_z > 0
        self.width = self._set_width()
        self._grid = self._set_grid()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _set_width(self):
        """LC thickness for π phase shift at val=1: d = λ / (2·Δn)."""
        delta_n = self.n_e - self.n_o
        if delta_n <= 0:
            raise ValueError(f"SLM: n_e ({self.n_e}) must be > n_o ({self.n_o})")
        return self.wavelength / (2 * delta_n)

    def _set_grid(self):
        """Return list of area dicts with geometry and pre-computed mp.Medium.

        2D: n_areas equal strips along y.
        3D: 2×2 grid. Order: [0]=y− z+  [1]=y+ z+  [2]=y− z−  [3]=y+ z−
        """
        grid = []
        n_o_sq = self.n_o ** 2
        n_e_sq = self.n_e ** 2
        cx = float(self.center.x)

        if not self._is_3d:
            dy = self.size_y / self.n_areas
            for i, val in enumerate(self.area_values):
                y_min = -self.size_y / 2 + i * dy
                y_c   = y_min + dy / 2
                theta = self._value_to_theta(val)
                main_diag, off_diag = get_dielectric_from_S_theta_yz(n_o_sq, n_e_sq, theta, 1.0)
                grid.append({
                    "center":   mp.Vector3(cx, y_c, 0),
                    "size":     mp.Vector3(self.width, dy, mp.inf),
                    "material": mp.Medium(epsilon_diag=main_diag, epsilon_offdiag=off_diag),
                })
        else:
            if self.n_areas != 4:
                warnings.warn(f"SLM 3D supports only 2×2 grid (4 areas); got {self.n_areas} — using first 4.")
            half_y, half_z = self.size_y / 2, self.size_z / 2
            bounds = [
                (-half_y, 0.0, 0.0,    half_z),   # [0] y− z+
                (0.0,  half_y, 0.0,    half_z),   # [1] y+ z+
                (-half_y, 0.0, -half_z, 0.0),     # [2] y− z−
                (0.0,  half_y, -half_z, 0.0),     # [3] y+ z−
            ]
            for i, (y_min, y_max, z_min, z_max) in enumerate(bounds):
                val   = self.area_values[i] if i < len(self.area_values) else 0.0
                theta = self._value_to_theta(val)
                main_diag, off_diag = get_dielectric_from_S_theta_yz(n_o_sq, n_e_sq, theta, 1.0)
                grid.append({
                    "center":   mp.Vector3(cx, (y_min + y_max) / 2, (z_min + z_max) / 2),
                    "size":     mp.Vector3(self.width, y_max - y_min, z_max - z_min),
                    "material": mp.Medium(epsilon_diag=main_diag, epsilon_offdiag=off_diag),
                })
        return grid

    def _value_to_theta(self, val):
        """Map area_value ∈ [0, 1] → theta ∈ [π/2, 0].

        val=0 → θ=π/2 (ordinary, zero relative phase)
        val=1 → θ=0   (extraordinary, π phase shift)
        """
        return (1.0 - float(np.clip(val, 0.0, 1.0))) * (np.pi / 2)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def get_geometry_blocks(self, mp_module=None):
        """One MEEP Block per area, each with a pre-computed fixed LC medium."""
        return [
            mp.Block(center=area["center"], size=area["size"], material=area["material"])
            for area in self._grid
        ]
