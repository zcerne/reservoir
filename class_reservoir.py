import numpy as np
from alcs_class import XaLCS


class Reservoir:
    """
    Liquid crystal rectangle/cube that relaxes to minimal Frank free energy
    under planar boundary conditions (theta=pi/2) on all anchored faces.

    dimensions: (sx, sy) for 2D or (sx, sy, sz) for 3D, in micrometers.
    resolution: grid points per micrometer.
    boundary_conditions: ('free'|'periodic', ...) per x, y, z axis.
    elastic_constants: (k1, k2, k3, q0) in pN.
    face_phi: 6-tuple of phi angles (radians) for faces
              (x_min, x_max, y_min, y_max, z_min, z_max).
              None for a face = free (no anchoring).
              Default: all faces planar at phi=0.
    """

    def __init__(self, dimensions=(10, 10), resolution=5,
                 boundary_conditions=("free", "free", "free"),
                 elastic_constants=(11.1, 6.5, 17.1, 0),
                 face_phi=(0, 0, 0, 0, 0, 0)):
        self.dimensions = dimensions
        self.resolution = resolution
        self.boundary_conditions = boundary_conditions
        self.elastic_constants = elastic_constants
        self.face_phi = face_phi
        self.maxeval = 2000
        self.f_tolerance = 1e-6
        self._sim = None

    def _cell_size(self):
        if len(self.dimensions) == 2:
            sx, sy = self.dimensions
            # 4 grid intervals in z (5 points) for quasi-2D
            sz = 4.0 / self.resolution
            return (sx, sy, sz)
        return tuple(self.dimensions)

    def run_minimization(self):
        cell = self._cell_size()
        sim = XaLCS(cell_size=cell, resolution=self.resolution,
                    elastic_constants=self.elastic_constants,
                    boundary_conditions=self.boundary_conditions)
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

    def get_results_2d(self, z_slice=None):
        """Returns (phi, theta, nx, ny, nz) for a single z-slice (default: middle)."""
        phi, theta, nx, ny, nz = self.get_results()
        iz = phi.shape[2] // 2 if z_slice is None else z_slice  # type: ignore[misc]
        return phi[:, :, iz], theta[:, :, iz], nx[:, :, iz], ny[:, :, iz], nz[:, :, iz]
