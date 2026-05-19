import nlopt
import numpy as np
from numpy.typing import NDArray
from jax import value_and_grad
from alcs_jax import n_sph, fe_core_director, fe_core_director_periodic


class XaLCS:

    iter_counter: int = 0
    pos = None
    ek = None
    dp = None
    bpad = None
    bpad_ixs = None
    periodic_b = False
    optimize_phi_theta = (True, True)

    def __init__(self, cell_size, resolution, elastic_constants,
                 boundary_conditions=("free", "free", "free"),
                 optimize_phi_theta=(True, True)):
        """
        cell_size: (sx, sy, sz) in micrometers
        resolution: points per micrometer
        elastic_constants: (k1, k2, k3, q0)
        boundary_conditions: tuple of 'free' or 'periodic' per axis
        optimize_phi_theta: (opt_phi, opt_theta) — which angles are optimized
        """
        self.cell_size = cell_size
        self.unit_resolution = resolution
        self.constants = elastic_constants
        self.boundary_conditions = boundary_conditions
        self.optimize_phi_theta = tuple(optimize_phi_theta)

        self.initial_state_phi: NDArray[np.float64] | None = None
        self.initial_state_theta: NDArray[np.float64] | None = None
        self.lower_bounds_phi: NDArray[np.float64] | None = None
        self.upper_bounds_phi: NDArray[np.float64] | None = None
        self.lower_bounds_theta: NDArray[np.float64] | None = None
        self.upper_bounds_theta: NDArray[np.float64] | None = None

        self.maxeval = 2000
        self.f_tolerance = 1e-6

        self._resolution: tuple | None = None
        self._spacings: NDArray[np.float64] | None = None
        self._coordinates = None
        self._co_flat = None
        self._boundary_pad = None
        self._boundary_ixs = None
        self._initial_state: NDArray[np.float64] | None = None
        self._lower_bounds: NDArray[np.float64] | None = None
        self._upper_bounds: NDArray[np.float64] | None = None
        self._n_parameters: int | None = None
        self._activated = False
        self._sim_ran = False
        self._result: NDArray[np.float64] | None = None

    @property
    def resolution(self):
        return self._resolution

    @property
    def spacings(self):
        assert self._spacings is not None
        return tuple(self._spacings)

    @property
    def n_parameters(self):
        return self._n_parameters

    @property
    def simulation_completed(self):
        return self._sim_ran

    @property
    def voxel_centers_pointcloud(self):
        return self._co_flat

    @staticmethod
    def _compute_boundary_pad(boundary_conditions):
        pad_size = {"free": (0, 0), "periodic": (1, 1)}
        pad_ixs = {"free": (None, None), "periodic": (1, -1)}
        pad_width = tuple(pad_size[bc] for bc in boundary_conditions)
        pad_inds = tuple(pad_ixs[bc] for bc in boundary_conditions)
        return pad_width, pad_inds

    def setup(self):
        assert self.initial_state_phi is not None
        assert self.initial_state_theta is not None
        assert self.lower_bounds_phi is not None
        assert self.upper_bounds_phi is not None
        assert self.lower_bounds_theta is not None
        assert self.upper_bounds_theta is not None

        self._sim_ran = False
        XaLCS.iter_counter = 0

        cs = np.asarray(self.cell_size)
        res = (cs * self.unit_resolution + 1).astype(int)
        self._resolution = tuple(res)
        self._spacings = cs / (res - 1)

        x = np.linspace(-cs[0]/2, cs[0]/2, res[0])
        y = np.linspace(-cs[1]/2, cs[1]/2, res[1])
        z = np.linspace(-cs[2]/2, cs[2]/2, res[2])
        pos_x, pos_y, pos_z = np.meshgrid(x, y, z)
        pos = np.swapaxes(np.asarray((pos_x, pos_y, pos_z)), 1, 2)
        self._coordinates = pos
        self._co_flat = pos.reshape(3, -1)

        XaLCS.pos = pos
        XaLCS.ek = self.constants
        XaLCS.dp = self.spacings
        XaLCS.optimize_phi_theta = tuple(self.optimize_phi_theta)

        opt_phi, opt_theta = self.optimize_phi_theta
        if opt_phi and opt_theta:
            self._initial_state = np.concatenate((self.initial_state_phi, self.initial_state_theta))
            self._lower_bounds  = np.concatenate((self.lower_bounds_phi,  self.lower_bounds_theta))
            self._upper_bounds  = np.concatenate((self.upper_bounds_phi,  self.upper_bounds_theta))
        elif opt_phi:
            self._initial_state = self.initial_state_phi
            self._lower_bounds  = self.lower_bounds_phi
            self._upper_bounds  = self.upper_bounds_phi
            XaLCS.fixed_theta   = self.initial_state_theta
        elif opt_theta:
            self._initial_state = self.initial_state_theta
            self._lower_bounds  = self.lower_bounds_theta
            self._upper_bounds  = self.upper_bounds_theta
            XaLCS.fixed_phi     = self.initial_state_phi
        else:
            raise ValueError("optimize_phi_theta must have at least one True")

        self._n_parameters = self._initial_state.size

        self._boundary_pad, self._boundary_ixs = self._compute_boundary_pad(self.boundary_conditions)
        XaLCS.bpad = self._boundary_pad
        XaLCS.bpad_ixs = self._boundary_ixs
        XaLCS.periodic_b = "periodic" in self.boundary_conditions

        self._activated = True

    @staticmethod
    def _fe_wrap(v):
        assert XaLCS.pos is not None
        opt_phi, opt_theta = XaLCS.optimize_phi_theta
        if opt_phi and opt_theta:
            n = v.shape[0] // 2
            nv = n_sph(v[:n], v[n:]).reshape(XaLCS.pos.shape)
        elif opt_phi:
            nv = n_sph(v, XaLCS.fixed_theta).reshape(XaLCS.pos.shape)
        else:
            nv = n_sph(XaLCS.fixed_phi, v).reshape(XaLCS.pos.shape)
        if XaLCS.periodic_b:
            out = fe_core_director_periodic(nv, XaLCS.ek, XaLCS.dp, XaLCS.bpad, XaLCS.bpad_ixs)
        else:
            out = fe_core_director(nv, XaLCS.ek, XaLCS.dp)
        XaLCS.iter_counter += 1
        return out

    @staticmethod
    def _objective(v, gradient):
        value, dv_du = value_and_grad(XaLCS._fe_wrap)(v)
        if gradient.size > 0:
            gradient[:] = dv_du
        return value.item()

    def minimize(self):
        assert self._n_parameters is not None
        assert self._lower_bounds is not None
        assert self._upper_bounds is not None
        assert self._initial_state is not None
        solver = nlopt.opt(nlopt.LD_MMA, self._n_parameters)
        solver.set_lower_bounds(self._lower_bounds)
        solver.set_upper_bounds(self._upper_bounds)
        solver.set_min_objective(self._objective)
        solver.set_maxeval(self.maxeval)
        solver.set_ftol_rel(self.f_tolerance)
        self._result = solver.optimize(self._initial_state)
        self._sim_ran = True

    def get_results(self):
        """Returns (phi, theta, nx, ny, nz) arrays shaped as grid resolution."""
        assert self._sim_ran
        assert self._result is not None
        assert self._n_parameters is not None
        opt_phi, opt_theta = self.optimize_phi_theta
        if opt_phi and opt_theta:
            n = self._n_parameters // 2
            phi   = self._result[:n].reshape(self._resolution)
            theta = self._result[n:].reshape(self._resolution)
        elif opt_phi:
            phi   = self._result.reshape(self._resolution)
            theta = self.initial_state_theta.reshape(self._resolution)  # type: ignore[union-attr]
        else:
            phi   = self.initial_state_phi.reshape(self._resolution)    # type: ignore[union-attr]
            theta = self._result.reshape(self._resolution)
        dirs = n_sph(phi.ravel(), theta.ravel())
        nx = np.asarray(dirs[0]).reshape(self._resolution)
        ny = np.asarray(dirs[1]).reshape(self._resolution)
        nz = np.asarray(dirs[2]).reshape(self._resolution)
        return phi, theta, nx, ny, nz
