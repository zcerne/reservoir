"""gpumeep material adapter for the LC reservoir.

Relax/load via class_reservoir.Reservoir (engine-agnostic LdG minimization),
then expose a VECTORIZED material function over the relaxed (phi, theta)
director + optional STED MultilevelAtom — the same medium class_reservoir
hands MEEP, point-for-point (lc_geometry.get_dielectric_3d formula)."""
import os

import numpy as np

from gpumeep_setup import gm


class _VecMaterial:
    """Callable material function carrying a vectorized tensor6_vec — the
    form gpumeep's _eps_at consumes to avoid the per-point python loop."""

    def __init__(self, fn, vec):
        self._fn = fn
        self.tensor6_vec = vec

    def __call__(self, v):
        return self._fn(v)


class ReservoirGPU:
    def __init__(self, folder_path, args, cell_y, cell_z):
        self.args = args
        self.center_x = float(args["center_x_meep"])
        # isotropic: true → uniform n medium (+ optional sted dye); skip the
        # LC relax / lc_fields.npz machinery entirely.
        self.isotropic = bool(args.get("isotropic", False))
        if self.isotropic:
            self.res = None
            sizes = args["sizes"]
            self.sx, self.sy = float(sizes[0]), float(sizes[1])
            self.sz = float(sizes[2]) if len(sizes) > 2 else 0.0
            self.n_iso = float(args.get("n", args.get("n_o", 1.5)))
            self._susc = self._sted_susceptibilities()
            return
        from class_reservoir import Reservoir
        self.res = Reservoir(folder_path)
        fields_file = os.path.join(folder_path, "simulation", "lc_fields.npz")
        if os.path.exists(fields_file):
            self.res.load_fields()
        else:
            self.res.run_minimization()
        cell = self.res._cell_size()
        self.sx, self.sy = float(cell[0]), float(cell[1])
        self.sz = float(cell[2]) if len(cell) > 2 else 0.0
        self.n_o_sq = self.res.n_o ** 2
        self.n_e_sq = self.res.n_e ** 2
        self.S = self.res.S
        # AFTER the director is available (sted.anisotropic reads it)
        self._susc = self._sted_susceptibilities()

    def _sted_susceptibilities(self):
        s = self.args.get("sted")
        if not s or not s.get("enabled", True):
            return ()
        # per-transition order overrides: absorption (pump 1->4) and emission
        # (2->3) dipole tensors need not match — S_em < S_abs from rotational
        # depolarization during the excited-state lifetime; sted.order is the
        # shared default.
        o_abs = s.get("order_absorption")
        o_em = s.get("order_emission")
        trans = [
            gm.Transition(1, 4, frequency=1 / float(s["lbdA"]), gamma=float(s["gammaA"]),
                          order=None if o_abs is None else float(o_abs)),
            gm.Transition(4, 3, transition_rate=float(s.get("rate_43", 10.0))),
            gm.Transition(2, 3, frequency=1 / float(s["lbdE"]), gamma=float(s["gammaE"]),
                          order=None if o_em is None else float(o_em)),
            gm.Transition(2, 1, transition_rate=float(s.get("rate_21", 100.0))),
        ]
        # sted.anisotropic=true -> dye transition dipole follows the LC
        # director: orientation callable u(X,Y,Z) from the SAME phi/theta
        # interpolators the eps-tensor uses; sigma-bar(x) = SGMA*[(1-S)/3*I
        # + S*u(x)(x)u(x)] with S = sted.order (dye order parameter).
        orientation = None
        if s.get("anisotropic", False):
            if self.isotropic or self.res is None:
                raise ValueError("sted.anisotropic requires an LC reservoir "
                                 "(director field), not isotropic:true")
            orientation = self._director_callable()
        atom = gm.MultilevelAtom(
            sigma=float(s["SGMA"]), transitions=trans,
            initial_populations=[float(s["N1_0"]), 0.0,
                                 float(s.get("N3_0", 0.0)), 0.0],
            orientation=orientation,
            order=float(s.get("order", 1.0)))
        return (atom,)

    def _director_callable(self):
        """u(X, Y, Z) -> (3, *shape) director components at arbitrary MEEP
        coordinates (2D: mid-z slice splines, Z ignored)."""
        from scipy.interpolate import RectBivariateSpline
        assert self.res is not None
        phi, theta, *_ = self.res.get_results_2d()
        x_lc = np.linspace(-self.sx / 2, self.sx / 2, phi.shape[0])
        y_lc = np.linspace(-self.sy / 2, self.sy / 2, phi.shape[1])
        phi_i = RectBivariateSpline(x_lc, y_lc, phi)
        theta_i = RectBivariateSpline(x_lc, y_lc, theta)
        cx = self.center_x

        def u(X, Y, Z=None):
            xq = np.clip(np.asarray(X) - cx, x_lc[0], x_lc[-1])
            yq = np.clip(np.asarray(Y), y_lc[0], y_lc[-1])
            p = phi_i(xq.ravel(), yq.ravel(), grid=False).reshape(xq.shape)
            t = theta_i(xq.ravel(), yq.ravel(), grid=False).reshape(xq.shape)
            return np.stack([np.sin(t) * np.cos(p),
                             np.sin(t) * np.sin(p),
                             np.cos(t)])

        return u

    def save_fields(self):
        if self.res is not None:
            self.res.save_fields()

    def get_isotropic_block(self):
        """Uniform isotropic block (index n) + optional sted dye — the
        `isotropic: true` fast path, no director/interpolation at all."""
        size = gm.Vector3(self.sx, self.sy, self.sz if self.sz > 0 else gm.inf)
        med = gm.Medium(index=self.n_iso, E_susceptibilities=self._susc)
        return [gm.Block(center=gm.Vector3(self.center_x, 0, 0),
                         size=size, material=med)]

    def _tensor_from_angles(self, phi, theta):
        """MEEP-convention uniaxial tensor from director angles (vectorized
        twin of lc_geometry.get_dielectric_3d)."""
        nx = np.sin(theta) * np.cos(phi)
        ny = np.sin(theta) * np.sin(phi)
        nz = np.cos(theta)
        d0 = self.n_e_sq - self.n_o_sq
        eps_avg = (self.n_e_sq + 2.0 * self.n_o_sq) / 3.0
        eps_perp = eps_avg - d0 * self.S / 3.0
        dS = d0 * self.S
        return (eps_perp + dS * nx * nx, eps_perp + dS * ny * ny,
                eps_perp + dS * nz * nz,
                dS * nx * ny, dS * nx * nz, dS * ny * nz)

    def _material_2d(self):
        from scipy.interpolate import RectBivariateSpline
        assert self.res is not None
        phi, theta, *_ = self.res.get_results_2d()
        x_lc = np.linspace(-self.sx / 2, self.sx / 2, phi.shape[0])
        y_lc = np.linspace(-self.sy / 2, self.sy / 2, phi.shape[1])
        phi_i = RectBivariateSpline(x_lc, y_lc, phi)
        theta_i = RectBivariateSpline(x_lc, y_lc, theta)
        cx, susc = self.center_x, self._susc
        tensor = self._tensor_from_angles

        def mat(v):
            p = float(np.asarray(phi_i(float(v.x) - cx, float(v.y))).flat[0])
            t = float(np.asarray(theta_i(float(v.x) - cx, float(v.y))).flat[0])
            t6 = tensor(p, t)
            return gm.Medium(epsilon_diag=t6[:3], epsilon_offdiag=t6[3:],
                             E_susceptibilities=susc)

        def tensor6_vec(X, Y, Z=None):
            xq = np.clip(X - cx, x_lc[0], x_lc[-1])
            yq = np.clip(Y, y_lc[0], y_lc[-1])
            p = phi_i(xq.ravel(), yq.ravel(), grid=False).reshape(X.shape)
            t = theta_i(xq.ravel(), yq.ravel(), grid=False).reshape(X.shape)
            return np.stack(tensor(p, t))

        return _VecMaterial(mat, tensor6_vec)

    def _material_3d(self):
        from scipy.interpolate import RegularGridInterpolator
        assert self.res is not None
        phi, theta, *_ = self.res.get_results()
        x_lc = np.linspace(-self.sx / 2, self.sx / 2, phi.shape[0])
        y_lc = np.linspace(-self.sy / 2, self.sy / 2, phi.shape[1])
        z_lc = np.linspace(-self.sz / 2, self.sz / 2, phi.shape[2])
        # fill_value=None → extrapolate at the block edge (scipy's sentinel;
        # its stub types it float, hence the ignores)
        phi_i = RegularGridInterpolator(
            (x_lc, y_lc, z_lc), phi, bounds_error=False,
            fill_value=None)  # type: ignore[arg-type]
        theta_i = RegularGridInterpolator(
            (x_lc, y_lc, z_lc), theta, bounds_error=False,
            fill_value=None)  # type: ignore[arg-type]
        cx, susc = self.center_x, self._susc
        tensor = self._tensor_from_angles

        def mat(v):
            pt = np.array([[float(v.x) - cx, float(v.y), float(v.z)]])
            t6 = tensor(float(phi_i(pt)[0]), float(theta_i(pt)[0]))
            return gm.Medium(epsilon_diag=t6[:3], epsilon_offdiag=t6[3:],
                             E_susceptibilities=susc)

        def tensor6_vec(X, Y, Z):
            pts = np.stack([np.clip(X - cx, x_lc[0], x_lc[-1]),
                            np.clip(Y, y_lc[0], y_lc[-1]),
                            np.clip(Z, z_lc[0], z_lc[-1])], axis=-1)
            p = phi_i(pts); t = theta_i(pts)
            return np.stack(tensor(p, t))

        return _VecMaterial(mat, tensor6_vec)

    def get_geometry_blocks(self, mp_module=None):
        if self.isotropic:
            return self.get_isotropic_block()
        assert self.res is not None
        mat = (self._material_2d()
               if self.sz == 0 or len(self.res.dimensions) == 2
               else self._material_3d())
        size = gm.Vector3(self.sx, self.sy, self.sz if self.sz > 0 else gm.inf)
        return [gm.Block(center=gm.Vector3(self.center_x, 0, 0),
                         size=size, material=mat)]
