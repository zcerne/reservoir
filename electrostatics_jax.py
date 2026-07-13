"""JAX-CG anisotropic Poisson solver for LC+electrode optimisation (option 2a).

Mirrors `electrostatics.py` but in pure JAX so the whole pipeline
(patches → voltages → V → E → director relax → FDTD → cost) is one differentiable
graph. `jax.grad(cost)(patches)` works end-to-end with no `pure_callback` boundaries.

Solves  −∇·(ε(n̂)·∇V) = 0   with Dirichlet at electrode cells, Neumann/periodic walls.

Discretisation: 7-point flux-divergence stencil with face-averaged ε per axis
(diagonal-ε only — same approximation as the numpy version's first cut). The
operator is SPD by construction (Dirichlet contributions moved to RHS, both
A[p, dir]=0 and A[dir, p]=0). Solved with `jax.scipy.sparse.linalg.cg`, which
provides its own custom_vjp so gradient flow is correct end-to-end.

Provides:
    build_eps_diag_jax(phi, theta, eps_perp, eps_a) → (3, nx, ny, nz)
    apply_neg_div_eps_grad(V, eps_diag, spacings, periodic) → (nx, ny, nz)
    solve_poisson_jax(eps_diag, spacings, dirichlet_mask, V_dirichlet, …) → V
    gradient_V_jax(V, spacings) → E = -∇V (3, nx, ny, nz)
"""
from __future__ import annotations
from typing import Sequence
import jax
import jax.numpy as jnp


# ===========================================================================
# 1. ε tensor diagonals from director angles
# ===========================================================================

def build_eps_diag_jax(phi: jnp.ndarray, theta: jnp.ndarray,
                       eps_perp: float, eps_a: float) -> jnp.ndarray:
    """Return (3, nx, ny, nz) array of ε_xx, ε_yy, ε_zz (diagonal of the full
    3×3 anisotropic ε tensor). Used by the diagonal-ε Poisson stencil.

    ε_xx = ε_⊥ + ε_a sin²θ cos²φ,  similar for yy, zz.
    """
    sin_t = jnp.sin(theta); cos_t = jnp.cos(theta)
    sin_p = jnp.sin(phi);   cos_p = jnp.cos(phi)
    nx_d = sin_t * cos_p
    ny_d = sin_t * sin_p
    nz_d = cos_t
    eps_xx = eps_perp + eps_a * nx_d**2
    eps_yy = eps_perp + eps_a * ny_d**2
    eps_zz = eps_perp + eps_a * nz_d**2
    return jnp.stack([eps_xx, eps_yy, eps_zz], axis=0)


# ===========================================================================
# 2. Discrete operator: per-axis -∇·(ε·∇) contribution
# ===========================================================================

def _pad_axis(arr: jnp.ndarray, axis: int, periodic: bool) -> jnp.ndarray:
    """Pad by one cell on each side along `axis`. Periodic → wrap; Neumann → edge."""
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (1, 1)
    return jnp.pad(arr, pad_width, mode="wrap" if periodic else "edge")


def _laplacian_axis(V: jnp.ndarray, eps_axis: jnp.ndarray, h: float,
                    axis: int, periodic: bool) -> jnp.ndarray:
    """Per-axis contribution of `-∂/∂x_axis (ε_axis ∂V/∂x_axis)` to the operator,
    discretised via face-flux divergence. Output has same shape as `V`.

    With Neumann padding (`mode='edge'`), boundary cells see neighbours = themselves
    → `V_centre − V_padded == 0` → zero flux through outer wall (Neumann zero-flux).
    With periodic padding (`mode='wrap'`), wraps to the opposite edge.
    """
    Vp = _pad_axis(V, axis, periodic)
    Ep = _pad_axis(eps_axis, axis, periodic)
    sl_c = [slice(None)] * V.ndim; sl_c[axis] = slice(1, -1)
    sl_l = [slice(None)] * V.ndim; sl_l[axis] = slice(0, -2)
    sl_r = [slice(None)] * V.ndim; sl_r[axis] = slice(2, None)
    Vc, Vl, Vr = Vp[tuple(sl_c)], Vp[tuple(sl_l)], Vp[tuple(sl_r)]
    Ec, El, Er = Ep[tuple(sl_c)], Ep[tuple(sl_l)], Ep[tuple(sl_r)]
    eps_face_p = 0.5 * (Ec + Er)   # face at +½ (between i and i+1)
    eps_face_m = 0.5 * (Ec + El)   # face at -½ (between i-1 and i)
    return (eps_face_p * (Vc - Vr) + eps_face_m * (Vc - Vl)) / (h * h)


def apply_neg_div_eps_grad(V: jnp.ndarray, eps_diag: jnp.ndarray,
                           spacings: tuple[float, float, float],
                           periodic: tuple[bool, bool, bool] = (False, False, True)
                           ) -> jnp.ndarray:
    """Apply the operator `−∇·(ε·∇)` (diagonal ε) to V at every cell.

    Each spacings[k] is a scalar (uniform axis) or a 1D array of cell widths
    (graded axis — see graded_axis_widths). Result shape = V.shape; the
    discrete operator matrix is symmetric in both cases (face fluxes share
    the same coefficient on both sides; volume form on graded axes).
    """
    out = jnp.zeros_like(V)
    for ax in range(3):
        geom = _axis_geometry(spacings[ax], V.shape[ax], periodic[ax])
        if geom is None:
            out = out + _laplacian_axis(V, eps_diag[ax], spacings[ax], ax,
                                        periodic[ax])
        else:
            out = out + _laplacian_axis_graded(V, eps_diag[ax], geom, ax,
                                               periodic[ax])
    return out



# ===========================================================================
# 2b. Non-uniform (graded) tensor-product grids
#
# `spacings` entries may be a SCALAR (uniform axis, original behaviour) or a
# 1D array of CELL WIDTHS w_i (length n_axis). The operator is assembled in
# finite-volume (volume-weighted) form, which keeps the matrix SYMMETRIC on a
# non-uniform grid: the coupling between cells i,i+1 is
#     c = eps_face * A_transverse / d_{i+1/2},   d_{i+1/2} = (w_i + w_{i+1})/2
# identical seen from both rows. Widths are normalised by a reference width
# so the uniform case reduces EXACTLY to the original 1/h**2 stencil (same
# floats), and Dirichlet identity rows stay O(1)-conditioned.
# ===========================================================================


def graded_axis_widths(n_core: int, w_core: float, n_pad: int,
                       growth: float = 1.3, w_max: float = None):
    """Cell widths for [lo-pad | uniform core | hi-pad] along one axis.

    The padding cells grow geometrically (w_core*growth, *growth**2, ...) away
    from the core, optionally capped at w_max. Returns (widths, i_core_lo,
    i_core_hi) with widths.shape = (n_pad + n_core + n_pad,) and the core
    occupying widths[i_core_lo:i_core_hi].
    """
    import numpy as _np
    ramp = w_core * growth ** _np.arange(1, n_pad + 1)
    if w_max is not None:
        ramp = _np.minimum(ramp, w_max)
    widths = _np.concatenate([ramp[::-1], _np.full(n_core, w_core), ramp])
    return widths, n_pad, n_pad + n_core


def axis_centers(widths) -> "jnp.ndarray":
    """Cell-centre coordinates from cell widths (origin at the low edge)."""
    w = jnp.asarray(widths, dtype=jnp.float64)
    edges = jnp.concatenate([jnp.zeros(1), jnp.cumsum(w)])
    return 0.5 * (edges[:-1] + edges[1:])


def _axis_geometry(spacing, n, periodic):
    """Per-axis (w_hat, dp_hat, dm_hat, w_ref): normalised cell widths and
    +/- face distances (each shape (n,)), or None for a uniform axis.
    Computed in NUMPY — widths are static configuration, and float() on a
    traced array inside CG's jitted matvec would raise ConcretizationError."""
    import numpy as _np
    if _np.ndim(spacing) == 0:
        return None
    w = _np.asarray(spacing, dtype=_np.float64)
    w_ref = float(w.min())
    wh = w / w_ref
    if periodic:
        w_r = _np.roll(wh, -1)
        w_l = _np.roll(wh, 1)
    else:
        w_r = _np.concatenate([wh[1:], wh[-1:]])   # edge-replicated ghosts
        w_l = _np.concatenate([wh[:1], wh[:-1]])
    dp = 0.5 * (wh + w_r)
    dm = 0.5 * (wh + w_l)
    return jnp.asarray(wh), jnp.asarray(dp), jnp.asarray(dm), w_ref


def _bcast_1d(a, axis, ndim):
    shape = [1] * ndim
    shape[axis] = -1
    return a.reshape(shape)


def _laplacian_axis_graded(V, eps_axis, geom, axis, periodic):
    """Volume-form per-axis operator on a graded axis:
        (1/w_i) * [eps_p (Vc - Vr)/dp + eps_m (Vc - Vl)/dm] / w_ref**2
    (transverse areas cancel against the cell volume per axis; Neumann outer
    faces carry zero flux via edge padding exactly as the uniform stencil)."""
    wh, dp, dm, w_ref = geom
    Vp = _pad_axis(V, axis, periodic)
    Ep = _pad_axis(eps_axis, axis, periodic)
    sl_c = [slice(None)] * V.ndim; sl_c[axis] = slice(1, -1)
    sl_l = [slice(None)] * V.ndim; sl_l[axis] = slice(0, -2)
    sl_r = [slice(None)] * V.ndim; sl_r[axis] = slice(2, None)
    Vc, Vl, Vr = Vp[tuple(sl_c)], Vp[tuple(sl_l)], Vp[tuple(sl_r)]
    Ec, El, Er = Ep[tuple(sl_c)], Ep[tuple(sl_l)], Ep[tuple(sl_r)]
    eps_face_p = 0.5 * (Ec + Er)
    eps_face_m = 0.5 * (Ec + El)
    nd = V.ndim
    whb = _bcast_1d(wh, axis, nd)
    dpb = _bcast_1d(dp, axis, nd)
    dmb = _bcast_1d(dm, axis, nd)
    return (eps_face_p * (Vc - Vr) / dpb
            + eps_face_m * (Vc - Vl) / dmb) / (whb * w_ref * w_ref)


# ===========================================================================
# 3. JAX-CG Poisson solve (SPD)
# ===========================================================================

def solve_poisson_jax(eps_diag: jnp.ndarray,
                      spacings: tuple[float, float, float],
                      dirichlet_mask: jnp.ndarray,
                      V_dirichlet: jnp.ndarray,
                      periodic: tuple[bool, bool, bool] = (False, False, True),
                      rtol: float = 1e-6,
                      maxiter: int = 2000) -> jnp.ndarray:
    """Solve `−∇·(ε·∇V) = 0` for V on the (nx, ny, nz) grid.

    Pins V at every cell where `dirichlet_mask` is True (electrodes) to the
    corresponding `V_dirichlet` value. Solves on the rest with Neumann zero-flux
    walls (or periodic on flagged axes).

    SPD construction (so CG converges in O(√κ) iterations):
      * matvec: at Dirichlet cells, A_pp = 1, A_p_other = 0 → identity row.
      * at free cells, neighbours that are Dirichlet contribute 0 to columns
        (projection `v_proj = where(mask, 0, v)` kills them BEFORE applying L).
      * RHS: at Dirichlet, b_p = V_dirichlet[p]; at free p, b_p = −L(v_dir_only)_p
        which equals `Σ_{n dir} coef[p,n] · V_dirichlet[n]` — the bit "moved to
        RHS" when eliminating Dirichlet columns from the LHS.

    The differentiation is handled by `jax.scipy.sparse.linalg.cg`'s built-in
    custom_vjp: backward = transpose-solve on the same operator with the
    incoming cotangent as RHS (linear-solve adjoint trick).

    Returns V of shape eps_diag.shape[1:]  i.e. (nx, ny, nz).
    """
    shape = V_dirichlet.shape

    def L(v: jnp.ndarray) -> jnp.ndarray:
        return apply_neg_div_eps_grad(v, eps_diag, spacings, periodic)

    def matvec(v_flat: jnp.ndarray) -> jnp.ndarray:
        v = v_flat.reshape(shape)
        v_proj = jnp.where(dirichlet_mask, 0.0, v)
        Lv = L(v_proj)
        Av = jnp.where(dirichlet_mask, v, Lv)
        return Av.flatten()

    # b at free p = Σ_{n dir} coef[p,n] V_dir[n] = −L(v_dir_field)_p
    v_dir_field = jnp.where(dirichlet_mask, V_dirichlet, 0.0)
    L_vdir = L(v_dir_field)
    b = jnp.where(dirichlet_mask, V_dirichlet, -L_vdir).flatten()

    # Jacobi (diagonal) preconditioner — essential on graded grids where the
    # cell-volume range inflates the condition number. diag(A) at a free cell
    # is the sum over axes of (eps_face_p/dp + eps_face_m/dm)/(w*wref^2)
    # (uniform axes: (eps_face_p + eps_face_m)/h^2); 1 at Dirichlet rows.
    diag = jnp.zeros(shape, dtype=jnp.float64)
    for ax in range(3):
        geom = _axis_geometry(spacings[ax], shape[ax], periodic[ax])
        Ep = _pad_axis(eps_diag[ax], ax, periodic[ax])
        sl_c = [slice(None)] * 3; sl_c[ax] = slice(1, -1)
        sl_l = [slice(None)] * 3; sl_l[ax] = slice(0, -2)
        sl_r = [slice(None)] * 3; sl_r[ax] = slice(2, None)
        Ec, El, Er = Ep[tuple(sl_c)], Ep[tuple(sl_l)], Ep[tuple(sl_r)]
        ep = 0.5 * (Ec + Er); em = 0.5 * (Ec + El)
        if geom is None:
            h = spacings[ax]
            diag = diag + (ep + em) / (h * h)
        else:
            wh, dp, dm, w_ref = geom
            whb = _bcast_1d(wh, ax, 3)
            dpb = _bcast_1d(dp, ax, 3)
            dmb = _bcast_1d(dm, ax, 3)
            diag = diag + (ep / dpb + em / dmb) / (whb * w_ref * w_ref)
    diag = jnp.where(dirichlet_mask, 1.0, diag)
    inv_diag = (1.0 / diag).flatten()

    def precond(r):
        return inv_diag * r

    # Warm start with the Dirichlet field itself — already satisfies Dirichlet rows.
    x0 = v_dir_field.flatten()
    V_flat, _info = jax.scipy.sparse.linalg.cg(matvec, b, x0=x0, tol=rtol,
                                               maxiter=maxiter, M=precond)
    return V_flat.reshape(shape)


# ===========================================================================
# 4. E = -∇V (central differences with edge replication)
# ===========================================================================

def gradient_V_jax(V: jnp.ndarray, spacings) -> jnp.ndarray:
    """E = -∇V via central differences; one-sided at outer boundaries via edge
    replication. Each spacings[k] is a scalar or a 1D array of cell widths
    (graded axis → differences use the true centre-to-centre distances).
    Returns (3, nx, ny, nz)."""
    grads = []
    for ax in range(3):
        sp = spacings[ax]
        if jnp.ndim(sp) == 0:
            d = jnp.asarray(sp, dtype=V.dtype)
            g = jnp.gradient(V, axis=ax) / d
        else:
            w = jnp.asarray(sp, dtype=V.dtype)
            x = axis_centers(w).astype(V.dtype)
            nd = V.ndim
            xb = _bcast_1d(x, ax, nd)
            Vp = _pad_axis(V, ax, False)
            xp = jnp.concatenate([x[:1] - w[:1], x, x[-1:] + w[-1:]])
            xpb = _bcast_1d(xp, ax, nd)
            sl_c = [slice(None)] * nd; sl_c[ax] = slice(1, -1)
            sl_l = [slice(None)] * nd; sl_l[ax] = slice(0, -2)
            sl_r = [slice(None)] * nd; sl_r[ax] = slice(2, None)
            # centre-to-centre central difference on the non-uniform axis
            g = (Vp[tuple(sl_r)] - Vp[tuple(sl_l)]) / (
                xpb[tuple(sl_r)] - xpb[tuple(sl_l)])
        grads.append(g)
    return -jnp.stack(grads, axis=0)
