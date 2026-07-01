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

    Result shape = V.shape. Symmetric in the sense that the discrete operator
    matrix is symmetric (face fluxes share the same coefficient on both sides).
    """
    return ( _laplacian_axis(V, eps_diag[0], spacings[0], 0, periodic[0])
           + _laplacian_axis(V, eps_diag[1], spacings[1], 1, periodic[1])
           + _laplacian_axis(V, eps_diag[2], spacings[2], 2, periodic[2]))


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

    # Warm start with the Dirichlet field itself — already satisfies Dirichlet rows.
    x0 = v_dir_field.flatten()
    V_flat, _info = jax.scipy.sparse.linalg.cg(matvec, b, x0=x0, tol=rtol, maxiter=maxiter)
    return V_flat.reshape(shape)


# ===========================================================================
# 4. E = -∇V (central differences with edge replication)
# ===========================================================================

def gradient_V_jax(V: jnp.ndarray, spacings: tuple[float, float, float]) -> jnp.ndarray:
    """E = -∇V via central differences; one-sided at outer boundaries via edge
    replication. Returns (3, nx, ny, nz)."""
    dx, dy, dz = spacings
    # jnp.gradient returns a list (one per axis), each shape == V.shape, central
    # differences interior + one-sided at boundaries — same convention as numpy.
    grads = jnp.gradient(V, dx, dy, dz)
    return -jnp.stack(grads, axis=0)
