import jax.numpy as jnp
from functools import partial
from jax import jit


@jit
def n_sph(phi, theta):
    x = jnp.cos(phi) * jnp.sin(theta)
    y = jnp.sin(phi) * jnp.sin(theta)
    z = jnp.cos(theta)
    return jnp.asarray((x, y, z))


@partial(jit, static_argnames=['spacings_'])
def construct_jacobian(nvf_, spacings_):
    nx, ny, nz = nvf_[0], nvf_[1], nvf_[2]
    gx = jnp.gradient(nx, spacings_[0])
    gy = jnp.gradient(ny, spacings_[0])
    gz = jnp.gradient(nz, spacings_[0])
    nvf_x = jnp.asarray(gx).reshape(3, -1)
    nvf_y = jnp.asarray(gy).reshape(3, -1)
    nvf_z = jnp.asarray(gz).reshape(3, -1)
    return jnp.asarray((nvf_x, nvf_y, nvf_z))


@partial(jit, static_argnames=['spacings_', 'pad_', 'pad_ixs_'])
def construct_jacobian_periodic(nvf_, spacings_, pad_, pad_ixs_):
    nx, ny, nz = nvf_[0], nvf_[1], nvf_[2]
    nx = jnp.pad(nx, pad_, mode="wrap")
    ny = jnp.pad(ny, pad_, mode="wrap")
    nz = jnp.pad(nz, pad_, mode="wrap")
    gx = jnp.gradient(nx, spacings_[0])
    gy = jnp.gradient(ny, spacings_[0])
    gz = jnp.gradient(nz, spacings_[0])
    ix = (slice(pad_ixs_[i][0], pad_ixs_[i][1]) for i in range(3))
    sl = (slice(None), *ix)
    nvf_x = jnp.asarray(gx)[sl].reshape(3, -1)
    nvf_y = jnp.asarray(gy)[sl].reshape(3, -1)
    nvf_z = jnp.asarray(gz)[sl].reshape(3, -1)
    return jnp.asarray((nvf_x, nvf_y, nvf_z))


@jit
def _divergence(jac_):
    return jnp.trace(jac_)


@jit
def _curl(jac_):
    x = jac_[1, 2] - jac_[2, 1]
    y = jac_[2, 0] - jac_[0, 2]
    z = jac_[0, 1] - jac_[1, 0]
    return jnp.array((-x, -y, -z))


@partial(jit, static_argnames=('constants_', 'spacings_'))
def fe_core_director(nvf_, constants_, spacings_):
    """Frank free energy. constants_ = (k1, k2, k3, q0)."""
    nv = nvf_.reshape(3, -1)
    jac = construct_jacobian(nvf_, spacings_)
    cur = _curl(jac)
    f = 0
    if constants_[0]:
        f += constants_[0] * jnp.power(_divergence(jac), 2)
    if constants_[1]:
        f += constants_[1] * jnp.power(jnp.sum(jnp.multiply(nv, cur), axis=0) - constants_[3], 2)
    if constants_[2]:
        bend = jnp.cross(nv.T, cur.T).T
        f += constants_[2] * jnp.sum(jnp.multiply(bend, bend), axis=0)
    return jnp.sum(f)


@partial(jit, static_argnames=('constants_', 'spacings_', 'pad_', 'pad_ixs_'))
def fe_core_director_periodic(nvf_, constants_, spacings_, pad_, pad_ixs_):
    """Frank free energy with periodic BC. constants_ = (k1, k2, k3, q0)."""
    nv = nvf_.reshape(3, -1)
    jac = construct_jacobian_periodic(nvf_, spacings_, pad_, pad_ixs_)
    cur = _curl(jac)
    f = 0
    if constants_[0]:
        f += constants_[0] * jnp.power(_divergence(jac), 2)
    if constants_[1]:
        f += constants_[1] * jnp.power(jnp.sum(jnp.multiply(nv, cur), axis=0) - constants_[3], 2)
    if constants_[2]:
        bend = jnp.cross(nv.T, cur.T).T
        f += constants_[2] * jnp.sum(jnp.multiply(bend, bend), axis=0)
    return jnp.sum(f)


# === Forward+backward Frank-Oseen (catches Nyquist mode that central is blind to) ===
# Central diff `jnp.gradient` is anti-symmetric → cancels the 1-cell-checkerboard
# mode → no energy penalty for spurious sub-grid director oscillations. Forward
# and backward differences are asymmetric → each registers Nyquist as |2/h| per
# cell → ½·E[fwd] + ½·E[bwd] gives the physics-correct elastic energy without
# changing behaviour on well-resolved smooth fields. See lattice fermion
# doubling for the same math.

def _diff_fwd_axis(field, axis, h):
    d = jnp.diff(field, axis=axis) / h
    edge = jnp.take(d, jnp.asarray([-1]), axis=axis)
    return jnp.concatenate([d, edge], axis=axis)


def _diff_bwd_axis(field, axis, h):
    d = jnp.diff(field, axis=axis) / h
    edge = jnp.take(d, jnp.asarray([0]), axis=axis)
    return jnp.concatenate([edge, d], axis=axis)


def _build_jacobian_one_sided(nvf_, spacings_, diff_fn):
    """Same (3 comp, 3 axis, n_total) layout as construct_jacobian, with one-sided diff."""
    def grad3(field):
        return jnp.stack([diff_fn(field, axis=a, h=spacings_[0]) for a in range(3)], axis=0)
    return jnp.asarray((grad3(nvf_[0]).reshape(3, -1),
                        grad3(nvf_[1]).reshape(3, -1),
                        grad3(nvf_[2]).reshape(3, -1)))


@partial(jit, static_argnames=('constants_', 'spacings_'))
def fe_core_director_fwdbwd(nvf_, constants_, spacings_, fields_=None):
    """Frank free energy with ½·E[fwd grad] + ½·E[bwd grad] discretization.

    constants_ : (k1, k2, k3, q0) — no E-field 4-tuple
              or (k1, k2, k3, q0, eps_a) — with E-field 5-tuple
    fields_   : optional E array. shape (3,) uniform or (3, nx, ny, nz) varying.
                When present + constants is 5-tuple, adds F_E = −½ε_a(n·E)².
    """
    nv = nvf_.reshape(3, -1)

    def _energy_for_jac(jac):
        cur = _curl(jac)
        f = 0
        if constants_[0]:
            f = f + constants_[0] * jnp.power(_divergence(jac), 2)
        if constants_[1]:
            f = f + constants_[1] * jnp.power(jnp.sum(jnp.multiply(nv, cur), axis=0) - constants_[3], 2)
        if constants_[2]:
            bend = jnp.cross(nv.T, cur.T).T
            f = f + constants_[2] * jnp.sum(jnp.multiply(bend, bend), axis=0)
        return jnp.sum(f)

    jac_fwd = _build_jacobian_one_sided(nvf_, spacings_, _diff_fwd_axis)
    jac_bwd = _build_jacobian_one_sided(nvf_, spacings_, _diff_bwd_axis)
    f_total = 0.5 * _energy_for_jac(jac_fwd) + 0.5 * _energy_for_jac(jac_bwd)

    if fields_ is not None and len(constants_) >= 5:
        eps_a = constants_[4]
        E = fields_.reshape(3, 1) if fields_.ndim == 1 else fields_.reshape(3, -1)
        nE = jnp.sum(jnp.multiply(nv, E), axis=0)
        f_total = f_total + jnp.sum(-0.5 * eps_a * jnp.power(nE, 2))
    return f_total


# === Q-tensor (Landau–de Gennes) free energy ==============================
# Optional alternative to the director (Frank–Oseen) core above. Works on the 5
# independent components q5 = (Qxx, Qxy, Qxz, Qyy, Qyz); Qzz = −Qxx − Qyy (Q is
# symmetric + traceless). Same fwdbwd (Nyquist-safe) elastic discretization as
# fe_core_director_fwdbwd. Director (n) and scalar order (S) are recovered by
# eigendecomposition downstream — see qtensor_to_director().

def _gradsq_fwdbwd(field, h):
    """Per-cell ½(|∇_fwd|² + |∇_bwd|²) of a scalar 3D field, summed over 3 axes.
    Mirrors the ½·E[fwd]+½·E[bwd] trick so the 1-cell Nyquist mode is penalized."""
    tot = 0.0
    for a in range(3):
        df = _diff_fwd_axis(field, a, h)
        db = _diff_bwd_axis(field, a, h)
        tot = tot + 0.5 * (df ** 2 + db ** 2)
    return tot


@partial(jit, static_argnames=('constants_', 'spacings_'))
def fe_core_qtensor(q5_, constants_, spacings_, fields_=None):
    """Landau–de Gennes free energy on q5 = (Qxx, Qxy, Qxz, Qyy, Qyz).

    constants_ : (L, A, B, C)          — one-constant elastic L + LdG bulk A,B,C
              or (L, A, B, C, eps_a)    — adds electric coupling F_E = −½·eps_a·EᵀQE
    spacings_  : grid spacings (uses spacings_[0], matching the director cores).
    fields_    : optional E array, (3,) uniform or (3, nx, ny, nz) varying.
    q5_        : shape (5,) + gshape.

    Energy density:
      f = ½L|∇Q|²                                   (elastic, one-constant)
        + ½A·trQ² + ⅓B·trQ³ + ¼C·(trQ²)²            (LdG bulk; sets S thermodynamically)
        − ½·eps_a·EᵀQE                              (electric, if fields_ given)
    Uses the traceless identity trQ³ = 3·detQ ⇒ ⅓B·trQ³ = B·detQ.
    """
    h = spacings_[0]
    Qxx, Qxy, Qxz, Qyy, Qyz = q5_[0], q5_[1], q5_[2], q5_[3], q5_[4]
    Qzz = -Qxx - Qyy
    L, A, B, C = constants_[0], constants_[1], constants_[2], constants_[3]

    # elastic ½L|∇Q|² (diagonal once, off-diagonal twice)
    elastic = (_gradsq_fwdbwd(Qxx, h) + _gradsq_fwdbwd(Qyy, h) + _gradsq_fwdbwd(Qzz, h)
               + 2.0 * (_gradsq_fwdbwd(Qxy, h) + _gradsq_fwdbwd(Qxz, h) + _gradsq_fwdbwd(Qyz, h)))
    f = 0.5 * L * elastic

    # bulk invariants
    trQ2 = Qxx ** 2 + Qyy ** 2 + Qzz ** 2 + 2.0 * (Qxy ** 2 + Qxz ** 2 + Qyz ** 2)
    detQ = (Qxx * (Qyy * Qzz - Qyz ** 2)
            - Qxy * (Qxy * Qzz - Qyz * Qxz)
            + Qxz * (Qxy * Qyz - Qyy * Qxz))
    f = f + 0.5 * A * trQ2 + B * detQ + 0.25 * C * trQ2 ** 2

    # electric coupling −½ eps_a EᵀQE
    if fields_ is not None and len(constants_) >= 5:
        eps_a = constants_[4]
        E = fields_
        Ex, Ey, Ez = E[0], E[1], E[2]
        EQE = (Ex * Ex * Qxx + Ey * Ey * Qyy + Ez * Ez * Qzz
               + 2.0 * (Ex * Ey * Qxy + Ex * Ez * Qxz + Ey * Ez * Qyz))
        f = f - 0.5 * eps_a * EQE
    return jnp.sum(f)


def ldg_constants_from_frank(k1, k2, k3, S_eq, eps_a=None):
    """Map Frank constants → one-constant LdG (L, A, B, C) calibrated so the bulk
    minimum sits at the given S_eq. One-constant elastic: L = 2K̄ / (9 S_eq²/2),
    with K̄ = (k1+k2+k3)/3 (so ½L|∇Q|² reproduces K̄|∇n|² at fixed S). Bulk A,B,C
    are fixed (up to scale) by requiring the uniaxial bulk minimum at S_eq:
    choose C=1 scale, then B = −3·A/S_eq and A from the well depth — here we use the
    standard 5CB-style ratio A:B:C giving S_eq = (−B + √(B²−24AC))/(6C). Returns a
    4- or 5-tuple matching fe_core_qtensor's constants_."""
    Kbar = (k1 + k2 + k3) / 3.0
    L = 2.0 * Kbar / (9.0 * S_eq ** 2 / 2.0)
    # Uniaxial bulk f(S) = (3/4)A S² − (1/4)B S³ + (9/16)C S⁴ (with Q=S(nn−I/3)).
    # Minimum at S_eq with curvature set by C=1: B = (8/ S_eq)·(... ) — use the
    # standard relation S_eq = (B + √(B² + 24|A|C))/(6C) inverted with C=1, A<0.
    C = 1.0
    # pick A so the well depth is O(1); B from the S_eq condition (df/dS=0):
    #   (3/2)A − (3/4)B S_eq + (9/4)C S_eq² = 0  ⇒  B = 2A/S_eq + 3 C S_eq.
    A = -1.0
    B = 2.0 * A / S_eq + 3.0 * C * S_eq
    out = (float(L), float(A), float(B), float(C))
    return out if eps_a is None else out + (float(eps_a),)


def qtensor_to_director(q5_):
    """Recover (phi, theta, S) from q5 by eigendecomposition of Q (largest
    eigenvalue → director). q5_ shape (5,)+gshape → returns three gshape arrays.
    S = (3/2)·λ_max (uniaxial scalar order). Pure-JAX, batched over cells."""
    Qxx, Qxy, Qxz, Qyy, Qyz = (q5_[i] for i in range(5))
    Qzz = -Qxx - Qyy
    gshape = Qxx.shape
    Q = jnp.stack([
        jnp.stack([Qxx, Qxy, Qxz], axis=-1),
        jnp.stack([Qxy, Qyy, Qyz], axis=-1),
        jnp.stack([Qxz, Qyz, Qzz], axis=-1),
    ], axis=-2).reshape(-1, 3, 3)                      # (ncell, 3, 3)
    w, v = jnp.linalg.eigh(Q)                          # ascending eigenvalues
    n = v[:, :, -1]                                    # director = largest-λ eigenvector
    S = 1.5 * w[:, -1]
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    phi = jnp.arctan2(ny, nx).reshape(gshape)
    theta = jnp.arccos(jnp.clip(nz, -1.0, 1.0)).reshape(gshape)
    return phi, theta, S.reshape(gshape)
