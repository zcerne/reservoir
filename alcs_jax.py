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
