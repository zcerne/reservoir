import numpy as np
from itertools import product
from numpy.polynomial import legendre as L


def _legendre_norm(u, n):
    """Orthonormal Legendre P̃ₙ on [-1,1] with uniform measure: E[P̃ₙ(u)²]=1."""
    c = np.zeros(n + 1); c[n] = 1.0
    return np.sqrt(2 * n + 1.0) * L.legval(u, c)


def dambre_ipc(data, max_degree=3, threshold=None, ridge=0.0):
    """Nonlinearity Method F — Dambre Information Processing Capacity (gold standard).

    Reconstruct a complete orthonormal basis of polynomial functions of the input and
    measure how much of each the reservoir can LINEARLY reconstruct. Total capacity is
    bounded by the number of independent output channels; the fraction at degree ≥ 2
    is the rigorous nonlinearity measure, and the degree spectrum gives the order.

    For this (single-shot, spatial) reservoir the targets are products of orthonormal
    Legendre polynomials of the input CHANNELS u₁..u_K (Dambre's delayed-input products
    with the delay axis collapsed): y_d(u) = Πₖ P̃_{dₖ}(uₖ), degree = Σₖ dₖ. Capacity of
    a target = R² of the best linear readout of the reservoir output onto it:
        C[y] = ‖proj_X(y)‖² / ‖y‖²  ∈ [0,1].

    **Inputs must be i.i.d. ~ Uniform[-1,1] per channel** for the Legendre family to be
    orthonormal (generate with generate_ipc_data.py). Outputs are the reservoir readout
    state (use the |E|² intensity — capacity is a property of the nonlinear readout).

    data : dict — inputs (M, K) REAL in [-1,1], outputs (M, F) reservoir states.
    max_degree : highest total polynomial degree in the target family.
    threshold : capacities below this are zeroed (noise floor; default 2·F/M ≈ the
        finite-sample bias of R² for F regressors on M samples).
    ridge : optional Tikhonov λ for the readout fit (stabilizes when F ≳ M).

    Returns dict: ipc_total, ipc_by_degree {d: Σ capacity}, nonlinear_fraction
        (deg≥2 / total), max_degree_present, bound (rank of X = capacity ceiling),
        n_targets, n, f_out, threshold, linear.
    """
    U = np.asarray(data["inputs"]).real
    X = np.asarray(data["outputs"])
    M, K = U.shape
    Xf = X.reshape(M, -1)
    F = Xf.shape[1]
    thr = (2.0 * F / M) if threshold is None else threshold

    # readout design: reservoir states + bias, projected via lstsq/ridge
    A = np.concatenate([Xf, np.ones((M, 1))], axis=1)          # (M, F+1)
    if ridge > 0:
        AtA = A.conj().T @ A + ridge * np.eye(A.shape[1])
        Pinv = np.linalg.solve(AtA, A.conj().T)                # (F+1, M)
    else:
        Pinv = np.linalg.pinv(A)                               # (F+1, M)

    def capacity(y):
        y = y - y.mean()
        yn = np.linalg.norm(y) ** 2 + 1e-30
        yhat = A @ (Pinv @ y)
        return float(np.real(np.vdot(yhat, yhat)) / yn)        # ‖proj‖²/‖y‖² = R²

    # enumerate multi-indices d=(d1..dK), 1 ≤ Σdk ≤ max_degree
    ipc_by_degree = {}
    n_targets = 0
    for combo in product(range(max_degree + 1), repeat=K):
        deg = sum(combo)
        if deg < 1 or deg > max_degree:
            continue
        y = np.ones(M)
        for k, dk in enumerate(combo):
            if dk:
                y = y * _legendre_norm(U[:, k], dk)
        c = capacity(y)
        c = c if c > thr else 0.0                              # noise-floor threshold
        ipc_by_degree[deg] = ipc_by_degree.get(deg, 0.0) + c
        n_targets += 1

    ipc_total = float(sum(ipc_by_degree.values()))
    nl = float(sum(v for d, v in ipc_by_degree.items() if d >= 2))
    present = [d for d, v in ipc_by_degree.items() if v > 1e-9]
    return dict(
        ipc_total=ipc_total, ipc_by_degree=ipc_by_degree,
        nonlinear_fraction=float(nl / (ipc_total + 1e-30)),
        max_degree_present=(max(present) if present else 0),
        bound=int(np.linalg.matrix_rank(Xf)), n_targets=int(n_targets),
        n=int(M), f_out=int(F), threshold=float(thr),
        linear=bool(nl < 1e-6),
    )


def report(res):
    byd = "  ".join(f"deg{d}:{res['ipc_by_degree'][d]:.2f}" for d in sorted(res["ipc_by_degree"]))
    return "\n".join([
        f"Dambre IPC (Method F) | N={res['n']} probes, {res['f_out']} outputs, "
        f"{res['n_targets']} targets, thr={res['threshold']:.3g}",
        f"  IPC total = {res['ipc_total']:.3f}   (ceiling = rank(X) = {res['bound']})",
        f"  by degree: {byd}",
        f"  nonlinear fraction (deg≥2) = {res['nonlinear_fraction']:.3f}   "
        f"max degree present = {res['max_degree_present']}",
        f"  ->  {'LINEAR' if res['linear'] else 'NONLINEAR'}",
    ])
