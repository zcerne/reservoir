import numpy as np
from itertools import product
from numpy.polynomial import legendre as L


def _legendre_norm(u, n):
    """Orthonormal Legendre P̃ₙ on [-1,1] with uniform measure: E[P̃ₙ(u)²]=1."""
    c = np.zeros(n + 1); c[n] = 1.0
    return np.sqrt(2 * n + 1.0) * L.legval(u, c)


def dambre_ipc(data, max_degree=3, threshold=None, ridge=0.0,
               max_features=None):
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
    max_features : cap on output channels actually used (evenly subsampled).
        Default max(8, M//10): more channels than ~M/10 makes the R² estimate
        degenerate (perfect in-sample fit, noise floor above 1, spectrum all
        zeros). Pass the full count explicitly to opt out.

    Returns dict: ipc_total, ipc_by_degree {d: Σ capacity}, nonlinear_fraction
        (deg≥2 / total), max_degree_present, bound (rank of X = capacity ceiling),
        n_targets, n, f_out (channels available), f_used (channels analysed),
        threshold, linear.
    """
    U = np.asarray(data["inputs"]).real
    X = np.asarray(data["outputs"])
    M, K = U.shape
    Xf = X.reshape(M, -1)
    f_out = Xf.shape[1]

    # Too many channels for the probe count is a DEGENERATE fit: with F ≳ M
    # every target is reconstructed perfectly in-sample (R²→1) and the noise
    # floor 2F/M rises above 1, so every real capacity is zeroed and the whole
    # spectrum reads 0. Sampling F_max evenly-spaced channels — physically, a
    # finite set of detectors on the screen — is what makes the measurement
    # meaningful; the ceiling then sits at min(rank, n_targets), not at F.
    f_cap = max(8, M // 10) if max_features is None else max_features
    if f_out > f_cap:
        Xf = Xf[:, np.linspace(0, f_out - 1, f_cap).astype(int)]
    F = Xf.shape[1]
    # complex features act as TWO real regressors each in the projection —
    # count them as such or the default noise floor sits a factor 2 low and
    # purely linear systems grow phantom high-degree capacity (seen on the
    # dye-free 02_2D_Q_tensor: in-sample deg3=5.4, held-out R2=0).
    F_eff = 2 * F if np.iscomplexobj(Xf) else F
    thr = (2.0 * F_eff / M) if threshold is None else threshold

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
    targets_by_degree = {}
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
        targets_by_degree[deg] = targets_by_degree.get(deg, 0) + 1
        n_targets += 1

    ipc_total = float(sum(ipc_by_degree.values()))
    nl = float(sum(v for d, v in ipc_by_degree.items() if d >= 2))
    present = [d for d, v in ipc_by_degree.items() if v > 1e-9]
    return dict(
        ipc_total=ipc_total, ipc_by_degree=ipc_by_degree,
        targets_by_degree=targets_by_degree,
        nonlinear_fraction=float(nl / (ipc_total + 1e-30)),
        max_degree_present=(max(present) if present else 0),
        bound=int(np.linalg.matrix_rank(Xf)), n_targets=int(n_targets),
        n=int(M), f_out=int(f_out), f_used=int(F), threshold=float(thr),
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
