import numpy as np
from itertools import combinations_with_replacement


def _poly_features(X, degree, include_conj=True):
    """Polynomial (Volterra) features of complex input X (M, K) up to `degree`.

    Uses the augmented vector a = [E, conj(E)] (length 2K when include_conj) so that
    the intensity term |E|² = Σ E_k·conj(E_k) appears as a degree-2 monomial — the
    reservoir's |E|² readout is degree-2 in this basis. Returns (Phi, degs):
    Phi (M, n_feat) includes a constant (degree 0) first column; degs (n_feat,) is the
    monomial degree of each column.
    """
    X = np.asarray(X)
    M, K = X.shape
    a = np.concatenate([X, np.conj(X)], axis=1) if include_conj else X
    cols = [np.ones(M, dtype=complex)]
    degs = [0]
    idx = list(range(a.shape[1]))
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(idx, d):
            f = np.ones(M, dtype=complex)
            for c in combo:
                f = f * a[:, c]
            cols.append(f); degs.append(d)
    return np.stack(cols, axis=1), np.asarray(degs)


def volterra_series(data, degree=2, include_conj=True, test_frac=0.3, rcond=None, seed=0):
    """Nonlinearity Method E — Volterra-series kernels.

    Fit  E_out = G₁·E + G₂·(E⊗E) + G₃·(E⊗E⊗E) + …  (up to `degree`) and resolve the
    nonlinearity BY POLYNOMIAL ORDER: G₁ is the linear part; energy in G₂, G₃, …
    quantifies the nonlinearity and its order. The complete static characterization.

    We fit least-squares kernels over polynomial features of the augmented input
    [E, conj(E)] and report, on a HELD-OUT split, the incremental variance explained
    by each order (nested models: R² using features up to degree d, minus d−1). For a
    passive optical reservoir the FIELD map is pure order 1; the |E|² readout is
    order 2 (E·conj(E)).

    data : dict — inputs (M, K) complex probes, outputs (M, ...) reservoir outputs.
    degree : max Volterra order to fit. include_conj : use [E,E*] (needed for |E|²).
    test_frac : held-out fraction for the R² (guards overfit; **> 0**).

    Returns dict: r2_by_maxdeg {d: held-out R² with features ≤ d}, gain_by_order
        {d: R²(d)−R²(d−1)} (variance explained by exactly order d), linear_fraction,
        nonlinear_fraction (orders ≥2), max_order, kernel_energy {d: ‖W_d‖²},
        n, n_features, degree, linear.
    """
    X = np.asarray(data["inputs"]); Y = np.asarray(data["outputs"])
    M = X.shape[0]
    Yf = Y.reshape(M, -1)
    if not (test_frac and test_frac > 0):
        raise ValueError("test_frac must be > 0")

    Phi, degs = _poly_features(X, degree, include_conj)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M); nte = max(1, int(round(M * test_frac)))
    te, tr = perm[:nte], perm[nte:]
    Yte = Yf[te]
    ybar = Yf[tr].mean(0)
    den = np.linalg.norm(Yte - ybar) ** 2 + 1e-30

    def fit_upto(dmax):
        P = Phi[:, degs <= dmax]
        W = Yf[tr].T @ (np.linalg.pinv(P[tr].T, rcond=rcond) if rcond is not None
                        else np.linalg.pinv(P[tr].T))          # (f_out, n_feat_d)
        Yhat = (W @ P[te].T).T
        return float(1.0 - np.linalg.norm(Yte - Yhat) ** 2 / den), W

    r2_by = {}; Wfull = None
    for d in range(0, degree + 1):
        r2_by[d], W = fit_upto(d)
        if d == degree:
            Wfull = W
    kernel_energy = {d: float(np.sum(np.abs(Wfull[:, degs == d]) ** 2)) for d in range(0, degree + 1)}

    gain = {d: float(r2_by[d] - r2_by.get(d - 1, 0.0)) for d in range(1, degree + 1)}
    total = sum(max(g, 0.0) for g in gain.values()) + 1e-30
    linear_fraction = float(max(gain.get(1, 0.0), 0.0) / total)
    nonlinear_fraction = float(sum(max(gain[d], 0.0) for d in gain if d >= 2) / total)
    sig = [d for d in gain if d >= 2 and gain[d] > 1e-6]
    return dict(
        r2_by_maxdeg=r2_by, gain_by_order=gain, kernel_energy=kernel_energy,
        linear_fraction=linear_fraction, nonlinear_fraction=nonlinear_fraction,
        max_order=(max(sig) if sig else 1), n=int(M), n_features=int(Phi.shape[1]),
        degree=int(degree), linear=bool(nonlinear_fraction < 1e-6),
    )


def report(res):
    g = "  ".join(f"{d}:{res['gain_by_order'][d]:+.3f}" for d in sorted(res["gain_by_order"]))
    return "\n".join([
        f"Volterra series (Method E) | degree≤{res['degree']}, N={res['n']}, "
        f"{res['n_features']} features",
        f"  held-out R² by max-degree: " +
        "  ".join(f"≤{d}:{res['r2_by_maxdeg'][d]:.3f}" for d in sorted(res["r2_by_maxdeg"])),
        f"  variance gain by order:    {g}",
        f"  linear frac={res['linear_fraction']:.3f}  nonlinear frac={res['nonlinear_fraction']:.3f}"
        f"  max order={res['max_order']}",
        f"  ->  {'LINEAR' if res['linear'] else 'NONLINEAR (order %d)' % res['max_order']}",
    ])
