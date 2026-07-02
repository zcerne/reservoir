"""Nonlinearity Method G — dimensionality expansion (output-rank growth with inputs).

A LINEAR map f: C^K → C^M preserves input dimensionality: the output lives in a
subspace of rank ≤ K. A NONLINEAR readout like |E|² INFLATES dimensionality: the
output after squaring lives in the space of all monomials Re(x_i x_j*), Im(x_i x_j*),
Re(x_i), Im(x_i) — up to ~K²+K effective dimensions.

Two complementary measures (both from a single ipc dataset):
  1. Full PCA on the output matrix → PR (participation ratio), d99, spectrum.
     Field ≈ K components; |E|² ≫ K components.
  2. Linear-fit R²(k) — for each k=1..K, fit the best LINEAR model using only the
     first k input channels to predict ALL outputs, and measure held-out R².
     LINEAR system: R²→1.0 at k=K (all variance explained by K inputs).
     |E|²: R² plateaus well below 1.0 (the linear model CAN'T capture the
     nonlinear cross-terms, no matter how many input channels you give it).

Returns dict: pr, d99, cum_explained (full PCA), r2_vs_k (linear-fit R² by input dim),
    max_k, linear (True if R²→1), plateau_r2 (R² at max_k), eigenvalues.
"""

import numpy as np


def dimension_expansion(data, max_k=None, test_frac=0.3, n_repeats=5, seed=0):
    """PCA dimensionality + linear-fit R²(k) from existing ipc dataset.

    Parameters
    ----------
    data : dict with keys "inputs" (N, K) real, "outputs" (N, M) complex
    max_k : int or None
        Maximum input dimension to test (default: all K).
    test_frac : float
        Held-out fraction for the linear-fit R².
    n_repeats : int
        Number of random splits per k (R² is averaged).

    Returns
    -------
    dict with pr, d99, eigenvalues, cum_explained, r2_vs_k, plateau_r2,
        max_k, linear.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(data["inputs"]).astype(np.float64)       # (N, K)
    Y = np.asarray(data["outputs"])                         # (N, M) complex
    N, K = X.shape; M = Y.shape[1]
    if max_k is None:
        max_k = K
    max_k = min(max_k, K)

    # ---- 1. Full-output PCA ----
    Ym = Y - Y.mean(0, keepdims=True)
    _, S, _ = np.linalg.svd(Ym, full_matrices=False)
    lam = S**2; lam = lam[lam > 1e-30]
    total = float(lam.sum())
    pr = float(total**2 / (lam**2).sum()) if total > 0 else 0.0
    cum = np.cumsum(lam) / (total + 1e-30)
    d99 = int(np.searchsorted(cum, 0.99) + 1)

    # ---- 2. Linear-fit R²(k) — how much variance can k linear inputs explain? ----
    r2_vs_k = {}
    for k in range(1, max_k + 1):
        r2s = []
        for _ in range(n_repeats):
            ix = rng.permutation(N); n_te = int(N * test_frac)
            X_tr, Y_tr = X[ix[n_te:], :k], Y[ix[n_te:]]
            X_te, Y_te = X[ix[:n_te], :k], Y[ix[:n_te]]
            # linear least squares: Y ≈ X @ beta  (real X → complex Y)
            # beta = (X^T X)^-1 X^T Y, real-valued pseudoinverse
            try:
                beta = np.linalg.lstsq(X_tr, Y_tr, rcond=None)[0]  # (k, M) complex
            except np.linalg.LinAlgError:
                r2s.append(0.0); continue
            Yp = X_te @ beta
            num = np.linalg.norm(Y_te - Yp)**2
            den = np.linalg.norm(Y_te - Y_tr.mean(0))**2 + 1e-30
            r2s.append(max(0.0, 1.0 - float(num / den)))
        r2_vs_k[k] = float(np.mean(r2s))

    plateau_r2 = r2_vs_k.get(max_k, 0.0)
    # linear if R² plateaus near 1.0 with K inputs (all variance = linear in inputs)
    linear = bool(plateau_r2 > 0.999)

    return dict(
        pr=pr, d99=d99, eigenvalues=lam[:max(2*K+2, min(20, len(lam)))],
        cum_explained=cum[:max(2*K+2, min(20, len(cum)))],
        r2_vs_k=r2_vs_k, plateau_r2=plateau_r2,
        max_k=int(max_k), linear=linear,
        n_inputs=int(K), n_outputs=int(M),
    )


def report(res):
    """One-line summary."""
    k = res["max_k"]
    lines = [f"Dimension expansion (Method G) | {res['n_inputs']}→{res['n_outputs']} "
             f"| {'LINEAR' if res['linear'] else 'NONLINEAR'}"]
    lines.append(f"  PCA: pr={res['pr']:.2f}  d99={res['d99']}  "
                 f"ceil=k(k+1)/2+k={k*(k+1)//2+k}")
    lines.append(f"  linear-fit R²(k): " +
                 " ".join(f"k{k}:{res['r2_vs_k'][k]:.4f}" for k in sorted(res["r2_vs_k"])))
    lines.append(f"  plateau R²(k={k}) = {res['plateau_r2']:.4f}")
    return "\n".join(lines)
