import numpy as np


def linear_residual(data, test_frac=0.3, n_repeats=5, rcond=None, seed=0):
    """Nonlinearity Method B — held-out linear residual (1 − R²).

    Fit the best linear map G on a TRAIN split of the probes and report the fraction
    of output variance no linear map can explain on the HELD-OUT test split:

        1 − R² = ‖Y_te − G·X_te‖² / ‖Y_te − mean(Y_tr)‖².

    0 = perfectly linear, →1 (and >1) = strongly nonlinear. Unlike the superposition
    test (Method A) this needs only a plain {inputs, outputs} probe set — one forward
    run per probe, no structured combinations.

    data : dict with
        inputs  : (N, ...) complex  — probe inputs E_in (one per forward run)
        outputs : (N, ...) [complex] — reservoir outputs. Feed the FIELD E_out → ~0
                  (confirms the physics is linear); feed the |E|² intensity → large
                  (the readout nonlinearity).
    test_frac : held-out fraction. **Required > 0** — with a full-data fit and
        N ≤ n_features the map overfits to ~0 and reads "linear" regardless.
    n_repeats : average 1−R² over this many random train/test splits (stabler).
    rcond     : pinv cutoff (default numpy).

    Returns dict: residual_fraction (mean 1−R²), r2, residual_std, per_repeat,
        n, f_in, f_out, test_frac, linear (< 1e-6), underdetermined (N_train ≤ f_in).
    """
    X = np.asarray(data["inputs"]); Y = np.asarray(data["outputs"])
    N = X.shape[0]
    if Y.shape[0] != N:
        raise ValueError(f"inputs/outputs must share axis 0 ({N} vs {Y.shape[0]})")
    if not (test_frac and test_frac > 0):
        raise ValueError("test_frac must be > 0 — a full-data fit overfits to ~0 (see docstring)")
    Xf = X.reshape(N, -1); Yf = Y.reshape(N, -1)
    f_in, f_out = Xf.shape[1], Yf.shape[1]

    rng = np.random.default_rng(seed)
    res = np.empty(n_repeats)
    n_te = max(1, int(round(N * test_frac)))
    n_tr = N - n_te
    for r in range(n_repeats):
        perm = rng.permutation(N)
        te, tr = perm[:n_te], perm[n_te:]
        Xtr = Xf[tr].T                                          # (f_in, n_tr)
        G = Yf[tr].T @ (np.linalg.pinv(Xtr, rcond=rcond) if rcond is not None
                        else np.linalg.pinv(Xtr))               # (f_out, f_in)
        Yhat = (G @ Xf[te].T).T
        num = np.linalg.norm(Yf[te] - Yhat) ** 2
        den = np.linalg.norm(Yf[te] - Yf[tr].mean(0)) ** 2 + 1e-30
        res[r] = num / den

    mean_r = float(res.mean())
    return dict(
        residual_fraction=mean_r, r2=float(1.0 - mean_r), residual_std=float(res.std()),
        per_repeat=res, n=int(N), f_in=int(f_in), f_out=int(f_out), test_frac=test_frac,
        linear=bool(mean_r < 1e-6), underdetermined=bool(n_tr <= f_in),
    )


def report(res):
    """One-screen summary of a linear_residual result."""
    lines = [
        f"Linear residual (Method B) | N={res['n']} probes, f_in={res['f_in']}, "
        f"f_out={res['f_out']}, test_frac={res['test_frac']}",
        f"  1−R² (held-out) = {res['residual_fraction']:.4e}  (±{res['residual_std']:.1e})   "
        f"->  {'LINEAR' if res['linear'] else 'NONLINEAR'}",
    ]
    if res["underdetermined"]:
        lines.append("  ⚠ underdetermined (n_train ≤ f_in): residual may be optimistic — "
                     "add more probes.")
    return "\n".join(lines)
