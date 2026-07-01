import numpy as np


def best_linear_approx(data, n_modes=None, test_frac=0.0, rcond=None, seed=0):
    """Best Linear Approximation (BLA) of a black-box reservoir + SVD.

    Extract the best least-squares linear coupling operator G from input/output
    complex-field data, then SVD it for the channel / capacity analysis. A held-out
    fit residual doubles as a nonlinearity measure.

    data : dict with
        data["inputs"]  : complex array (n_inputs, n_y, n_z, 3)  — probe input fields
        data["outputs"] : complex array (n_inputs, n_y, n_z, 3)  — measured outputs
      Works for 2D and 3D: everything after axis 0 is flattened, so 2D is just
      n_z = 1 (or shape (n_inputs, n_y, 3)); input and output planes may differ.

    n_modes   : keep top-n singular vectors in the returned modes (default: all).
    test_frac : if >0, fit G on (1-test_frac) of the probes and score the residual
                on the held-out rest -> the honest nonlinearity (1-R^2). With
                test_frac=0 and n_inputs <= n_features the fit overfits to ~0, so a
                split is required to read nonlinearity.

    Returns dict: G, s, power, n_eff, sum_rule, cond, n_significant, rank,
        n_inputs, f_in, f_out, U, Vh, input_modes, output_modes, residual_fraction.
    """
    inputs = np.asarray(data["inputs"])
    outputs = np.asarray(data["outputs"])
    n_inputs = inputs.shape[0]
    if outputs.shape[0] != n_inputs:
        raise ValueError(f"inputs/outputs must share axis 0 ({n_inputs} vs {outputs.shape[0]})")
    in_shape, out_shape = inputs.shape[1:], outputs.shape[1:]

    # flatten each field to a complex feature vector: (n_inputs, n_features)
    Xin = inputs.reshape(n_inputs, -1)
    Yout = outputs.reshape(n_inputs, -1)

    # --- optional held-out residual = nonlinearity (1 - R^2) ---
    residual_fraction = None
    if test_frac and test_frac > 0:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_inputs)
        n_te = max(1, int(round(n_inputs * test_frac)))
        te, tr = perm[:n_te], perm[n_te:]
        G_tr = Yout[tr].T @ np.linalg.pinv(Xin[tr].T, rcond=rcond) if rcond is not None \
            else Yout[tr].T @ np.linalg.pinv(Xin[tr].T)
        Yhat = (G_tr @ Xin[te].T).T
        num = np.linalg.norm(Yout[te] - Yhat) ** 2
        den = np.linalg.norm(Yout[te] - Yout[tr].mean(0)) ** 2 + 1e-30
        residual_fraction = float(num / den)

    # --- BLA on all data: G = Y * pinv(X) ---
    X, Y = Xin.T, Yout.T                                      # (f_in, n), (f_out, n)
    Xpinv = np.linalg.pinv(X, rcond=rcond) if rcond is not None else np.linalg.pinv(X)
    G = Y @ Xpinv                                             # (f_out, f_in)

    # --- SVD -> channels ---
    U, s, Vh = np.linalg.svd(G, full_matrices=False)
    s2 = s ** 2
    n_eff = float((s2.sum() ** 2) / (np.square(s2).sum() + 1e-30))   # participation ratio
    sum_rule = float(np.sum(np.abs(G) ** 2))                          # = sum |s|^2
    throughput = float(s2.sum())                                      # total linear power Σ|s|² (== sum_rule)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    n_significant = int(np.sum(s2 >= 0.01 * (s2[0] + 1e-30)))
    rank = int(np.sum(s > 1e-9 * (s[0] + 1e-30)))

    # --- communication modes, reshaped back to field shape ---
    k = len(s) if n_modes is None else min(n_modes, len(s))
    input_modes = Vh[:k].conj().reshape((k,) + in_shape)      # psi_S (input plane)
    output_modes = U[:, :k].T.reshape((k,) + out_shape)       # phi_R (output plane)

    return dict(
        G=G, s=s, power=s2 / (s2[0] + 1e-30),
        n_eff=n_eff, sum_rule=sum_rule, throughput=throughput, cond=cond,
        n_significant=n_significant, rank=rank,
        n_inputs=n_inputs, f_in=X.shape[0], f_out=Y.shape[0],
        U=U, Vh=Vh, input_modes=input_modes, output_modes=output_modes,
        residual_fraction=residual_fraction,
    )


def report(res):
    """One-screen human summary of a best_linear_approx result."""
    lines = [
        f"BLA + SVD | {res['n_inputs']} probes, f_in={res['f_in']}, f_out={res['f_out']}, rank={res['rank']}",
        f"  singular values (top 8): {np.array2string(res['s'][:8], precision=4)}",
        f"  power |s|^2 (top 8):     {np.array2string(res['power'][:8], precision=3)}",
        f"  effective channels n_eff = {res['n_eff']:.3f}",
        f"  significant (|s|^2 >= 1% max) = {res['n_significant']}",
        f"  sum rule S = sum|s|^2 = {res['sum_rule']:.4g}",
        f"  throughput (total linear power Σ|s|^2) = {res['throughput']:.4g}",
        f"  condition number = {res['cond']:.3g}",
    ]
    if res["residual_fraction"] is not None:
        lines.append(f"  nonlinearity 1-R^2 (held-out) = {res['residual_fraction']:.4f}  (0=linear)")
    else:
        lines.append("  (pass test_frac>0 to measure nonlinearity 1-R^2)")
    return "\n".join(lines)
