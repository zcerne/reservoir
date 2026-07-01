import numpy as np


def covariance_pca(data, thresh=0.01, var_targets=(0.90, 0.99)):
    """Method C — output-covariance PCA (model-free effective dimensionality).

    Unlike A (Best Linear Approximation), this assumes NO model: it just asks how
    many dimensions the set of output fields actually spans. Eigendecompose the
    output covariance (via SVD of the centered data) -> the number of significant
    eigenvalues = effective output dimensionality, valid whether the reservoir is
    linear or not.

    Why it complements A: a *linear* map can't produce more output modes than input
    modes, so A's channel count is capped by the input rank. A *nonlinear* map
    creates new independent output features (products/powers of the inputs). PCA
    sees those as extra significant dimensions, so comparing C's effective output
    dim to the input dim is a direct signature of the nonlinear feature-expansion
    (returned as `expansion_ratio` when inputs are provided).

    data : dict with
        data["outputs"] : complex array (n_samples, n_y, n_z, 3)  — measured output
                          fields, one per probe.
        data["inputs"]  : (optional) same shape — to also report input dimensionality
                          and the input->output expansion ratio.
      n_samples = number of probes (one driven input -> one measured output field).
      2D and 3D both work (everything after axis 0 is flattened; 2D = n_z 1).

    thresh : fraction-of-max-eigenvalue bar for `n_significant`.
    var_targets : cumulative-variance levels to report channel counts for.

    Returns dict: eigenvalues (variance per PC, descending), explained_var_ratio,
        n_eff (effective output dimensionality = participation ratio), n_significant,
        cum_counts {target: k}, components (top PCs reshaped to field shape),
        mean_field, n_samples, f_out (features per output field = n_y*n_z*3),
        and (if inputs given) in_n_eff, in_f, expansion_ratio.
    """
    outputs = np.asarray(data["outputs"])
    n_samples = outputs.shape[0]                            # number of probes (one output field each)
    out_shape = outputs.shape[1:]
    Y = outputs.reshape(n_samples, -1)                      # (n_samples, f_out) complex

    eig, evr, n_eff, n_sig, comps, cum, mean = _pca_spectrum(Y, thresh, var_targets)
    res = dict(
        eigenvalues=eig, explained_var_ratio=evr, n_eff=float(n_eff),
        n_significant=int(n_sig), cum_counts=cum,
        components=comps.reshape((comps.shape[0],) + out_shape),
        mean_field=mean.reshape(out_shape),
        n_samples=n_samples, f_out=Y.shape[1],
    )

    # optional input-side PCA + expansion ratio (nonlinear-lift signature)
    if "inputs" in data and data["inputs"] is not None:
        Xin = np.asarray(data["inputs"]).reshape(n_samples, -1)
        _, _, in_neff, _, _, _, _ = _pca_spectrum(Xin, thresh, var_targets)
        res["in_n_eff"] = float(in_neff)
        res["in_f"] = Xin.shape[1]
        res["expansion_ratio"] = float(n_eff / (in_neff + 1e-30))
    return res


def _pca_spectrum(Y, thresh, var_targets, n_keep=8):
    """Centered-data SVD -> covariance eigenvalues + effective-rank metrics.
    Handles complex data and tall/wide matrices efficiently."""
    mean = Y.mean(axis=0)
    Yc = Y - mean
    n = Yc.shape[0]
    # eigenvalues of the covariance = (singular values of centered data)^2 / (n-1)
    s = np.linalg.svd(Yc, compute_uv=False)
    eig = (s ** 2) / max(n - 1, 1)
    total = eig.sum() + 1e-30
    evr = eig / total
    n_eff = (eig.sum() ** 2) / (np.square(eig).sum() + 1e-30)   # participation ratio
    n_sig = int(np.sum(eig >= thresh * (eig[0] + 1e-30)))
    cum = np.cumsum(evr)
    cum_counts = {float(t): int(np.searchsorted(cum, t) + 1) for t in var_targets}
    # principal directions (right singular vectors) for the leading components
    k = min(n_keep, Yc.shape[0], Yc.shape[1])
    _, _, Vh = np.linalg.svd(Yc, full_matrices=False)
    comps = Vh[:k].conj()                                        # (k, f) field-space PCs
    return eig, evr, n_eff, n_sig, comps, cum_counts, mean


def report(res):
    """One-screen human summary of a covariance_pca result."""
    lines = [
        f"Output-covariance PCA | {res['n_samples']} probes, f_out={res['f_out']} (features/field)",
        f"  eigenvalues (top 8):       {np.array2string(res['eigenvalues'][:8], precision=4)}",
        f"  explained-var ratio (top8):{np.array2string(res['explained_var_ratio'][:8], precision=3)}",
        f"  effective output dim n_eff = {res['n_eff']:.3f}",
        f"  significant (>= {0.01:.0%} max) = {res['n_significant']}",
        f"  channels for cumulative variance: " +
        ", ".join(f"{int(t*100)}%->{k}" for t, k in res['cum_counts'].items()),
    ]
    if "expansion_ratio" in res:
        lines.append(f"  input n_eff = {res['in_n_eff']:.3f}  ->  EXPANSION ratio out/in = "
                     f"{res['expansion_ratio']:.2f}  (>1 = nonlinear feature lift)")
    return "\n".join(lines)
