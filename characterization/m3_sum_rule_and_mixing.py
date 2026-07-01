import numpy as np


def sum_rule(G):
    """Method D — sum rule / Frobenius total coupling strength (no SVD needed).

    S = Σ|Gᵢⱼ|² = Σ|sⱼ|² = ‖G‖²_F. This is the total power-coupling of the operator:
    an upper bound on the channel count (each channel contributes ≤ its s² to S) and
    a quick throughput check against the diffraction/geometry ceiling — computable
    directly from G without an SVD. A linear/field quantity: compute on the field G
    (before any |E|² readout).

    G : complex array (f_out, f_in) — the coupling operator (Miller's G / T-matrix /
        BLA operator from Method A `best_linear_approx(...)['G']`).

    Returns dict:
        sum_rule   : S = Σ|Gᵢⱼ|²  (== Σ|sⱼ|²)
        frobenius  : ‖G‖_F = √S
        f_in, f_out, shape.
    """
    G = np.asarray(G)
    S = float(np.sum(np.abs(G) ** 2))
    return dict(sum_rule=S, frobenius=float(np.sqrt(S)),
                f_out=int(G.shape[0]), f_in=int(G.shape[1]), shape=tuple(G.shape))


def mixing(G, s=None, Vh=None, hi_frac=0.5):
    """Method E — mixing diagnostic (necessary companion to the capacity metrics).

    Capacity numbers (n_eff, sum rule) are meaningless without confirming the
    operator actually SCRAMBLES its inputs rather than passing them straight through:
    the identity has full mode count and zero mixing. Two independent probes:

    1. Off-diagonal energy fraction — for a (near-)square G, the fraction of ‖G‖²_F
       NOT on the diagonal. ~0 = pass-through (identity-like), →1 = strongly mixing.
       (Only meaningful when input and output bases are comparable / co-indexed.)
    2. Singular-vector delocalization — how spread each input singular vector Vh is
       across the input basis: participation ratio 1/Σ|v_i|⁴ (normalized by length,
       ∈[1/N,1]). A localized (pass-through) mode ≈ 1 element (low); a mixing mode
       spreads over many (high). Reported as the mean over the `hi_frac` strongest
       channels (the ones that carry the signal), plus the inverse-participation
       ratio (IPR) as the localization companion.

    G  : complex (f_out, f_in) coupling operator.
    s  : (optional) singular values from Method A; recomputed if None.
    Vh : (optional) right singular vectors (k, f_in) from Method A; recomputed if None.
    hi_frac : fraction of top channels (by s) to average the delocalization over.

    Returns dict:
        offdiag_frac     : off-diagonal energy fraction (None if G not ~square)
        delocalization   : mean normalized participation ratio of top input modes ∈[0,1]
        ipr              : mean inverse-participation ratio (1/Σ|v|⁴) of top modes (#basis elts spanned)
        per_mode_deloc   : per-mode normalized participation ratio (top channels)
        n_hi             : number of top channels averaged.
    """
    G = np.asarray(G)
    f_out, f_in = G.shape

    # 1. off-diagonal energy fraction (only if bases are co-indexed / ~square)
    offdiag_frac = None
    if f_out == f_in:
        total = float(np.sum(np.abs(G) ** 2)) + 1e-30
        diag = float(np.sum(np.abs(np.diag(G)) ** 2))
        offdiag_frac = float(1.0 - diag / total)

    # 2. singular-vector delocalization (participation ratio of input modes)
    if s is None or Vh is None:
        _, s, Vh = np.linalg.svd(G, full_matrices=False)
    k = len(s)
    n_hi = max(1, int(round(hi_frac * k)))
    order = np.argsort(s)[::-1][:n_hi]                     # strongest channels
    N = Vh.shape[1]
    per_mode, iprs = [], []
    for j in order:
        v = Vh[j]
        p = np.abs(v) ** 2
        p = p / (p.sum() + 1e-30)                          # normalize to a distribution
        ipr = 1.0 / (np.sum(p ** 2) + 1e-30)               # # basis elements spanned ∈[1,N]
        iprs.append(float(ipr))
        per_mode.append(float((ipr - 1.0) / (N - 1.0)))    # normalize to [0,1]
    return dict(
        offdiag_frac=offdiag_frac,
        delocalization=float(np.mean(per_mode)),
        ipr=float(np.mean(iprs)),
        per_mode_deloc=np.asarray(per_mode),
        n_hi=int(n_hi),
    )


def report(res_sum=None, res_mix=None):
    """One-screen summary for the sum-rule (D) and mixing (E) diagnostics."""
    lines = []
    if res_sum is not None:
        lines += [
            "Sum rule (D) | coupling strength",
            f"  S = Σ|Gᵢⱼ|² = Σ|sⱼ|² = {res_sum['sum_rule']:.4g}   (‖G‖_F = {res_sum['frobenius']:.4g})",
            f"  operator shape = {res_sum['shape']}",
        ]
    if res_mix is not None:
        od = res_mix["offdiag_frac"]
        lines += [
            "Mixing (E) | does it scramble?",
            f"  off-diagonal energy fraction = {'%.3f' % od if od is not None else 'n/a (non-square)'}"
            "   (~0 = pass-through, →1 = mixing)",
            f"  mode delocalization = {res_mix['delocalization']:.3f} ∈[0,1]   "
            f"(IPR = {res_mix['ipr']:.1f} basis elts, top {res_mix['n_hi']} channels)",
        ]
    return "\n".join(lines)
