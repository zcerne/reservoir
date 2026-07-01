import numpy as np


def _fit_G(X, Y, rcond=None):
    """Best linear map G = Y·pinv(X) for probe sets X (M,f_in), Y (M,f_out)."""
    Xt = X.T
    return Y.T @ (np.linalg.pinv(Xt, rcond=rcond) if rcond is not None else np.linalg.pinv(Xt))


def amplitude_dependance(data, ref="min", norm_by_scale=False, rcond=None):
    """Nonlinearity Method C — amplitude-dependent BLA (Pintelon–Schoukens).

    A linear system's best-linear map G is amplitude-INDEPENDENT. So re-fit G at
    several input drive levels and measure how much it DRIFTS: any drift with drive
    level reveals (and quantifies) nonlinearity — and, unlike the other methods, it
    is itself amplitude-resolved (tells you at what drive the nonlinearity turns on).

    The least-squares fit G_ℓ = Y_ℓ·pinv(X_ℓ) is ALREADY scale-invariant for a linear
    map (Y=G·X ⇒ fit = G at every level → zero drift). For a nonlinear readout it
    drifts: e.g. the |E|² readout gives Y_ℓ = |G·X_ℓ|² ∝ scale², X_ℓ ∝ scale, so the
    fitted G_ℓ ∝ scale — it grows with drive. Comparing the RAW fitted G_ℓ is thus the
    correct probe (`norm_by_scale=False`, default). `norm_by_scale=True` divides G_ℓ by
    its level scale — do NOT use it here, it manufactures drift for the linear case.

    data : dict, grouped probes at L amplitude levels (same input DIRECTIONS, scaled):
        levels   : (L,) drive amplitudes/powers
        one of:
          inputs, outputs : lists/arrays len L, each (M_ℓ, ...) — per-level probe sets
          OR flat: inputs (ΣM, ...), outputs (ΣM, ...), level_id (ΣM,) mapping to levels
    ref : which level is the reference for the drift ("min" | "max" | index int).
    norm_by_scale : divide each G_ℓ by its level scale before comparing (default False;
        see above — leave False).

    Returns dict:
        levels, drift (per-level ‖Ĝ_ℓ−Ĝ_ref‖_F / ‖Ĝ_ref‖_F), max_drift,
        sv_drift (per-level relative singular-value-spectrum drift),
        ref_level, linear (max_drift < 1e-6), n_per_level.
    """
    levels = np.asarray(data["levels"], dtype=float).reshape(-1)
    L = levels.size

    # ---- gather per-level (X, Y) ----
    if "level_id" in data:                               # flat storage
        X = np.asarray(data["inputs"]); Y = np.asarray(data["outputs"])
        lid = np.asarray(data["level_id"]).reshape(-1)
        Xs = [X[lid == k].reshape((lid == k).sum(), -1) for k in range(L)]
        Ys = [Y[lid == k].reshape((lid == k).sum(), -1) for k in range(L)]
    else:                                                # list/array per level
        Xin, Yin = data["inputs"], data["outputs"]
        Xs = [np.asarray(Xin[k]).reshape(np.asarray(Xin[k]).shape[0], -1) for k in range(L)]
        Ys = [np.asarray(Yin[k]).reshape(np.asarray(Yin[k]).shape[0], -1) for k in range(L)]

    n_per_level = [int(x.shape[0]) for x in Xs]
    Gs = [_fit_G(Xs[k], Ys[k], rcond) for k in range(L)]

    ref_idx = int(np.argmin(levels)) if ref == "min" else \
        int(np.argmax(levels)) if ref == "max" else int(ref)

    def norm(k):
        g = Gs[k]
        return g * (levels[ref_idx] / (levels[k] + 1e-30)) if norm_by_scale else g

    Gref = norm(ref_idx)
    denom = np.linalg.norm(Gref) + 1e-30
    svref = np.linalg.svd(Gref, compute_uv=False)
    drift = np.empty(L); sv_drift = np.empty(L)
    for k in range(L):
        Gk = norm(k)
        drift[k] = np.linalg.norm(Gk - Gref) / denom
        sv = np.linalg.svd(Gk, compute_uv=False)
        m = min(len(sv), len(svref))
        sv_drift[k] = float(np.linalg.norm(sv[:m] - svref[:m]) / (np.linalg.norm(svref[:m]) + 1e-30))

    max_drift = float(np.max(drift))
    return dict(
        levels=levels, drift=drift, max_drift=max_drift, sv_drift=sv_drift,
        ref_level=float(levels[ref_idx]), ref_idx=ref_idx, n_per_level=n_per_level,
        norm_by_scale=bool(norm_by_scale), linear=bool(max_drift < 1e-6),
    )


def report(res):
    """One-screen summary of an amplitude_dependance result."""
    rows = "  ".join(f"{lv:g}:{d:.2e}" for lv, d in zip(res["levels"], res["drift"]))
    return "\n".join([
        f"Amplitude-dependent BLA (Method C) | {len(res['levels'])} levels, "
        f"ref={res['ref_level']:g}, scale-normalized={res['norm_by_scale']}",
        f"  G drift vs ref (level:‖ΔG‖/‖G‖):  {rows}",
        f"  max drift = {res['max_drift']:.3e}   ->  "
        f"{'LINEAR (amplitude-independent G)' if res['linear'] else 'NONLINEAR (G drifts with drive)'}",
    ])
