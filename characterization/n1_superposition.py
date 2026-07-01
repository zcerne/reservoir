import numpy as np


def super_position_test(data):
    """Nonlinearity Method A — superposition test (the definition of linearity).

    A map f is linear iff f(αE₁+βE₂) = α·f(E₁) + β·f(E₂) for all inputs and scalars.
    The most direct black-box linearity check — no model. Run from a PRE-GENERATED
    dataset: for each trial, forward-run the reservoir on the two base inputs AND on
    the combined input αE₁+βE₂ (as ONE input), then compare the MEASURED combined
    output against the COMPUTED linear prediction α·f(E₁)+β·f(E₂).

    data : dict of arrays, T = #trials
        alpha, beta  : (T,) complex          — the mixing scalars
        out1, out2   : (T, n_out) [complex]  — reservoir outputs f(E₁), f(E₂)
        out_combo    : (T, n_out) [complex]  — reservoir output f(αE₁+βE₂)
        E1, E2       : (T, n_in)  (optional) — the base inputs, kept for provenance
      Outputs may be fields OR the |E|² readout — the test is agnostic; a nonlinear
      readout simply shows a large violation. All three forward runs must be saved:
      the α·f(E₁)+β·f(E₂) prediction is built from out1/out2, so E₁ and E₂ each need
      their own run — not just the combined one.

    Per trial:  violation = ‖out_combo − (α·out1 + β·out2)‖ / ‖out_combo‖.
    ≈0 → linear; O(1) → strongly nonlinear.

    Returns dict: violation (mean rel-residual), violation_std, violation_max,
        per_trial, r2 (1 − mean(violation²)), n_trials, linear (violation < 1e-6).
    """
    a = np.asarray(data["alpha"]).reshape(-1)
    b = np.asarray(data["beta"]).reshape(-1)
    out1 = np.asarray(data["out1"]); out2 = np.asarray(data["out2"])
    combo = np.asarray(data["out_combo"])
    T = combo.shape[0]
    if not (out1.shape[0] == out2.shape[0] == T == a.shape[0] == b.shape[0]):
        raise ValueError("out1/out2/out_combo/alpha/beta must share trial count T")

    o1 = out1.reshape(T, -1); o2 = out2.reshape(T, -1); oc = combo.reshape(T, -1)
    rhs = a[:, None] * o1 + b[:, None] * o2               # linear prediction
    num = np.linalg.norm(oc - rhs, axis=1)
    den = np.linalg.norm(oc, axis=1) + 1e-30
    viol = num / den

    mean_v = float(viol.mean())
    return dict(
        violation=mean_v, violation_std=float(viol.std()), violation_max=float(viol.max()),
        per_trial=viol, r2=float(1.0 - np.mean(viol ** 2)), n_trials=int(T),
        linear=bool(mean_v < 1e-6),
    )


def report(res):
    """One-screen summary of a superposition_test result."""
    return "\n".join([
        f"Superposition test (Method A) | {res['n_trials']} trials",
        f"  mean rel. violation = {res['violation']:.3e}  (±{res['violation_std']:.1e}, "
        f"max {res['violation_max']:.2e})",
        f"  linearity R² = {res['r2']:.4f}   ->  {'LINEAR' if res['linear'] else 'NONLINEAR'} "
        f"(threshold 1e-6)",
    ])
