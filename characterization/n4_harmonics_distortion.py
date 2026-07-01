import numpy as np


def _classify_bin(nu, tones, max_order):
    """Decompose an integer frequency ν = Σ aₖ·toneₖ with small integer coeffs.

    Returns (order, kind) where order = Σ|aₖ| and kind ∈ {"dc","fundamental",
    "harmonic","intermod","other"}. Picks the lowest-order decomposition found.
    """
    if nu == 0:
        return 0, "dc"
    best = None
    R = max_order
    if len(tones) == 1:
        f0 = tones[0]
        if f0 and nu % f0 == 0:
            m = abs(nu // f0)
            if m <= R:
                best = (m, [nu // f0])
    else:
        f1, f2 = tones[0], tones[1]
        for a in range(-R, R + 1):
            for b in range(-R, R + 1):
                if a * f1 + b * f2 == nu and (abs(a) + abs(b)) <= R:
                    cand = (abs(a) + abs(b), [a, b])
                    if best is None or cand[0] < best[0]:
                        best = cand
    if best is None:
        return None, "other"
    order, coeffs = best
    if order == 1:
        return 1, "fundamental"
    nz = [c for c in coeffs if c != 0]
    kind = "harmonic" if len(nz) == 1 else "intermod"
    return order, kind


def harmonic_specter(harmonic_data, max_order=6, rel_thresh=1e-9):
    """Nonlinearity Method D — harmonic / intermodulation distortion.

    The reservoir is driven with one or two tones via a PHASE-SWEEP parameter t:
        E_in(t) = Σₖ Aₖ · e^{i·toneₖ·t} · uₖ ,  t = 2π·j/N_t ,  j = 0..N_t−1
    and forward-run at each t (see data_gen/generate_harmonics_data.py). We DFT the
    output over t and inspect the spectrum: a LINEAR field map reproduces only the
    fundamental tones; any nonlinearity (the |E|² readout) creates power at DC,
    harmonics (m·toneₖ) and intermodulation products (a·tone₁+b·tone₂). Their power
    relative to the fundamental measures the nonlinearity, and WHICH orders appear
    give its order (|E|² → order 2).

    harmonic_data : dict with
        outputs : (N_t, n_out) [complex] — reservoir output sampled over the sweep
                  (fields → linearity check ~0 distortion; |E|² → the readout order).
        tones   : (1 or 2,) int  — integer tone frequencies driven (well-separated,
                  e.g. [5,7], so decompositions are unique).
        inputs  : (optional) (N_t, n_in) — for provenance.
    max_order : largest harmonic/intermod order to attribute bins to.
    rel_thresh: power (relative to total) below which a bin is treated as numerical 0.

    Returns dict: power_by_kind {dc,fundamental,harmonic,intermod,other},
        power_by_order {order: power}, thd (√(harmonic/fundamental)),
        imd (intermod/fundamental), distortion_ratio ((total−dc−fund)/fund),
        max_order (highest order with significant power), n_t, tones, linear.
    """
    Y = np.asarray(harmonic_data["outputs"])
    tones = [int(t) for t in np.asarray(harmonic_data["tones"]).reshape(-1)]
    N_t = Y.shape[0]
    Yf = Y.reshape(N_t, -1)

    # DFT over the sweep axis; power per integer frequency = Σ over output features
    F = np.fft.fft(Yf, axis=0) / N_t                       # (N_t, f_out) complex
    P = np.sum(np.abs(F) ** 2, axis=1)                     # (N_t,) power per freq bin
    total = float(P.sum()) + 1e-30
    freqs = np.fft.fftfreq(N_t, d=1.0 / N_t).round().astype(int)   # integer bin freqs

    by_kind = {"dc": 0.0, "fundamental": 0.0, "harmonic": 0.0, "intermod": 0.0, "other": 0.0}
    by_order = {}
    max_ord = 0
    for nu, p in zip(freqs, P):
        if p < rel_thresh * total:
            continue
        order, kind = _classify_bin(int(nu), tones, max_order)
        by_kind[kind] += float(p)
        if order is not None:
            by_order[order] = by_order.get(order, 0.0) + float(p)
            if order >= 2:
                max_ord = max(max_ord, order)

    fund = by_kind["fundamental"]
    nonlin_power = by_kind["harmonic"] + by_kind["intermod"] + by_kind["other"]
    # classic THD/IMD are relative to the fundamental — only meaningful when a
    # fundamental survives (weakly-nonlinear regime). The |E|² readout ANNIHILATES
    # the linear term (fund≈0), so these blow up; use distortion_frac there instead.
    thd = float(np.sqrt(by_kind["harmonic"] / fund)) if fund > rel_thresh * total else float("inf")
    imd = float(by_kind["intermod"] / fund) if fund > rel_thresh * total else float("inf")
    # robust, fundamental-independent: fraction of the AC (non-DC) power that is
    # nonlinear (order ≥ 2). 0 = linear, 1 = purely nonlinear. Well-defined even when
    # the fundamental vanishes.
    ac = total - by_kind["dc"]
    distortion_frac = float(nonlin_power / (ac + 1e-30))
    return dict(
        power_by_kind=by_kind, power_by_order=by_order,
        thd=thd, imd=imd, distortion_frac=distortion_frac,
        max_order=int(max_ord), n_t=int(N_t), tones=tones,
        linear=bool(nonlin_power < rel_thresh * total),
    )


def report(res):
    """One-screen summary of a harmonic_specter result."""
    k = res["power_by_kind"]
    orders = ", ".join(f"{o}:{p:.2e}" for o, p in sorted(res["power_by_order"].items()))
    return "\n".join([
        f"Harmonic/intermod distortion (Method D) | tones={res['tones']}, N_t={res['n_t']}",
        f"  power: dc={k['dc']:.2e} fund={k['fundamental']:.2e} "
        f"harm={k['harmonic']:.2e} intermod={k['intermod']:.2e} other={k['other']:.2e}",
        f"  THD={res['thd']:.3e}  IMD={res['imd']:.3e}  "
        f"distortion_frac(AC)={res['distortion_frac']:.3e}",
        f"  power by order: {{{orders}}}   max nonlinear order = {res['max_order']}",
        f"  ->  {'LINEAR' if res['linear'] else 'NONLINEAR (order %d)' % res['max_order']}",
    ])
