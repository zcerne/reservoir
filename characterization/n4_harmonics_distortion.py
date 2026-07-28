import numpy as np


def _build_lookup(tones, max_order):
    """Map every achievable integer frequency ν = Σ aₖ·toneₖ (Σ|aₖ| ≤ max_order) to
    its LOWEST-order integer-coefficient decomposition. Handles any number of tones
    (earlier version hardcoded a 2-tone basis and silently misclassified a 3rd/4th
    driven tone's fundamental as a high-order intermod product of tones[0], tones[1])."""
    from itertools import product
    n = len(tones)
    R = max_order
    lookup = {0: (0, (0,) * n)}
    if n == 0:
        return lookup
    for coeffs in product(range(-R, R + 1), repeat=n):
        order = sum(abs(c) for c in coeffs)
        if order == 0 or order > R:
            continue
        nu = sum(c * f for c, f in zip(coeffs, tones))
        if nu not in lookup or order < lookup[nu][0]:
            lookup[nu] = (order, coeffs)
    return lookup


def _bin_label(coeffs):
    """Human-readable name for a bin from its tone coefficients: (2,0) -> '2f1',
    (-1,1) -> 'f2-f1', (2,2) -> '2f1+2f2'. Empty string for DC/unclassified."""
    # positive terms first so a difference reads "f2-f1", not "-f1+f2"
    terms = sorted(((k, c) for k, c in enumerate(coeffs) if c != 0),
                   key=lambda kc: (kc[1] < 0, kc[0]))
    parts = []
    for k, c in terms:
        mag = "" if abs(c) == 1 else str(abs(c))
        sign = "-" if c < 0 else ("+" if parts else "")
        parts.append(f"{sign}{mag}f{k + 1}")
    return "".join(parts)


def _classify_bin(nu, lookup):
    """Classify an integer frequency bin using a precomputed `_build_lookup` table.

    Returns (order, kind) where order = Σ|aₖ| and kind ∈ {"dc","fundamental",
    "harmonic","intermod","other"}.
    """
    if nu == 0:
        return 0, "dc"
    entry = lookup.get(int(nu))
    if entry is None:
        return None, "other"
    order, coeffs = entry
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
        max_order (highest order with significant power), n_t, tones, linear,
        plus the resolved per-bin spectrum the classification is built from —
        spec_nu / spec_power / spec_kind / spec_order / spec_label over the
        non-negative bins (what the plot draws; carries no raw field data).
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

    lookup = _build_lookup(tones, max_order)
    by_kind = {"dc": 0.0, "fundamental": 0.0, "harmonic": 0.0, "intermod": 0.0, "other": 0.0}
    by_order = {}
    max_ord = 0
    for nu, p in zip(freqs, P):
        if p < rel_thresh * total:
            continue
        order, kind = _classify_bin(int(nu), lookup)
        by_kind[kind] += float(p)
        if order is not None:
            by_order[order] = by_order.get(order, 0.0) + float(p)
            if order >= 2:
                max_ord = max(max_ord, order)

    # per-bin spectrum over the non-negative half, for plotting/inspection
    keep = freqs >= 0
    o_sort = np.argsort(freqs[keep])
    spec_nu = freqs[keep][o_sort]
    spec_power = P[keep][o_sort]
    spec_kind, spec_order, spec_label = [], [], []
    for v in spec_nu:
        o, k = _classify_bin(int(v), lookup)
        spec_kind.append(k)
        spec_order.append(-1 if o is None else int(o))
        ent = lookup.get(int(v))
        spec_label.append(_bin_label(ent[1]) if ent is not None else "")

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
        spec_nu=spec_nu, spec_power=spec_power, spec_kind=spec_kind,
        spec_order=spec_order, spec_label=spec_label,
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
