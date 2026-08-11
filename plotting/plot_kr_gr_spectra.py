"""Kernel- and generalisation-rank singular spectra, one figure per design set.

  python plotting/plot_kr_gr_spectra.py
  python plotting/plot_kr_gr_spectra.py --paths <dirA> <dirB> --labels A B --out fig.png

Reads <path>/datasets/ipc_4src_a<drive>.npz (M distinct inputs -> kernel spectrum)
and <path>/datasets/gr_4src_a<drive>.npz (few bases x many jittered replicas ->
generalisation spectrum), and plots both as normalised singular-value spectra.

The number quoted for each curve is the participation dimension
    D = exp(-sum_i v_i ln v_i),   v_i = s_i^2 / sum_j s_j^2,
i.e. the perplexity of the variance spectrum, not a thresholded numeric rank. A
threshold count is unusable here: the spectra decay smoothly over ~70 non-zero
directions, so the numeric rank is set by whatever tolerance is chosen, whereas D
is threshold-free and reports how many directions actually carry the variance.

Legenstein & Maass 2007 predict computational performance by KR - GR: the kernel
spectrum counts directions the reservoir can reach with *different* inputs, the
generalisation spectrum counts directions it wanders into when the *same* input is
merely jittered.

READ THE DOTTED VERTICAL CAREFULLY. With n_sources real drive amplitudes, the
linear part of the response spans at most n_sources directions, so variance BEYOND
that index is necessarily nonlinear. The converse does NOT hold: a singular
direction is a mode, not a label, and nonlinearity that acts along the linear
directions (gain saturation compressing the response) lands inside the leading
block. The tail is therefore a *lower bound* on nonlinearity, never a measure of
it, and the leading block must not be called "the linear part". Measured against
the true linear subspace (`linear_alignment` below, obtained by regressing the
state on the drive amplitudes), 72% of design 01's nonlinear power and 55% of
design 05's sits inside the leading four directions.

For the same reason, the fraction of a jitter cloud lying in its OWN top four is
near 100% by construction and says nothing. `linear_alignment` is the meaningful
quantity: the fraction of the cloud that lies in the linear response subspace
fitted from the kernel set. Only where that is high is KR - GR real capacity
rather than a noise-fitting margin.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUNS = Path("/home/ziga/Lips_project/reservoir_runs")
DEFAULT_PATHS = [RUNS / "reservoir_types" / "block_iso_gain" / "01", RUNS / "05_cav_4src"]
DEFAULT_LABELS = ["01 block + gain", "05 cavity"]
COLORS = ["#0a6ebd", "#d1701a"]     # one hue per design, fixed order, never cycled
N_SHOW = 20                         # leading components drawn


def features(out):
    """Complex readout -> real feature matrix [Re | Im].

    A complex SVD would count a direction and its conjugate as one; the readout is
    a real linear map on (Re, Im) separately, so the real stacking is what its
    reachable dimension actually is.
    """
    out = np.asarray(out)
    return np.concatenate([out.real, out.imag], axis=1)


def spectrum(X):
    """Variance fractions of the centred rows of X, plus participation dimension."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    v = s ** 2 / (s ** 2).sum()
    eff = float(np.exp(-(v * np.log(v + 1e-300)).sum()))
    return v, eff


def load_member(npz_path, *keys):
    """Read only the named members -- these files run to several GB of field data."""
    with np.load(npz_path, allow_pickle=True, mmap_mode=None) as d:
        return [np.asarray(d[k]) for k in keys]


def linear_subspace(U, X):
    """Orthonormal basis of the linear response subspace, and the linear/nonlinear split.

    Regressing the state on the drive amplitudes (with intercept) splits the centred
    state exactly: because the least-squares residual is orthogonal to the regressors
    by construction, the cross-term vanishes and
        total variance = linear variance + nonlinear variance
    holds to machine precision. `QL` spans the linear image; the residual is free to
    have components both inside and outside it, which is precisely why the singular
    tail undercounts nonlinearity.
    """
    A = np.concatenate([U, np.ones((len(U), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    fit = A @ coef
    lin = fit - fit.mean(0)
    res = X - fit
    res = res - res.mean(0)
    tot = ((X - X.mean(0)) ** 2).sum()
    QL, _ = np.linalg.qr(coef[:U.shape[1]].T)
    return QL, float((res ** 2).sum() / tot)


def kernel_spectrum(path, drive):
    """Spectrum over M distinct inputs: the reservoir's reachable state space."""
    ds = Path(path) / "datasets" / f"ipc_4src_a{drive}.npz"
    outputs, inputs = load_member(ds, "outputs", "inputs")
    X = features(outputs)
    v, eff = spectrum(X)
    QL, nonlinear = linear_subspace(inputs, X)
    return v, eff, outputs.shape[0], QL, nonlinear


def generalisation_spectrum(path, drive, QL):
    """Spectrum of the replica cloud about each base input, averaged over bases.

    Each base is centred on its own cloud mean rather than the global mean, so what
    is measured is the spread caused by the jitter alone and not the spread between
    the (deliberately different) base inputs.

    `QL` is the linear response subspace fitted from the kernel set. The fraction of
    the cloud lying in it is the quantity that decides whether the jitter is
    confined to the directions the drive itself uses — asking instead how much of
    the cloud lies in the cloud's own leading directions answers nothing, since that
    is near 100% for any cloud.
    """
    ds = Path(path) / "datasets" / f"gr_4src_a{drive}.npz"
    outputs, base_id = load_member(ds, "outputs", "base_id")
    X = features(outputs)
    vs, effs, radii, align = [], [], [], []
    for b in np.unique(base_id):
        sel = base_id == b
        v, eff = spectrum(X[sel])
        vs.append(v[:N_SHOW])
        effs.append(eff)
        m = X[sel].mean(0)
        radii.append(np.linalg.norm(X[sel] - m, axis=1).mean() / np.linalg.norm(m))
        D = X[sel] - m
        align.append(((D @ QL) ** 2).sum() / (D ** 2).sum())
    return (np.mean(vs, axis=0), float(np.mean(effs)), float(np.mean(radii)),
            int(np.unique(base_id).size), int(sel.sum()), float(np.mean(align)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", nargs="+", default=[str(p) for p in DEFAULT_PATHS],
                    help="reservoir dirs holding datasets/ (default: block_iso_gain/01, 05_cav_4src)")
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--drive", type=int, default=100, help="drive amplitude tag (default: 100)")
    ap.add_argument("--n-sources", type=int, default=4,
                    help="input strip count — marks the subspace the drive itself spans")
    ap.add_argument("--out", default=str(REPO / "data" / "reservoir_types"
                                         / "kr_gr_spectra_01_vs_05.png"),
                    help="figure path; a stats_<stem>.npz is written beside it, as "
                         "for the other cross-design figures in that directory")
    a = ap.parse_args()

    if len(a.labels) != len(a.paths):
        raise SystemExit(f"{len(a.paths)} paths but {len(a.labels)} labels")

    fig, (ax_k, ax_g) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    k = np.arange(1, N_SHOW + 1)
    summary, stats = [], {}

    for path, label, colour in zip(a.paths, a.labels, COLORS):
        vk, eff_k, M, QL, nonlinear = kernel_spectrum(path, a.drive)
        vg, eff_g, radius, n_base, n_rep, align = generalisation_spectrum(
            path, a.drive, QL)

        ax_k.plot(k, vk[:N_SHOW], "o-", ms=4, lw=1.6, color=colour,
                  label=f"{label}   D = {eff_k:.2f}")
        ax_g.plot(k, vg, "o-", ms=4, lw=1.6, color=colour,
                  label=f"{label}   D = {eff_g:.2f}")
        summary.append((label, M, n_base, n_rep, eff_k, eff_g, radius,
                        nonlinear, align))
        stats[label] = {"kernel_var": vk, "gr_var": vg, "kr_dim": eff_k,
                        "gr_dim": eff_g, "cloud_radius": radius, "n_inputs": M,
                        "n_base": n_base, "n_rep": n_rep, "path": str(path),
                        "nonlinear_power": nonlinear, "linear_alignment": align}

    for ax, title, sub in ((ax_k, "kernel spectrum", f"{summary[0][1]} distinct inputs"),
                           (ax_g, "generalisation spectrum",
                            f"{summary[0][2]} bases x {summary[0][3]} jittered replicas")):
        ax.axvline(a.n_sources, color="0.45", ls=":", lw=1.2)
        ax.set_yscale("log")
        # The index is a count, so only whole numbers may appear on it.
        ax.set_xticks(np.arange(0, N_SHOW + 1, 4)[1:])
        ax.set_xlabel("singular direction")
        ax.set_title(f"{title}\n{sub}", fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8.5, frameon=False)
    ax_k.set_ylabel("fraction of variance")
    ax_g.annotate(f"{a.n_sources} drive\ndirections", xy=(a.n_sources, 1),
                  xytext=(a.n_sources + 0.7, 3e-3), fontsize=8, color="0.35")

    fig.suptitle("Reachable vs noise-driven state directions at drive "
                 f"a={a.drive}", fontsize=11)
    fig.tight_layout()
    out = Path(a.out)
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    np.savez(out.parent / f"stats_{out.stem}.npz", drive=a.drive,
             n_sources=a.n_sources, res=np.asarray(stats, dtype=object))

    print(f"\n{'design':<22}{'KR D':>8}{'GR D':>8}{'KR-GR':>8}"
          f"{'nonlinear':>12}{'jitter in lin.':>16}{'cloud radius':>14}")
    for label, M, nb, nr, ek, eg, rad, nonlin, align in summary:
        print(f"{label:<22}{ek:8.2f}{eg:8.2f}{ek-eg:8.2f}"
              f"{nonlin*100:11.1f}%{align*100:15.1f}%{rad*100:13.2f}%")
    print(f"\nwrote {out}  and  {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    raise SystemExit(main())
