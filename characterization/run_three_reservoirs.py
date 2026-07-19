"""Characterize three reservoirs with Method A (BLA+SVD) and Method C (covariance
PCA): an identity reservoir, the single-pass 2D LC reservoir, and the MNIST one.

For each we probe with random complex inputs, form (a) the FIELD output E_out = G·E_in
(linear) and (b) the INTENSITY output |E_out|² (the actual reservoir readout,
nonlinear), and report capacity (n_eff) + nonlinearity (residual / expansion).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m1_best_linear_approx import best_linear_approx
from c_covariance_PCA import covariance_pca

RES = os.path.expanduser("~/Orion/resevoir/data")
rng = np.random.default_rng(0)


def operators():
    ops = {}
    # identity reservoir: N->N pass-through (the trivial max-capacity benchmark)
    N = 100
    ops["identity (100)"] = np.eye(N, dtype=complex)
    # single-pass 2D LC reservoir: field transfer T_Ey (n_y_out, n_in=4)
    for name, sub in [("single-pass 2D", "test2D"), ("MNIST", "source_mnist")]:
        p = os.path.join(RES, sub, "simulation_T", "T_matrix.npz")
        if os.path.exists(p):
            ops[f"{name} ({sub})"] = np.load(p)["T_Ey"].astype(complex)
    return ops


def characterize(name, G, n_samples):
    n_in = G.shape[1]
    Ein = rng.normal(size=(n_samples, n_in)) + 1j * rng.normal(size=(n_samples, n_in))
    Efield = (G @ Ein.T).T                      # complex field output (linear)
    Iout = np.abs(Efield) ** 2                  # intensity output (nonlinear readout)
    print(f"\n{'='*72}\n{name}   G:{G.shape}  probes:{n_samples}\n{'='*72}")

    a_f = best_linear_approx({"inputs": Ein, "outputs": Efield}, test_frac=0.3)
    a_i = best_linear_approx({"inputs": Ein, "outputs": Iout.astype(complex)}, test_frac=0.3)
    c_f = covariance_pca({"inputs": Ein, "outputs": Efield})
    c_i = covariance_pca({"inputs": Ein, "outputs": Iout.astype(complex)})
    print(f"[A field]      n_eff={a_f['n_eff']:6.2f}  significant={a_f['n_significant']:4d}  "
          f"rank={a_f['rank']:4d}  cond={a_f['cond']:.2g}  nonlin(1-R2)={a_f['residual_fraction']:.4f}")
    print(f"[A intensity]  n_eff={a_i['n_eff']:6.2f}  significant={a_i['n_significant']:4d}  "
          f"nonlin(1-R2)={a_i['residual_fraction']:.4f}   <- |E|^2 readout nonlinearity")
    print(f"[C field]      out n_eff={c_f['n_eff']:6.2f}  in n_eff={c_f['in_n_eff']:6.2f}  "
          f"expansion={c_f['expansion_ratio']:.2f}")
    print(f"[C intensity]  out n_eff={c_i['n_eff']:6.2f}  expansion={c_i['expansion_ratio']:.2f}"
          f"   <- nonlinear mode lift")
    return dict(name=name, a_f=a_f, a_i=a_i, c_f=c_f, c_i=c_i, in_neff=c_f['in_n_eff'])


def _spec(ax, vals, label, color):
    v = np.asarray(vals); v = v / (v[0] + 1e-30)
    ax.semilogy(np.arange(1, min(len(v), 220) + 1), v[:220] + 1e-12, lw=2, color=color, label=label)


def main():
    ops = operators()
    res = [characterize(name, G, n_samples=max(60, 3 * G.shape[1] + 20)) for name, G in ops.items()]
    cols = ["C0", "C1", "C2", "C3"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    for r, c in zip(res, cols):
        _spec(ax[0, 0], r["a_f"]["power"], f"{r['name']}  n_eff={r['a_f']['n_eff']:.1f}", c)
        _spec(ax[0, 1], r["a_i"]["power"], f"{r['name']}  n_eff={r['a_i']['n_eff']:.1f}, "
              f"1-R²={r['a_i']['residual_fraction']:.2f}", c)
        _spec(ax[1, 0], r["c_f"]["explained_var_ratio"], f"{r['name']}  out/in={r['c_f']['expansion_ratio']:.2f}", c)
        _spec(ax[1, 1], r["c_i"]["explained_var_ratio"], f"{r['name']}  out/in={r['c_i']['expansion_ratio']:.2f}", c)
    titles = [("A — SVD spectrum, FIELD map (linear)", "channel power |s|²/|s₁|²"),
              ("A — SVD spectrum, INTENSITY |E|² (readout)", "channel power"),
              ("C — covariance-PCA spectrum, FIELD", "explained variance ratio"),
              ("C — covariance-PCA spectrum, INTENSITY |E|²", "explained variance ratio")]
    for a, (t, yl) in zip(ax.ravel(), titles):
        a.set_title(t, fontsize=10); a.set_xlabel("channel index"); a.set_ylabel(yl)
        a.legend(fontsize=7); a.grid(alpha=.3, which="both"); a.set_ylim(1e-6, 2)
    plt.suptitle("Reservoir characterization: identity vs single-pass 2D vs MNIST  (A=BLA+SVD, C=cov-PCA)", y=1.01)
    out = os.path.expanduser("~/Orion/resevoir/data/reservoir_characterization.png")
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
