"""Combined figure of all reservoir-characterization results:
 (1) A SVD spectra (field) — capacity per reservoir
 (2) C PCA spectra (intensity) — the nonlinear mode-expansion
 (3) expansion ratio bar chart — {identity, single-pass, MNIST} x {field, |E|^2, sigmoid}
 (4) sigmoid-gain scan on MNIST — expansion vs how hard the nonlinearity is driven
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m1_best_linear_approx import best_linear_approx
from c_covariance_PCA import covariance_pca

RES = os.path.expanduser("~/Orion/resevoir/data"); rng = np.random.default_rng(0)
def sig(x): return 1 / (1 + np.exp(-x))


def load_ops():
    ops = {"identity": np.eye(100, dtype=complex)}
    for nm, sub in [("single-pass 2D", "test2D"), ("MNIST", "source_mnist")]:
        p = os.path.join(RES, sub, "simulation_T", "T_matrix.npz")
        if os.path.exists(p):
            ops[nm] = np.load(p)["T_Ey"].astype(complex)
    return ops


def readouts(G, Ein):
    Ef = (G @ Ein.T).T
    s = np.std(Ef.real) + 1e-30
    return {"field": Ef, "|E|^2": np.abs(Ef) ** 2 + 0j, "sigmoid": sig(Ef.real / s) + 1j * sig(Ef.imag / s)}


def main():
    ops = load_ops(); cols = {"identity": "C0", "single-pass 2D": "C1", "MNIST": "C2"}
    A_field, C_int, expansion = {}, {}, {}
    for nm, G in ops.items():
        N = G.shape[1]; Ein = rng.normal(size=(3 * N + 40, N)) + 1j * rng.normal(size=(3 * N + 40, N))
        ro = readouts(G, Ein)
        A_field[nm] = best_linear_approx({"inputs": Ein, "outputs": ro["field"]})["power"]
        C_int[nm] = covariance_pca({"inputs": Ein, "outputs": ro["|E|^2"]})["explained_var_ratio"]
        expansion[nm] = {k: covariance_pca({"inputs": Ein, "outputs": v})["expansion_ratio"] for k, v in ro.items()}

    # sigmoid gain scan on MNIST
    Gm = ops["MNIST"]; N = Gm.shape[1]; Ein = rng.normal(size=(3 * N + 40, N)) + 1j * rng.normal(size=(3 * N + 40, N))
    Ef = (Gm @ Ein.T).T; sd = np.std(Ef.real) + 1e-30
    gains = [0.3, 1, 3, 8, 20]
    gain_exp = [covariance_pca({"inputs": Ein, "outputs": sig(g * Ef.real / sd) + 1j * sig(g * Ef.imag / sd)})["expansion_ratio"] for g in gains]
    sq_exp = covariance_pca({"inputs": Ein, "outputs": np.abs(Ef) ** 2 + 0j})["expansion_ratio"]

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    for nm in ops:
        v = A_field[nm] / (A_field[nm][0] + 1e-30)
        ax[0, 0].semilogy(np.arange(1, min(len(v), 220) + 1), v[:220] + 1e-12, lw=2, color=cols[nm], label=nm)
        w = C_int[nm]
        ax[0, 1].semilogy(np.arange(1, min(len(w), 360) + 1), w[:360] + 1e-12, lw=2, color=cols[nm],
                          label=f"{nm}  exp={expansion[nm]['|E|^2']:.2f}")
    ax[0, 0].set_title("(1) A — SVD spectrum, FIELD map (capacity)"); ax[0, 0].set_xlabel("channel"); ax[0, 0].set_ylabel("|s|²/|s₁|²"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3, which="both"); ax[0, 0].set_ylim(1e-6, 2)
    ax[0, 1].set_title("(2) C — PCA spectrum, |E|² (nonlinear lift)"); ax[0, 1].set_xlabel("component"); ax[0, 1].set_ylabel("expl. var ratio"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3, which="both"); ax[0, 1].set_ylim(1e-6, 1)

    # (3) expansion bar chart
    reads = ["field", "|E|^2", "sigmoid"]; xpos = np.arange(len(reads)); w = 0.25
    for i, nm in enumerate(ops):
        ax[1, 0].bar(xpos + (i - 1) * w, [expansion[nm][r] for r in reads], w, color=cols[nm], label=nm)
    ax[1, 0].axhline(1.0, color="k", ls="--", lw=1, label="=1 (no lift / linear ceiling)")
    ax[1, 0].set_xticks(xpos); ax[1, 0].set_xticklabels(reads); ax[1, 0].set_ylabel("expansion ratio (out/in n_eff)")
    ax[1, 0].set_title("(3) Mode expansion by reservoir × readout"); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3, axis="y")

    # (4) sigmoid gain scan
    ax[1, 1].plot(gains, gain_exp, "o-", lw=2, color="C3", label="MNIST + sigmoid(gain·E)")
    ax[1, 1].axhline(sq_exp, color="C2", ls="--", lw=1.5, label=f"MNIST + |E|² = {sq_exp:.2f}")
    ax[1, 1].axhline(1.0, color="k", ls=":", lw=1, label="=1 (no lift)")
    ax[1, 1].set_xscale("log"); ax[1, 1].set_xlabel("sigmoid gain (drive into curved region)"); ax[1, 1].set_ylabel("expansion ratio")
    ax[1, 1].set_title("(4) Sigmoid only lifts when driven hard (MNIST)"); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=.3)

    plt.suptitle("Reservoir characterization — all results  (capacity, nonlinear expansion, gain dependence)", y=1.01)
    out = os.path.join(RES, "reservoir_all_results.png"); plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
