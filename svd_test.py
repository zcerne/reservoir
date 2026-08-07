"""Does SVD reveal nonlinearity? Synthetic check with known ground truth.

Two inputs x, y. The "reservoir" exposes three things: x, y and their product xy.
A linear medium could only ever produce x and y, so:

    rank 2  = linear medium (the floor: one mode per input)
    rank 3  = the product exists -> genuine nonlinearity

Then each mode is identified by regressing its per-probe amplitude against the
inputs: linear fit vs fit allowing products. R2_lin ~ 0 with R2_quad ~ 1 means
that direction IS the nonlinear one.
"""
import numpy as np


def xor(x, y):
    return x*y


# probes on [-1,1], NOT [0,1]: on [0,1] everything is positive so x, y and xy are
# strongly correlated and the modes come out as mixtures. Symmetric range makes
# the product orthogonal to the linear terms, so mode 3 separates cleanly.
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

X, Y = np.meshgrid(x, y)

M = xor(X, Y)

inputs = np.array([X.ravel(), Y.ravel()]).T          # (10000, 2) — the u's
features = np.array([X.ravel(), Y.ravel(), M.ravel()]).T  # what the medium computes

# Handing the SVD [x, y, xy] as three tidy columns would be cheating: the answer
# is then the columns we inserted. A real sensor sees MIXTURES — every detector
# picks up some combination of everything. So scramble into n_det detectors with
# a random matrix. Rank is invariant under that (invertible mixing), so if SVD
# still says 3 it has genuinely recovered the dimensionality rather than read it
# off the layout.
rng = np.random.default_rng(0)
n_det = 20
dataset = features @ rng.normal(size=(features.shape[1], n_det))

centered = dataset - dataset.mean(0)
U, s, vt = np.linalg.svd(centered, full_matrices=False)

print("singular values      :", np.round(s, 4))
print("normalised           :", np.round(s/s[0], 4))
print("rank (>1e-8)         :", int((s/s[0] > 1e-8).sum()))


def r2(target, A):
    """Fraction of `target`'s variance a least-squares fit on columns A explains."""
    t = target - target.mean()
    w = np.linalg.lstsq(A, t, rcond=None)[0]
    print(w.shape)
    fit = A @ w
    return float(1 - np.var(t - fit) / np.var(t))


# design matrices: what we allow the fit to use
ones = np.ones(len(inputs))
lin = np.c_[ones, inputs]                                    # 1, x, y
quad = np.c_[lin, inputs[:, 0]*inputs[:, 1],                 # + xy, x^2, y^2
             inputs[:, 0]**2, inputs[:, 1]**2]

print("\nmode   s_i/s_1   R2 linear   R2 +quadratic")
rank = int((s/s[0] > 1e-8).sum())
lin_total = 0.0
for i in range(rank):
    amp = U[:, i]                                # this mode's amplitude per probe
    a, b = r2(amp, lin), r2(amp, quad)
    lin_total += a
    tag = "  <- nonlinear direction" if a < 0.5 <= b else ""
    print(f"  {i+1}     {s[i]/s[0]:.4f}     {a:8.3f}   {b:12.3f}{tag}")

# After mixing, an individual mode is an arbitrary rotation inside the span, so a
# single mode need not be purely linear or purely quadratic. What IS invariant is
# how much of the span linear functions can reach: summing R2_linear over the
# modes counts the linear dimensions.
print(f"\nlinear dimensions in the span : {lin_total:.2f}  (= number of inputs if"
      f" the medium were linear)")
print(f"total rank                    : {rank}")
print(f"NONLINEAR dimensions          : {rank - lin_total:.2f}")

# ------------------------------------------------------------------ XOR readout
# The point of rank>2: a linear readout on the state can only separate XOR once
# the xy mode exists.
P = np.array([[-1., 1.], [1., 1.], [-1., -1.], [1., -1.]])
label = -np.sign(P[:, 0]*P[:, 1])
state_lin = np.c_[np.ones(4), P]                       # linear medium: x, y only
state_xy = np.c_[state_lin, P[:, 0]*P[:, 1]]           # + the product

print("\nXOR label, linear readout:")
print(f"  linear medium (x,y)    R2 = {r2(label, state_lin):.3f}")
print(f"  with product (x,y,xy)  R2 = {r2(label, state_xy):.3f}")
