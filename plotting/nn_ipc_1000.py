"""1000-probe Dambre IPC for the random-sigmoid reference NN (ELM-style).

Same layer as characterization/random_sigmoid_reference.py (4 -> 200 sigmoid,
gain 1.5, seed 0), same probe protocol as the physical devices' 1000-probe IPC
sets: u ~ Uniform[-1,1]^4, max_degree 5.
"""
import os, sys
import numpy as np

REPO = "/home/ziga/Nextcloud/Doktorski/Projects/Reservoir/gitcode"
sys.path[:0] = [os.path.join(REPO, "characterization"), os.path.join(REPO, "plotting"), REPO]
from random_sigmoid_reference import make_nn                   # noqa: E402
import n6_dambre as n6                                          # noqa: E402
from plot_function_n6_dambre_ipc import plot_n6_dambre_ipc      # noqa: E402

rng = np.random.default_rng(0)
U = rng.uniform(-1.0, 1.0, size=(1000, 4))
nn = make_nn(n_in=4, n_out=200, gain=1.5, seed=0)
X = nn(U)

res = n6.dambre_ipc({"inputs": U, "outputs": X}, max_degree=5)
byd = {d: round(v, 2) for d, v in sorted(res["ipc_by_degree"].items())}
print(f"total={res['ipc_total']:.3f} NL={res['nonlinear_fraction']:.3f} "
      f"thr={res['threshold']:.3f} f_used={res['f_used']}/{res['f_out']} "
      f"rank bound={res['bound']}\n{byd}")

out = plot_n6_dambre_ipc(res, f"{REPO}/data/reservoir_types", "_nn1000_field")
print("wrote", out)
