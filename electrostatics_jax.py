"""electrostatics_jax — SHIM re-exporting the canonical module from LCrelax.

The anisotropic-Poisson JAX solver now lives in the standalone LCrelax project
(`LCrelax/E_field_stuff/electrostatics_jax.py`); this file used to be a copy. To
avoid divergence it just rebinds this module name to LCrelax's, so existing
`import electrostatics_jax as esj` call sites keep working unchanged (full API,
including private helpers).

Rewired 2026-07-19 (Reservoir → import from LCrelax). Public API:
    build_eps_diag_jax, apply_neg_div_eps_grad, graded_axis_widths, axis_centers,
    solve_poisson_jax, gradient_V_jax
"""
import importlib
import os
import sys

# Add the directory CONTAINING the LCrelax package (Orion mount == smaug, shared).
for _p in ("/home/ziga/Orion", "/home/cernez"):
    if os.path.isdir(os.path.join(_p, "LCrelax")) and _p not in sys.path:
        sys.path.append(_p)

# Replace THIS module object with LCrelax's real one so `esj.<anything>` (public
# or private) resolves to the single canonical implementation.
sys.modules[__name__] = importlib.import_module("LCrelax.E_field_stuff.electrostatics_jax")
