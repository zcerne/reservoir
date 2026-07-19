"""lc_geometry.py — re-export from the canonical LCrelax package (single source; no local copy).

Rewired 2026-07-19: this LC module now lives in the standalone LCrelax project
(LCrelax is the superset — it has this module's functions plus more, verified
numerically identical for the shared ones). This thin shim rebinds the bare
top-level name to LCrelax's module so existing `import lc_geometry.py` / `from lc_geometry.py import …`
call sites keep working unchanged.
"""
import sys as _sys
import _lcrelax_locate  # noqa: F401  (resolves the canonical LCrelax package)

from LCrelax.lc_stuff.lc_geometry import *          # noqa: F401,F403
import importlib as _il
_sys.modules[__name__] = _il.import_module("LCrelax.lc_stuff.lc_geometry")
