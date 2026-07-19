"""alcs_class.py — re-export from the canonical LCrelax package (single source; no local copy).

Rewired 2026-07-19: this LC module now lives in the standalone LCrelax project
(LCrelax is the superset — it has this module's functions plus more, verified
numerically identical for the shared ones). This thin shim rebinds the bare
top-level name to LCrelax's module so existing `import alcs_class.py` / `from alcs_class.py import …`
call sites keep working unchanged.
"""
import os as _os, sys as _sys
for _p in ("/home/ziga/Orion", "/home/cernez"):        # dir CONTAINING LCrelax pkg
    if _os.path.isdir(_os.path.join(_p, "LCrelax")) and _p not in _sys.path:
        _sys.path.append(_p)
from LCrelax.lc_stuff.alcs_class import *          # noqa: F401,F403
import importlib as _il
_sys.modules[__name__] = _il.import_module("LCrelax.lc_stuff.alcs_class")
