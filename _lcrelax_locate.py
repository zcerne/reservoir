"""Locate the canonical LCrelax package so `import LCrelax` resolves.

Single shared implementation for every shim / runtime import in this repo
(was: a copy-pasted path-probe loop in each file). Resolution order:

1. A checkout found by walking UP from this repo: a directory that actually
   contains ``LCrelax/__init__.py``. Co-located checkouts stay in sync with
   the copy being edited (Orion: /home/ziga/, smaug: /home/cernez/,
   workbox mount: ~/Orion/).
2. The Nextcloud layout, where the package directory is ``LCrelax/gitcode/``:
   aliased to the name ``LCrelax`` via importlib, so local (workbox) runs
   import the canonical Nextcloud copy without needing the Orion mount.
3. Hardcoded fallbacks /home/ziga/Orion and /home/cernez (sshfs / NFS views).

The old probe used ``isdir(<p>/LCrelax)``, which false-positived on the
Nextcloud *project* folder (it contains gitcode/, not the package) and died
with a confusing ``No module named 'LCrelax.lc_stuff'`` when the Orion mount
was absent.
"""
from __future__ import annotations

import importlib.util
import os
import sys


def ensure() -> None:
    """Idempotent: after this returns, `import LCrelax` finds a real package."""
    if "LCrelax" in sys.modules:
        return
    d = os.path.dirname(os.path.abspath(__file__))
    cands = []
    while d not in ("/", ""):
        cands.append(d)
        d = os.path.dirname(d)
    cands += ["/home/ziga/Orion", "/home/cernez"]
    for p in cands:
        pkg = os.path.join(p, "LCrelax")
        if os.path.isfile(os.path.join(pkg, "__init__.py")):     # real package
            if p not in sys.path:
                sys.path.append(p)   # append: never shadow this repo's modules
            return
        git = os.path.join(pkg, "gitcode")                       # Nextcloud layout
        if os.path.isfile(os.path.join(git, "__init__.py")):
            spec = importlib.util.spec_from_file_location(
                "LCrelax", os.path.join(git, "__init__.py"),
                submodule_search_locations=[git])
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            sys.modules["LCrelax"] = mod
            spec.loader.exec_module(mod)
            return
    raise ModuleNotFoundError(
        "LCrelax package not found: no LCrelax/__init__.py (or "
        "LCrelax/gitcode/__init__.py) under any ancestor of "
        f"{os.path.dirname(os.path.abspath(__file__))!r}, /home/ziga/Orion, "
        "or /home/cernez")


ensure()
