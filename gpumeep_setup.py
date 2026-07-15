"""Locate GPUmeep and import the gpumeep API — nothing else.

gpumeep is self-contained (it loads its own engine-support module by
explicit path), so this is just: find the src dir, put it on sys.path,
import. Set GPUMEEP_PATH to override the default checkout locations.

Usage (all GPU class files):  from gpumeep_setup import gm, FS_PER_MEEP
"""
from __future__ import annotations

import os
import sys

FS_PER_MEEP = 3.335640952        # fs per MEEP time unit (a = 1 µm)

_candidates = [
    os.environ.get("GPUMEEP_PATH"),
    os.path.expanduser("~/Nextcloud/Doktorski/Projects/GPUmeep/gitcode/src"),
    os.path.expanduser("~/GPUmeep/gitcode/src"),
    os.path.expanduser("~/GPUmeep/src"),
]
for _p in _candidates:
    if _p and os.path.exists(os.path.join(_p, "gpumeep.py")):
        GM_SRC = _p
        break
else:
    raise ImportError("Could not find GPUmeep src/ (set GPUMEEP_PATH)")
if GM_SRC not in sys.path:
    sys.path.insert(0, GM_SRC)

import gpumeep as gm  # noqa: E402,F401
