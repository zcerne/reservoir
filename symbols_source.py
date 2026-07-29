"""source_type "symbols" — reservoir-repo extension to SimpleSim.

Piecewise-constant random-amplitude CW drive for memory-curve measurements:
u(n) ~ Uniform(amp_range) held for symbol_length t.u. each, multiplying the
lam carrier. The sequence is reproducible — analysis regenerates it with
np.random.default_rng(seed) and the same n_sym; nothing is written to disk.

Lives here (not in SimpleSim core) because it is a reservoir-computing
protocol, not a general source. Registered via Simulation.SOURCE_TYPES,
the hook SimpleSim provides for exactly this.

JSON (same as any source, plus):
    "source_type": "symbols",
    "symbol_length": 95.0,          # t.u. per symbol (one cavity round trip)
    "amp_range": [0.5, 1.5],        # uniform amplitude range
    "seed": 0                       # sequence seed
"""
from __future__ import annotations

import numpy as np


def symbol_sequence(seed: int, n_sym: int, amp_range=(0.5, 1.5)) -> np.ndarray:
    """The exact input sequence a run with this seed used — call from the
    readout/analysis side to build regression targets."""
    lo, hi = amp_range
    return np.random.default_rng(int(seed)).uniform(float(lo), float(hi), n_sym)


def register():
    from simplesim.simulation import Simulation
    from simplesim.source import Source

    class SymbolsSource(Source):
        def _set_source(self):
            if self.source_type != "symbols":
                return super()._set_source()
            mp = self.mp
            if not hasattr(mp, "CustomSource"):
                raise NotImplementedError(
                    "source_type 'symbols' needs the MEEP backend "
                    "(CustomSource has no gpumeep equivalent yet)")
            T = float(self.args["symbol_length"])
            end_time = float(self.args.get(
                "end_time", self.args.get("_run_until", 100.0)))
            n_sym = int(np.ceil(end_time / T)) + 1
            u = symbol_sequence(self.args.get("seed", 0), n_sym,
                                self.args.get("amp_range", [0.5, 1.5]))
            f0 = 1.0 / self.lam

            def _sym_src(t, _u=u, _T=T, _f0=f0):
                n = min(int(t / _T), len(_u) - 1) if t >= 0 else 0
                return _u[n] * np.exp(-2j * np.pi * _f0 * t)

            self.source = mp.CustomSource(src_func=_sym_src,
                                          end_time=end_time)

    Simulation.SOURCE_TYPES["symbols"] = SymbolsSource
