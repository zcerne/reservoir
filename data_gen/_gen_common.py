"""Shared helpers for the characterization data generators (index/assemble pattern).

Each generator enumerates a deterministic list of input vectors (from a seed) and
supports three modes:
  --index K   : run ONE forward (input K) → write <out>.parts/part_K.npz immediately
                (this IS the incremental save; a killed array loses only unfinished tasks).
  --assemble  : gather all parts → the final <out>.npz with the analysis's keys.
  --serial    : loop all indices in one process, part-saving each (incremental), then
                assemble. Fallback when not array-parallelizing.
  --count     : print the number of work items (for `sbatch --array=0-(N-1)`).

Array-parallel: `slurm_char_array.sh` runs `--index $SLURM_ARRAY_TASK_ID` per task,
then a final `--assemble`. Wall-clock ≈ one forward run, not N of them.
"""
from __future__ import annotations
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def open_reservoir(path, components):
    """Load the fixed reservoir; return (forward_fn, n_strips, is_master).

    forward(E): real/complex input amplitudes (n_strips,) → stacked complex sensor
    field over `components` (Ey[,Ex,Ez]). NOTE the source casts amplitude to real,
    so pass REAL amplitudes unless a tone's imaginary part is intended as a phase.
    """
    from class_simulation_T import SimulationT
    try:
        import meep as mp
        is_master = bool(mp.am_master())
    except Exception:
        is_master = True
    sim = SimulationT(path)                                   # design DIR, not the json
    sim._set_data()
    src_key = sim._source_key(sim.args)
    amp0 = sim.args[src_key].get("amplitude", [1.0])
    n_strips = len(amp0) if isinstance(amp0, (list, tuple)) else 1

    def forward(E):
        Ey, Ex, Ez = sim._run_basis(list(E))
        f = {"Ey": Ey, "Ex": Ex, "Ez": Ez}
        return np.concatenate([np.asarray(f[c]).ravel() for c in components])

    return forward, n_strips, is_master


def _parts_dir(out_path):
    return out_path + ".parts"


def save_part(out_path, k, is_master, **arrays):
    """Write one part file (master rank only, MPI-safe)."""
    if not is_master:
        return
    d = _parts_dir(out_path)
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f"part_{int(k):06d}.npz"), idx=int(k), **arrays)


def load_parts(out_path):
    """Return parts as a list of dicts sorted by idx. Errors if any are missing/gapped."""
    d = _parts_dir(out_path)
    files = sorted(glob.glob(os.path.join(d, "part_*.npz")))
    if not files:
        raise SystemExit(f"no parts in {d} — run --index tasks (or --serial) first")
    parts = [dict(np.load(f, allow_pickle=True)) for f in files]
    parts.sort(key=lambda p: int(p["idx"]))
    idxs = [int(p["idx"]) for p in parts]
    if idxs != list(range(len(idxs))):
        missing = sorted(set(range(max(idxs) + 1)) - set(idxs))
        raise SystemExit(f"parts incomplete in {d}: {len(idxs)} present, missing idx {missing[:10]}...")
    return parts


def run_mode(args, n_items, run_one, assemble, is_master):
    """Dispatch --count / --index / --serial / --assemble. `run_one(k)` executes one
    forward + save_part; `assemble()` builds the final npz. Returns an exit code."""
    if getattr(args, "count", False):
        print(n_items)                                        # for sbatch --array
        return 0
    if getattr(args, "index", None) is not None:
        k = int(args.index)
        if not (0 <= k < n_items):
            raise SystemExit(f"--index {k} out of range [0,{n_items})")
        run_one(k)
        return 0
    if getattr(args, "assemble", False):
        if is_master:
            assemble()
        return 0
    # --serial (default): loop all, part-save each (incremental), then assemble
    for k in range(n_items):
        run_one(k)
        if is_master:
            print(f"[gen] serial {k+1}/{n_items}", flush=True)
    if is_master:
        assemble()
    return 0


def add_common_args(ap):
    ap.add_argument("--index", type=int, default=None, help="run one work item K → part file")
    ap.add_argument("--assemble", action="store_true", help="combine parts → final npz")
    ap.add_argument("--count", action="store_true", help="print #work items (for sbatch --array)")
    ap.add_argument("--components", default="Ey", help="sensor components to save (Ey[,Ex,Ez])")
    ap.add_argument("--seed", type=int, default=0)
