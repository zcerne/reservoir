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

    Runs through SimpleSim's ReservoirSimulation (class_simulation.py) — the
    current, actively-maintained engine (same one the interactive run()/plot()
    workflow uses) — for BOTH backends. The old standalone class_simulation_gpu.py
    run_basis()/_run_2d_sted() path and the (stale, broken since the SimpleSim
    migration) class_simulation_T.py path are no longer used here: they're a
    separate, unsynced reimplementation that was never validated against
    ReservoirSimulation and gave measurably wrong gain for STED designs
    (2026-07-25: ~2.5x vs SimpleSim's ~4.1x at the same amplitude/pump).
    """
    import json as _json
    with open(os.path.join(path, "simulation_data.json")) as _f:
        cfg = _json.load(_f)
    backend = str(cfg.get("solver", "meep")).lower()
    # Env override: the same design JSON can run MEEP on Orion (CPU/MPI) and
    # GPUmeep on smaug without editing the file.
    backend = os.environ.get("RESERVOIR_SOLVER", backend).lower()
    backend = "gpumeep" if backend in ("gpumeep", "gpu", "gpumma") else "meep"

    from simplesim import Simulation as ReservoirSimulation

    # First source in JSON key order that isn't "source_2" (same convention the
    # old pipeline used) -- for 02_adding_pump this is source_1 (signal), not
    # source_pump (fixed CW pump, comes later in the JSON).
    src_key = next(k for k, v in cfg.items()
                   if isinstance(v, dict) and v.get("class") == "source"
                   and k != "source_2")
    amp0 = cfg[src_key].get("amplitude", [1.0])
    n_strips = len(amp0) if isinstance(amp0, (list, tuple)) else 1

    # Per-process scratch tag (matches the old GPUMEEP_SCRATCH_TAG use): two
    # concurrent characterization batches (e.g. split across smaug1/smaug2)
    # must not share one suffix, or they'd race on the same output npz.
    suffix = os.environ.get("SIMPLESIM_SCRATCH_TAG", "fwd")
    out_dir = os.path.join(path, f"simulation_{backend}")

    from simplesim.simulation import resolve_engine
    _mp = resolve_engine(backend)
    is_master = bool(_mp.am_master()) if hasattr(_mp, "am_master") else True

    def forward(E):
        sim = ReservoirSimulation(path, backend=backend, suffix=suffix,
                                  overrides={src_key: {"amplitude": list(E)}})
        sim.relax()
        sim.run(empty=False)
        m2 = np.load(os.path.join(out_dir, f"monitor_2_{suffix}.npz"))
        zeros = None
        vals = []
        for c in components:
            if c in m2.files:
                v = np.asarray(m2[c])
                zeros = np.zeros_like(v)
            vals.append(m2[c] if c in m2.files else None)
        vals = [v if v is not None else zeros for v in vals]
        return np.concatenate([np.asarray(v).ravel() for v in vals])

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
    if getattr(args, "batch", None) is not None:
        S = int(args.batch_size); lo = int(args.batch) * S; hi = min(lo + S, n_items)
        if lo >= n_items:
            raise SystemExit(f"--batch {args.batch} (size {S}) starts at {lo} ≥ n_items {n_items}")
        for k in range(lo, hi):
            run_one(k)
            if is_master:
                print(f"[gen] batch {args.batch}: {k-lo+1}/{hi-lo} (idx {k}/{n_items})", flush=True)
        return 0
    if getattr(args, "assemble", False):
        if is_master:
            assemble()
        return 0
    # --serial (default): loop all, part-save each (incremental), then assemble.
    # --reverse iterates high→low so a second (backward) worker can share the job
    # with a forward array run and meet in the middle (run_one skips done parts).
    order = range(n_items - 1, -1, -1) if getattr(args, "reverse", False) else range(n_items)
    for i, k in enumerate(order):
        run_one(k)
        if is_master:
            print(f"[gen] serial {i+1}/{n_items} (idx {k})", flush=True)
    if is_master:
        assemble()
    return 0


def add_common_args(ap):
    ap.add_argument("--index", type=int, default=None, help="run one work item K → part file")
    ap.add_argument("--batch", type=int, default=None, help="run item batch B: indices [B*batch_size, +batch_size)")
    ap.add_argument("--batch_size", type=int, default=50, help="items per batch (with --batch)")
    ap.add_argument("--assemble", action="store_true", help="combine parts → final npz")
    ap.add_argument("--count", action="store_true", help="print #work items (for sbatch --array)")
    ap.add_argument("--components", default="Ey", help="sensor components to save (Ey[,Ex,Ez])")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reverse", action="store_true", help="serial: iterate indices high→low")
    ap.add_argument("--skip_existing", action="store_true", help="skip an index whose part file already exists")
