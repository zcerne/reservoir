"""Train two NNs (one on raw features, one on T-matrix-transformed features)
on either iris or MNIST. Dataset is selected via the "dataset" key in each
model_data.json ("iris" or "mnist"). MNIST is loaded full and block-mean
downsampled to 14×14 = 196 features so it matches the source_mnist reservoir's
14×14 source pixel grid.
"""
import argparse
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

from class_neural_network import DenseNN, _load_dataset_normalized


def _load_config(path: str) -> dict:
    with open(os.path.join(path, "model_data.json")) as f:
        return json.load(f)


def _detector_shape_from_source(source_path: str):
    """Read monitor_2 size + simulation resolution from
    <source_path>/simulation_data.json and return the detector flatten shape
    (n_y, n_z). Used only to make the structural sampler aware of the spatial
    grid; if anything is missing or the geometry is 2D, returns None and the
    caller falls back to 1D layout.
    """
    cfg_path = os.path.join(source_path, "simulation_data.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        sd = json.load(open(cfg_path))
        res = float(sd["resolution"])
        dim = int(sd.get("dimention", 3))
        mon = next((o for o in sd.get("objects", sd.get("objects_args", []))
                    if o.get("class") == "sensor" and o.get("name") == "monitor_2"), None)
        if mon is None:
            return None
        size = mon["size"]
        if dim == 2:
            n_y = int(round(size[1] * res)) + 2
            return (n_y, 1)
        # 3D: monitor is yz-plane; z spans full cell, y is monitor size
        cell_z = float(sd["cell_size_z"])
        n_y = int(round(size[1] * res)) + 2
        n_z = int(round(cell_z * res))
        return (n_y, n_z)
    except Exception:
        return None


def _pick_detector_pixels(
    n_pixels: int,
    N_y_total: int,
    mode: str,
    seed: int = 42,
    source_path: str | None = None,
    grid_override=None,
):
    """Choose `n_pixels` flat indices into the (n_y, n_z) detector.

    Modes:
      * ``"random"``    — `np.random.default_rng(seed).choice(N_y_total,
        n_pixels, replace=False)`. Reproducible.
      * ``"structural"`` — regular `n_y_pts × n_z_pts` spatial grid on the
        detector. Defaults to a near-square factoring of `n_pixels`
        (e.g. 196 → 14×14); override via ``"pixel_grid": [n_y_pts, n_z_pts]``.
        Falls back to evenly-spaced indices along the flat axis when the
        detector is 1-D (2D simulations, n_z = 1).
      * ``"linspace"``  — legacy `np.linspace(0, N_y_total-1, n_pixels)`
        along the flat index. The original behaviour; kept for backward
        compatibility when ``pixel_sampling`` is not in the config.

    Returns a 1-D `np.ndarray[int]` of length `n_pixels`.
    """
    mode = mode.lower()
    if mode == "linspace":
        return np.linspace(0, N_y_total - 1, n_pixels, dtype=int)
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        return rng.choice(N_y_total, size=int(n_pixels), replace=False)
    if mode == "structural":
        det_shape = (tuple(grid_override) if grid_override is not None
                     else _detector_shape_from_source(source_path))
        if det_shape is None or det_shape[1] <= 1:
            return np.linspace(0, N_y_total - 1, n_pixels, dtype=int)
        n_y_det, n_z_det = det_shape
        # Factor n_pixels into n_y_pts × n_z_pts ≈ aspect of detector
        n_y_pts = max(1, int(round(np.sqrt(n_pixels * n_y_det / n_z_det))))
        while n_y_pts > 1 and n_pixels % n_y_pts != 0:
            n_y_pts -= 1
        n_z_pts = n_pixels // n_y_pts
        iy = np.linspace(0, n_y_det - 1, n_y_pts, dtype=int)
        iz = np.linspace(0, n_z_det - 1, n_z_pts, dtype=int)
        idx_px = (iy[:, None] * n_z_det + iz[None, :]).ravel()
        if idx_px.size != n_pixels:
            # Pad/trim — exact factoring may be off by one for awkward n
            idx_px = np.linspace(0, N_y_total - 1, n_pixels, dtype=int)
        return idx_px
    raise ValueError(f"unknown pixel_sampling mode '{mode}' "
                     "(use 'random', 'structural', or 'linspace')")


def _apply_T_pixelwise(T_matrices, X, idx_px):
    """Project a batch of source vectors through (subsampled) T matrices.

    Avoids materialising the full ``(N_samples, N_y_total)`` field; the heavy
    matmul shrinks to ``(N_samples, n_pixels)`` per polarisation.

    Args:
        T_matrices: tuple ``(T_Ey, T_Ex, T_Ez)``, each shape (N_y_total, N_strips).
        X:          source vectors, shape (N_samples, N_strips), real.
        idx_px:     flat detector indices to keep, shape (n_pixels,).

    Returns:
        ``T_X`` complex ndarray shape (N_samples, n_pixels, 3) — order (Ey, Ex, Ez).
    """
    out = np.empty((X.shape[0], idx_px.size, 3),
                   dtype=np.complex64 if T_matrices[0].dtype.itemsize <= 8
                                       else np.complex128)
    for i, Tm in enumerate(T_matrices):
        T_sub = Tm[idx_px]                          # (n_pixels, N_strips)
        out[..., i] = (T_sub @ X.T).T               # (N_samples, n_pixels)
    return out


def _build_and_train(cfg: dict, X_tr: np.ndarray, X_te: np.ndarray,
                     y_tr: np.ndarray, y_te: np.ndarray,
                     layer_sizes: list) -> DenseNN:
    model = DenseNN(
        layer_sizes=layer_sizes,
        activation=cfg.get("activation", "relu"),
        dropout=cfg.get("dropout", 0.0),
        batch_norm=cfg.get("batch_norm", False),
        device=cfg.get("device"),   # None → auto-pick cuda if available, else cpu
    )
    print(f"[device] {model.device}")
    print(model)
    model.fit(
        X_tr, y_tr,
        epochs=cfg.get("epochs", 100),
        lr=cfg.get("lr", 1e-3),
        batch_size=cfg.get("batch_size", 32),
        weight_decay=cfg.get("weight_decay", 1e-4),
        val_split=cfg.get("val_split", 0.1),
    )
    print(f"test accuracy: {model.score(X_te, y_te):.3f}")
    return model


def create_T_matrix_dataset(folder_path: str, dataset_name: str = "iris") -> str:
    """Apply the reservoir T matrix to every sample of the requested dataset
    and save (raw_X, T_X, y) to <folder_path>/NN_dataset/T_matrix_dataset.npz.

    `dataset_name` selects which classification dataset is run through the
    reservoir ("iris" → 150×4, "mnist" → 70000×196).
    """
    t_path = os.path.join(folder_path, "simulation_T", "T_matrix.npz")
    if not os.path.exists(t_path):
        raise FileNotFoundError(
            f"T_matrix.npz not found: {t_path}\n"
            "Run build_T_matrix() first."
        )

    d = np.load(t_path)
    n_complete = int(d["n_complete"]) if "n_complete" in d.files else d["T_Ey"].shape[1]
    n_total    = int(d["n_total"])    if "n_total"    in d.files else d["T_Ey"].shape[1]
    if n_complete < n_total:
        raise RuntimeError(
            f"T_matrix.npz is incomplete: {n_complete}/{n_total} basis runs done.\n"
            "Rerun: sbatch slurm.sh <path> build-T-LC"
        )

    T_Ey = d["T_Ey"]
    T_Ex = d["T_Ex"]
    T_Ez = d["T_Ez"]

    built_with_cw = bool(d["use_cw"])    if "use_cw"    in d.files else False
    run_until     = float(d["run_until"]) if "run_until" in d.files else 0.0
    if not built_with_cw and run_until > 0:
        T_Ey = T_Ey / run_until
        T_Ex = T_Ex / run_until
        T_Ez = T_Ez / run_until

    X, y = _load_dataset_normalized(dataset_name)
    if X.shape[1] != T_Ey.shape[1]:
        raise ValueError(
            f"Dataset '{dataset_name}' has {X.shape[1]} features, but the "
            f"T matrix expects a source vector of length {T_Ey.shape[1]}. "
            f"For mnist + source_mnist (14×14 = 196 source pixels) these must match."
        )

    E_Ey = (T_Ey @ X.T).T
    E_Ex = (T_Ex @ X.T).T
    E_Ez = (T_Ez @ X.T).T
    T_X = np.stack([E_Ey, E_Ex, E_Ez], axis=-1)

    out_dir = os.path.join(folder_path, "NN_dataset")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, f"T_matrix_dataset_{dataset_name}.npz"),
        X=X, T_X=T_X, y=y, dataset_name=dataset_name,
    )
    print(f"Saved T_matrix_dataset_{dataset_name}.npz  X={X.shape}  "
          f"T_X={T_X.shape}  y={y.shape}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",   default="data/NN_data/1_normal_model",
                        help="model_data.json folder for the raw-feature NN")
    parser.add_argument("--path_T", default="data/NN_data/1_T_model",
                        help="model_data.json folder for the T-matrix NN")
    args = parser.parse_args()

    cfg_n = _load_config(args.path)
    cfg_t = _load_config(args.path_T)

    dataset_n = cfg_n.get("dataset", "iris")
    dataset_t = cfg_t.get("dataset", dataset_n)
    if dataset_n != dataset_t:
        raise ValueError(
            f"Datasets differ: {args.path}={dataset_n}  vs  "
            f"{args.path_T}={dataset_t}. Both configs must use the same dataset.")
    dataset_name = dataset_n
    print(f"=== Dataset: {dataset_name} ===")

    X_full, y = _load_dataset_normalized(dataset_name)
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=cfg_n.get("test_split", 0.2),
        random_state=42, stratify=y,
    )

    # --- raw-feature NN ---
    print(f"\n=== Normal NN ({dataset_name}) ===")
    model_n = _build_and_train(
        cfg_n,
        X_full[idx_tr], X_full[idx_te], y[idx_tr], y[idx_te],
        layer_sizes=cfg_n["layer_sizes"],
    )
    model_n.save(os.path.join(args.path, "model.pt"))
    np.savez(os.path.join(args.path, "data_split.npz"),
             X_tr=X_full[idx_tr], X_te=X_full[idx_te],
             y_tr=y[idx_tr],     y_te=y[idx_te])
    print(f"Saved: {args.path}/model.pt  +  model_history.json  +  data_split.npz")

    # --- T-matrix NN ---
    print(f"\n=== T-matrix NN ({dataset_name}) ===")
    source = cfg_t["T_matrix_source"]

    # Load T matrices directly (skip the precomputed dataset to avoid the
    # ~100 GB intermediate (70000, 101520, 3) array on MNIST).
    t_path = os.path.join(source, "simulation_T", "T_matrix.npz")
    if not os.path.exists(t_path):
        raise FileNotFoundError(f"T_matrix.npz not found: {t_path}")
    d = np.load(t_path)
    n_complete = int(d["n_complete"]) if "n_complete" in d.files else d["T_Ey"].shape[1]
    n_total    = int(d["n_total"])    if "n_total"    in d.files else d["T_Ey"].shape[1]
    if n_complete < n_total:
        raise RuntimeError(f"T_matrix.npz incomplete: {n_complete}/{n_total}")
    T_Ey, T_Ex, T_Ez = d["T_Ey"], d["T_Ex"], d["T_Ez"]
    built_with_cw = bool(d["use_cw"])    if "use_cw"    in d.files else False
    run_until     = float(d["run_until"]) if "run_until" in d.files else 0.0
    if not built_with_cw and run_until > 0:
        T_Ey, T_Ex, T_Ez = T_Ey/run_until, T_Ex/run_until, T_Ez/run_until
    if X_full.shape[1] != T_Ey.shape[1]:
        raise ValueError(f"Dataset features ({X_full.shape[1]}) must equal "
                         f"T-matrix source dim ({T_Ey.shape[1]}). "
                         "Check the dataset / T_matrix_source pairing.")

    N_y_total = T_Ey.shape[0]
    layer_sizes_t = list(cfg_t["layer_sizes"])
    n_pixels = layer_sizes_t[0]
    if n_pixels is None or n_pixels == N_y_total:
        n_pixels = N_y_total
        idx_px = np.arange(N_y_total)
        layer_sizes_t[0] = N_y_total
    else:
        idx_px = _pick_detector_pixels(
            n_pixels=n_pixels, N_y_total=N_y_total,
            mode=cfg_t.get("pixel_sampling", "linspace"),
            seed=cfg_t.get("pixel_sampling_seed", 42),
            source_path=source,
            grid_override=cfg_t.get("pixel_grid"),
        )
    print(f"detector subsampling: mode={cfg_t.get('pixel_sampling', 'linspace')}, "
          f"n_pixels={len(idx_px)}/{N_y_total}")

    # Subsample T rows BEFORE matmul → never materialize (N_samples, N_y_total).
    T_X = _apply_T_pixelwise((T_Ey, T_Ex, T_Ez), X_full, idx_px)   # (N, n_px, 3)
    I   = np.sum(np.abs(T_X) ** 2, axis=-1).astype(np.float32)     # (N, n_px)
    del T_X, T_Ey, T_Ex, T_Ez

    I_min, I_max = I.min(axis=0), I.max(axis=0)
    I_norm = np.where(I_max > I_min, (I - I_min) / (I_max - I_min), 0.0)

    model_t = _build_and_train(
        cfg_t,
        I_norm[idx_tr], I_norm[idx_te], y[idx_tr], y[idx_te],
        layer_sizes=layer_sizes_t,
    )
    model_t.save(os.path.join(args.path_T, "model.pt"))
    np.savez(os.path.join(args.path_T, "data_split.npz"),
             X_tr=I_norm[idx_tr], X_te=I_norm[idx_te],
             y_tr=y[idx_tr],      y_te=y[idx_te])
    print(f"Saved: {args.path_T}/model.pt  +  model_history.json  +  data_split.npz")
