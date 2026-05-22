import argparse
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

from class_neural_network import DenseNN, _load_iris_normalized


def _load_config(path: str) -> dict:
    with open(os.path.join(path, "model_data.json")) as f:
        return json.load(f)


def _build_and_train(cfg: dict, X_tr: np.ndarray, X_te: np.ndarray,
                     y_tr: np.ndarray, y_te: np.ndarray,
                     layer_sizes: list) -> DenseNN:
    model = DenseNN(
        layer_sizes=layer_sizes,
        activation=cfg.get("activation", "relu"),
        dropout=cfg.get("dropout", 0.0),
        batch_norm=cfg.get("batch_norm", False),
    )
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


def create_T_matrix_dataset(folder_path: str) -> str:
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

    iris_X, iris_Y = _load_iris_normalized()

    E_Ey = (T_Ey @ iris_X.T).T
    E_Ex = (T_Ex @ iris_X.T).T
    E_Ez = (T_Ez @ iris_X.T).T
    T_iris_X = np.stack([E_Ey, E_Ex, E_Ez], axis=-1)

    out_dir = os.path.join(folder_path, "NN_dataset")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "T_matrix_dataset.npz"),
        iris_X=iris_X,
        T_iris_X=T_iris_X,
        iris_Y=iris_Y,
    )
    print(f"Saved T_matrix_dataset.npz  iris_X={iris_X.shape}  "
          f"T_iris_X={T_iris_X.shape}  iris_Y={iris_Y.shape}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/NN_data/1_normal_model")
    parser.add_argument("--path_T", default="data/NN_data/1_T_model")
    args = parser.parse_args()

    cfg_n = _load_config(args.path)
    cfg_t = _load_config(args.path_T)

    # shared split — same indices for both networks
    X_iris, y = _load_iris_normalized()
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=cfg_n.get("test_split", 0.2),
        random_state=42, stratify=y
    )

    # --- normal NN ---
    print("=== Normal NN ===")
    model_n = _build_and_train(
        cfg_n,
        X_iris[idx_tr], X_iris[idx_te], y[idx_tr], y[idx_te],
        layer_sizes=cfg_n["layer_sizes"],
    )
    model_n.save(os.path.join(args.path, "model.pt"))
    np.savez(os.path.join(args.path, "data_split.npz"),
             X_tr=X_iris[idx_tr], X_te=X_iris[idx_te], y_tr=y[idx_tr], y_te=y[idx_te])
    print(f"Saved: {args.path}/model.pt  +  model_history.json  +  data_split.npz")

    # --- T-matrix NN ---
    print("\n=== T-matrix NN ===")
    source = cfg_t["T_matrix_source"]
    dataset_path = os.path.join(source, "NN_dataset", "T_matrix_dataset.npz")
    if not os.path.exists(dataset_path):
        print("T-matrix dataset not found, building...")
        create_T_matrix_dataset(source)

    data = np.load(dataset_path, allow_pickle=True)
    T_iris_X = data["T_iris_X"]   # (150, N_y, 3) complex
    I = np.sum(np.abs(T_iris_X) ** 2, axis=-1).astype(np.float32)  # (150, N_y)

    # subsample pixels if layer_sizes[0] is set; otherwise use all N_y
    layer_sizes_t = list(cfg_t["layer_sizes"])
    n_pixels = layer_sizes_t[0]
    if n_pixels is not None:
        idx_px = np.linspace(0, I.shape[1] - 1, n_pixels, dtype=int)
        I = I[:, idx_px]
    else:
        layer_sizes_t[0] = I.shape[1]

    I_min, I_max = I.min(axis=0), I.max(axis=0)
    I_norm = np.where(I_max > I_min, (I - I_min) / (I_max - I_min), 0.0)

    model_t = _build_and_train(
        cfg_t,
        I_norm[idx_tr], I_norm[idx_te], y[idx_tr], y[idx_te],
        layer_sizes=layer_sizes_t,
    )
    model_t.save(os.path.join(args.path_T, "model.pt"))
    np.savez(os.path.join(args.path_T, "data_split.npz"),
             X_tr=I_norm[idx_tr], X_te=I_norm[idx_te], y_tr=y[idx_tr], y_te=y[idx_te])
    print(f"Saved: {args.path_T}/model.pt  +  model_history.json  +  data_split.npz")
