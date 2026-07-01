"""Train a NN on voltage_reservoir output. Reads:

    <path>/model_data.json   — config: layer_sizes, lr, epochs, val/test split, …
    <path>/<input_source>    — npz with feature array under `input_key` (default
                                "I_out") + labels under `label_key` ("labels").

Saves model.pt, model_history.json, data_split.npz next to model_data.json.
Computes test accuracy and writes it to model_history.json + prints summary.

Usage:
    python train_voltage_reservoir.py --path data/NN_data/3_electrodes_iris
"""
from __future__ import annotations
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # find root core modules
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from class_neural_network import DenseNN


def _load_config(folder: Path) -> dict:
    with open(folder / "model_data.json") as f:
        return json.load(f)


def _load_features(folder: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    src = cfg.get("input_source", "iris_dataset.npz")
    in_key = cfg.get("input_key", "I_out")
    lbl_key = cfg.get("label_key", "labels")
    p = folder / src
    if not p.exists():
        raise FileNotFoundError(
            f"input npz not found at {p} — generate it first "
            f"(e.g. `python generate_iris_data.py`)")
    d = np.load(p, allow_pickle=True)
    if in_key not in d.files:
        raise KeyError(f"{in_key!r} not in {p} (have {d.files})")
    if lbl_key not in d.files:
        raise KeyError(f"{lbl_key!r} not in {p} (have {d.files})")
    X = np.asarray(d[in_key], dtype=np.float32)
    y = np.asarray(d[lbl_key], dtype=np.int64)
    return X, y


def _normalize_features(X: np.ndarray) -> np.ndarray:
    """Min-max normalize per-feature to [0,1]. Avoids huge magnitude shifts that
    hurt cross-entropy + Kaiming init."""
    x_min = X.min(axis=0); x_max = X.max(axis=0)
    rng = np.where(x_max > x_min, x_max - x_min, 1.0)
    return (X - x_min) / rng


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True, help="folder containing model_data.json")
    ap.add_argument("--no-normalize", action="store_true",
                    help="skip per-feature min-max normalization (default normalizes)")
    args = ap.parse_args()

    folder = Path(args.path).resolve()
    cfg = _load_config(folder)
    print(f"=== {folder.name} ===")
    print(f"  dataset: {cfg.get('dataset')}")
    print(f"  layer_sizes: {cfg['layer_sizes']}")
    print(f"  epochs: {cfg.get('epochs', 100)}, lr: {cfg.get('lr', 1e-3)}, "
          f"batch_size: {cfg.get('batch_size', 32)}")

    X, y = _load_features(folder, cfg)
    print(f"  X shape: {X.shape}, y shape: {y.shape}, "
          f"classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    if X.shape[1] != cfg['layer_sizes'][0]:
        raise ValueError(
            f"X has {X.shape[1]} features but layer_sizes[0] = {cfg['layer_sizes'][0]}. "
            f"Edit model_data.json's layer_sizes[0].")

    if not args.no_normalize:
        X = _normalize_features(X)
        print(f"  per-feature min-max normalized to [0,1]")

    # Train / test split (stratified)
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)),
        test_size=cfg.get("test_split", 0.2),
        random_state=42, stratify=y,
    )
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    print(f"  train/test: {len(idx_tr)}/{len(idx_te)}")

    model = DenseNN(
        layer_sizes=list(cfg["layer_sizes"]),
        activation=cfg.get("activation", "relu"),
        dropout=float(cfg.get("dropout", 0.0)),
        batch_norm=bool(cfg.get("batch_norm", False)),
    )
    print(f"\n=== Training ===")
    history = model.fit(
        X_tr, y_tr,
        epochs=int(cfg.get("epochs", 100)),
        lr=float(cfg.get("lr", 1e-3)),
        batch_size=int(cfg.get("batch_size", 32)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        val_split=float(cfg.get("val_split", 0.1)),
        verbose=True,
    )

    # Evaluate test accuracy
    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te, dtype=torch.float32).to(model.device)
        pred = model(X_te_t).argmax(dim=1).cpu().numpy()
    test_acc = float((pred == y_te).mean())
    print(f"\n=== Test accuracy: {test_acc*100:.2f}% ({(pred == y_te).sum()}/{len(y_te)}) ===")

    # Persist
    out_model = folder / "model.pt"
    torch.save(model.state_dict(), out_model)
    print(f"saved {out_model}")

    history["test_accuracy"] = test_acc
    history["test_size"] = len(y_te)
    history["train_size"] = len(y_tr)
    with open(folder / "model_history.json", "w") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in history.items()},
                  f, indent=2)
    print(f"saved {folder / 'model_history.json'}")

    np.savez(folder / "data_split.npz",
             X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
             idx_tr=idx_tr, idx_te=idx_te)
    print(f"saved {folder / 'data_split.npz'}")


if __name__ == "__main__":
    main()
