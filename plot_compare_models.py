import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from class_neural_network import DenseNN, _load_iris_normalized


LABELS = ["setosa", "versicolor", "virginica"]


def _load_model(path: str) -> tuple:
    with open(os.path.join(path, "model_data.json")) as f:
        cfg = json.load(f)
    data = np.load(os.path.join(path, "data_split.npz"))
    X_te, y_te = data["X_te"].astype(np.float32), data["y_te"]

    with open(os.path.join(path, "model_history.json")) as f:
        history = json.load(f)

    layer_sizes = list(cfg["layer_sizes"])
    if layer_sizes[0] is None:
        layer_sizes[0] = X_te.shape[1]

    model = DenseNN(
        layer_sizes=layer_sizes,
        activation=cfg.get("activation", "relu"),
        dropout=cfg.get("dropout", 0.0),
        batch_norm=cfg.get("batch_norm", False),
    )
    model.load(os.path.join(path, "model.pt"))
    model.history = history
    return model, X_te, y_te, history, cfg


def _make_model(cfg: dict, input_dim: int) -> DenseNN:
    layer_sizes = list(cfg["layer_sizes"])
    if layer_sizes[0] is None:
        layer_sizes[0] = input_dim
    return DenseNN(
        layer_sizes=layer_sizes,
        activation=cfg.get("activation", "relu"),
        dropout=cfg.get("dropout", 0.0),
        batch_norm=cfg.get("batch_norm", False),
    )


def plot_loss(path_n: str, path_t: str, fig_dir: str) -> None:
    _, _, _, h_n, _ = _load_model(path_n)
    _, _, _, h_t, _ = _load_model(path_t)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, h, name in zip(axes, [h_n, h_t], ["Normal NN", "T-matrix NN"]):
        epochs = range(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], label="train loss")
        ax.plot(epochs, h["val_loss"],   label="val loss")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir}/loss_curves.png")


def plot_accuracy(path_n: str, path_t: str, fig_dir: str) -> None:
    model_n, X_te_n, y_te_n, h_n, _ = _load_model(path_n)
    model_t, X_te_t, y_te_t, h_t, _ = _load_model(path_t)

    acc_n = model_n.score(X_te_n, y_te_n)
    acc_t = model_t.score(X_te_t, y_te_t)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for h, name in [(h_n, "Normal NN"), (h_t, "T-matrix NN")]:
        axes[0].plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], label=name)
    axes[0].set_title("Validation accuracy per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(["Normal NN", "T-matrix NN"], [acc_n, acc_t], color=["steelblue", "darkorange"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Test accuracy comparison")
    for i, v in enumerate([acc_n, acc_t]):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=12)

    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "accuracy.png"), dpi=150)
    plt.close(fig)

    stats = {
        "normal_NN":   {"test_acc": acc_n, "best_val_acc": max(h_n["val_acc"]), "epochs": len(h_n["val_acc"])},
        "T_matrix_NN": {"test_acc": acc_t, "best_val_acc": max(h_t["val_acc"]), "epochs": len(h_t["val_acc"])},
    }
    with open(os.path.join(fig_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Normal NN   — test acc: {acc_n:.3f}  best val: {max(h_n['val_acc']):.3f}")
    print(f"T-matrix NN — test acc: {acc_t:.3f}  best val: {max(h_t['val_acc']):.3f}")
    print(f"Saved: {fig_dir}/accuracy.png  +  stats.json")


def plot_confusion(path_n: str, path_t: str, fig_dir: str) -> None:
    model_n, X_te_n, y_te_n, _, _ = _load_model(path_n)
    model_t, X_te_t, y_te_t, _, _ = _load_model(path_t)

    cm_n = confusion_matrix(y_te_n, model_n.predict(X_te_n))
    cm_t = confusion_matrix(y_te_t, model_t.predict(X_te_t))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cm, name in zip(axes, [cm_n, cm_t], ["Normal NN", "T-matrix NN"]):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, rotation=30, ha="right")
        ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(name)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "confusion.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir}/confusion.png")


def plot_per_class_accuracy(path_n: str, path_t: str, fig_dir: str) -> None:
    model_n, X_te_n, y_te_n, _, _ = _load_model(path_n)
    model_t, X_te_t, y_te_t, _, _ = _load_model(path_t)

    def per_class(model, X, y):
        pred = model.predict(X)
        return [(pred[y == c] == c).mean() for c in range(3)]

    acc_n = per_class(model_n, X_te_n, y_te_n)
    acc_t = per_class(model_t, X_te_t, y_te_t)

    x = np.arange(3)
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, acc_n, w, label="Normal NN",   color="steelblue")
    ax.bar(x + w/2, acc_t, w, label="T-matrix NN", color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy"); ax.set_title("Per-class accuracy")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "per_class_accuracy.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir}/per_class_accuracy.png")


def plot_crossval(path_n: str, path_t: str, fig_dir: str, n_splits: int = 5) -> None:
    _, _, _, _, cfg_n = _load_model(path_n)
    _, _, _, _, cfg_t = _load_model(path_t)

    source = cfg_t["T_matrix_source"]
    dataset_path = os.path.join(source, "NN_dataset", "T_matrix_dataset.npz")
    data = np.load(dataset_path, allow_pickle=True)
    T_iris_X = data["T_iris_X"]
    I = np.sum(np.abs(T_iris_X) ** 2, axis=-1).astype(np.float32)

    n_pixels = list(cfg_t["layer_sizes"])[0]
    if n_pixels is not None:
        idx_px = np.linspace(0, I.shape[1] - 1, n_pixels, dtype=int)
        I = I[:, idx_px]

    I_min, I_max = I.min(axis=0), I.max(axis=0)
    I_norm = np.where(I_max > I_min, (I - I_min) / (I_max - I_min), 0.0)

    X_iris, y = _load_iris_normalized()

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_n_folds, acc_t_folds = [], []

    fit_kwargs = dict(
        epochs=cfg_n.get("epochs", 100),
        lr=cfg_n.get("lr", 1e-3),
        batch_size=cfg_n.get("batch_size", 32),
        weight_decay=cfg_n.get("weight_decay", 1e-4),
        val_split=cfg_n.get("val_split", 0.1),
        verbose=False,
    )

    for fold, (idx_tr, idx_te) in enumerate(kf.split(X_iris, y)):
        model_n = _make_model(cfg_n, X_iris.shape[1])
        model_n.fit(X_iris[idx_tr], y[idx_tr], **fit_kwargs)
        acc_n_folds.append(model_n.score(X_iris[idx_te], y[idx_te]))

        model_t = _make_model(cfg_t, I_norm.shape[1])
        model_t.fit(I_norm[idx_tr], y[idx_tr], **fit_kwargs)
        acc_t_folds.append(model_t.score(I_norm[idx_te], y[idx_te]))

        print(f"Fold {fold+1}/{n_splits}  normal={acc_n_folds[-1]:.3f}  T-matrix={acc_t_folds[-1]:.3f}")

    folds = np.arange(1, n_splits + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(folds, acc_n_folds, "o-", color="steelblue",   label=f"Normal NN   (mean={np.mean(acc_n_folds):.3f})")
    ax.plot(folds, acc_t_folds, "o-", color="darkorange", label=f"T-matrix NN (mean={np.mean(acc_t_folds):.3f})")
    ax.axhline(np.mean(acc_n_folds), color="steelblue",   linestyle="--", alpha=0.5)
    ax.axhline(np.mean(acc_t_folds), color="darkorange", linestyle="--", alpha=0.5)
    ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{n_splits}-fold cross-validation")
    ax.set_xticks(folds); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "crossval.png"), dpi=150)
    plt.close(fig)

    cv_stats = {
        "normal_NN":   {"mean": float(np.mean(acc_n_folds)), "std": float(np.std(acc_n_folds)), "folds": acc_n_folds},
        "T_matrix_NN": {"mean": float(np.mean(acc_t_folds)), "std": float(np.std(acc_t_folds)), "folds": acc_t_folds},
    }
    with open(os.path.join(fig_dir, "crossval_stats.json"), "w") as f:
        json.dump(cv_stats, f, indent=4)

    print(f"Normal NN   — mean={np.mean(acc_n_folds):.3f} ± {np.std(acc_n_folds):.3f}")
    print(f"T-matrix NN — mean={np.mean(acc_t_folds):.3f} ± {np.std(acc_t_folds):.3f}")
    print(f"Saved: {fig_dir}/crossval.png  +  crossval_stats.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",     default="data/NN_data/1_normal_model")
    parser.add_argument("--path_T",   default="data/NN_data/1_T_model")
    parser.add_argument("--fig_dir",  default="data/NN_data/figures")
    parser.add_argument("--n_splits", default=5, type=int)
    args = parser.parse_args()

    plot_loss(args.path, args.path_T, args.fig_dir)
    plot_accuracy(args.path, args.path_T, args.fig_dir)
    plot_confusion(args.path, args.path_T, args.fig_dir)
    plot_per_class_accuracy(args.path, args.path_T, args.fig_dir)
    plot_crossval(args.path, args.path_T, args.fig_dir, n_splits=args.n_splits)
