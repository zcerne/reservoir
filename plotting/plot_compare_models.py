"""Compare the Reservoir NN (--path) and the Baseline NN (--path_T) trained
by ``train_voltage_reservoir.py``. Reservoir = 200-d I(y) input; Baseline =
raw-feature input. (Legacy MNIST T-matrix bits remain in plot_samples_grid /
plot_input_comparison but are no-ops for non-MNIST datasets.)

Dataset-agnostic — reads ``"dataset"`` from each ``model_data.json`` and adapts
class labels, axis ticks, and crossval data loading for either iris (3 classes,
4 features) or MNIST (10 classes, 196 features = 14×14 block-mean downsample).

Plots produced (saved to ``--fig_dir``):
    * loss_curves.png       — train/val loss vs epoch
    * accuracy.png          — val acc curves + test acc bars
    * confusion.png         — confusion matrices (per model)
    * per_class_accuracy.png
    * crossval.png          — k-fold CV mean ± std
    * samples_grid.png      — (MNIST only) grid of digits with predicted labels
    * stats.json, crossval_stats.json
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # find root core modules
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from class_neural_network import DenseNN, _load_dataset_normalized
from fun_training import (
    _pick_detector_pixels,
    _apply_T_pixelwise,
)


IRIS_LABELS = ["setosa", "versicolor", "virginica"]
WINE_LABELS = ["class_0", "class_1", "class_2"]
SYNTHETIC_LABELS = ["small_r", "mid_r", "large_r"]


def _labels_for(dataset_name: str) -> list:
    if dataset_name == "iris":
        return IRIS_LABELS
    if dataset_name == "wine":
        return WINE_LABELS
    if dataset_name == "synthetic":
        return SYNTHETIC_LABELS
    if dataset_name == "mnist":
        return [str(d) for d in range(10)]
    raise ValueError(f"unknown dataset '{dataset_name}'")


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
        device=cfg.get("device"),
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
        device=cfg.get("device"),
    )


def _dataset_name(cfg_n: dict, cfg_t: dict) -> str:
    """Returns the common dataset name; raises if the two configs disagree."""
    d_n = cfg_n.get("dataset", "iris")
    d_t = cfg_t.get("dataset", d_n)
    if d_n != d_t:
        raise ValueError(f"normal config dataset ({d_n}) != T config dataset ({d_t})")
    return d_n


def plot_loss(path_n: str, path_t: str, fig_dir: str) -> None:
    """All four loss curves (train+val for both models) on a single axis.

    Colour distinguishes model, linestyle distinguishes train vs val.
    """
    _, _, _, h_n, _ = _load_model(path_n)
    _, _, _, h_t, _ = _load_model(path_t)

    fig, ax = plt.subplots(figsize=(8, 5))
    series = [
        ("Reservoir NN — train",   h_n["train_loss"], "steelblue",  "-"),
        ("Reservoir NN — val",     h_n["val_loss"],   "steelblue",  "--"),
        ("Baseline NN — train", h_t["train_loss"], "darkorange", "-"),
        ("Baseline NN — val",   h_t["val_loss"],   "darkorange", "--"),
    ]
    for label, y, color, ls in series:
        ax.plot(range(1, len(y) + 1), y, label=label, color=color, linestyle=ls)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
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
    for h, name in [(h_n, "Reservoir NN"), (h_t, "Baseline NN")]:
        axes[0].plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], label=name)
    axes[0].set_title("Validation accuracy per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(["Reservoir NN", "Baseline NN"], [acc_n, acc_t], color=["steelblue", "darkorange"])
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
        "normal_NN":   {"test_acc": float(acc_n), "best_val_acc": float(max(h_n["val_acc"])),
                        "epochs": len(h_n["val_acc"])},
        "T_matrix_NN": {"test_acc": float(acc_t), "best_val_acc": float(max(h_t["val_acc"])),
                        "epochs": len(h_t["val_acc"])},
    }
    with open(os.path.join(fig_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Reservoir NN   — test acc: {acc_n:.3f}  best val: {max(h_n['val_acc']):.3f}")
    print(f"Baseline NN — test acc: {acc_t:.3f}  best val: {max(h_t['val_acc']):.3f}")
    print(f"Saved: {fig_dir}/accuracy.png  +  stats.json")


def plot_confusion(path_n: str, path_t: str, fig_dir: str) -> None:
    model_n, X_te_n, y_te_n, _, cfg_n = _load_model(path_n)
    model_t, X_te_t, y_te_t, _, cfg_t = _load_model(path_t)
    dataset_name = _dataset_name(cfg_n, cfg_t)
    labels = _labels_for(dataset_name)
    n_classes = len(labels)

    cm_n = confusion_matrix(y_te_n, model_n.predict(X_te_n), labels=np.arange(n_classes))
    cm_t = confusion_matrix(y_te_t, model_t.predict(X_te_t), labels=np.arange(n_classes))

    # MNIST: a 10×10 grid renders better large and without per-cell text;
    # iris/small datasets keep the with-text annotated style.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5) if n_classes <= 4 else (14, 6))
    for ax, cm, name in zip(axes, [cm_n, cm_t], ["Reservoir NN", "Baseline NN"]):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(labels, rotation=45 if n_classes > 4 else 30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{name}  ({dataset_name})")
        # Per-cell counts on every confusion matrix; shrink font and skip
        # zero-cells for large class counts (MNIST 10×10) so it stays legible.
        font_sz = 12 if n_classes <= 4 else max(6, 11 - n_classes // 2)
        thresh = cm.max() / 2 if cm.max() > 0 else 1
        for i in range(n_classes):
            for j in range(n_classes):
                v = cm[i, j]
                if n_classes > 4 and v == 0:
                    continue
                ax.text(j, i, str(int(v)), ha="center", va="center",
                        color="white" if v > thresh else "black",
                        fontsize=font_sz)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "confusion.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir}/confusion.png")


def plot_per_class_accuracy(path_n: str, path_t: str, fig_dir: str) -> None:
    model_n, X_te_n, y_te_n, _, cfg_n = _load_model(path_n)
    model_t, X_te_t, y_te_t, _, cfg_t = _load_model(path_t)
    dataset_name = _dataset_name(cfg_n, cfg_t)
    labels = _labels_for(dataset_name)
    n_classes = len(labels)

    def per_class(model, X, y):
        pred = model.predict(X)
        out = np.zeros(n_classes, dtype=float)
        for c in range(n_classes):
            mask = (y == c)
            out[c] = (pred[mask] == c).mean() if mask.any() else np.nan
        return out

    acc_n = per_class(model_n, X_te_n, y_te_n)
    acc_t = per_class(model_t, X_te_t, y_te_t)

    x = np.arange(n_classes)
    w = 0.4 if n_classes <= 4 else 0.35
    fig, ax = plt.subplots(figsize=(7, 4) if n_classes <= 4 else (12, 4))
    ax.bar(x - w/2, acc_n, w, label="Reservoir NN",   color="steelblue")
    ax.bar(x + w/2, acc_t, w, label="Baseline NN", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if n_classes > 4 else 0, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-class accuracy ({dataset_name})")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "per_class_accuracy.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir}/per_class_accuracy.png")


def plot_input_comparison(path_n: str, path_t: str, fig_dir: str,
                          n_examples: int = 10) -> None:
    """Side-by-side: raw MNIST digit vs. its T-matrix-reservoir intensity map.

    Picks one example per digit (0..9) by default; for non-MNIST datasets,
    picks the first `n_examples` test samples. Both models were trained with
    the same `random_state=42` split, so `y_te` aligns sample-by-sample.

    Layouts
    -------
    raw input  : sqrt(n_features) × sqrt(n_features) grid (14×14 for MNIST)
    T input    : sqrt(n_pixels)    × sqrt(n_pixels)    grid; falls back to
                 1×N when not square. If `pixel_grid` is set in the T config,
                 that shape is honoured exactly.
    """
    _, X_te_n, y_te_n, _, cfg_n = _load_model(path_n)
    _, X_te_t, y_te_t, _, cfg_t = _load_model(path_t)
    dataset_name = _dataset_name(cfg_n, cfg_t)

    def _square_or_strip(vec: np.ndarray, grid_hint=None):
        n = vec.size
        if grid_hint is not None:
            return vec.reshape(int(grid_hint[0]), int(grid_hint[1]))
        side = int(round(np.sqrt(n)))
        if side * side == n:
            return vec.reshape(side, side)
        return vec.reshape(1, n)

    # Choose which samples to show: one per class for MNIST, otherwise the
    # first n_examples of the test set.
    if dataset_name == "mnist":
        classes = list(range(10))
        idx = []
        for c in classes:
            hits = np.where(y_te_n == c)[0]
            if hits.size:
                idx.append(hits[0])
        # If we somehow miss some classes, top up with leading test samples.
        while len(idx) < n_examples and len(idx) < len(y_te_n):
            i = len(idx)
            if i not in idx:
                idx.append(i)
        idx = idx[:n_examples]
    else:
        idx = list(range(min(n_examples, len(y_te_n))))

    n = len(idx)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n + 1, 4),
                              gridspec_kw={"hspace": 0.35})
    if n == 1:
        axes = axes.reshape(2, 1)

    grid_hint_t = cfg_t.get("pixel_grid")

    for col, i in enumerate(idx):
        # Top row: raw input
        raw_img = _square_or_strip(X_te_n[i])
        axR = axes[0, col]
        axR.imshow(raw_img, cmap="gray_r", aspect="equal" if raw_img.ndim == 2 and raw_img.shape[0] > 1 else "auto")
        axR.set_xticks([]); axR.set_yticks([])
        axR.set_title(f"y={int(y_te_n[i])}", fontsize=9)

        # Bottom row: T-matrix reservoir output (intensity)
        t_img = _square_or_strip(X_te_t[i], grid_hint=grid_hint_t)
        axT = axes[1, col]
        axT.imshow(t_img, cmap="inferno", aspect="equal" if t_img.shape[0] > 1 else "auto")
        axT.set_xticks([]); axT.set_yticks([])

    axes[0, 0].set_ylabel("raw\ninput", rotation=0, ha="right", va="center", fontsize=10)
    axes[1, 0].set_ylabel("T·input\n(intensity)", rotation=0, ha="right", va="center", fontsize=10)

    fig.suptitle(f"Raw input vs. T-matrix reservoir output  ({dataset_name})",
                 fontsize=12)
    fig.tight_layout(); fig.subplots_adjust(top=0.86)
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "input_comparison.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir}/input_comparison.png")


def plot_samples_grid(path_n: str, path_t: str, fig_dir: str, n_per_digit: int = 5) -> None:
    """MNIST-only: grid of `n_per_digit` test images per digit, annotated with
    each model's prediction. Skips silently for non-image datasets."""
    model_n, X_te_n, y_te_n, _, cfg_n = _load_model(path_n)
    model_t, _, _, _, cfg_t = _load_model(path_t)
    dataset_name = _dataset_name(cfg_n, cfg_t)
    if dataset_name != "mnist":
        return  # only meaningful for image data

    # Side of the downsampled square (default 14 for 196-feature MNIST)
    side = int(round(np.sqrt(X_te_n.shape[1])))
    if side * side != X_te_n.shape[1]:
        print(f"plot_samples_grid: input dim {X_te_n.shape[1]} isn't a perfect square, skipping")
        return

    pred_n = model_n.predict(X_te_n)
    # The T-matrix model was trained on the T-projected detector pixels, NOT
    # on the raw image — so we can't predict it from X_te_n. We DO have its
    # own test-set predictions via _load_model's X_te_t; we line them up by
    # ground-truth index instead of trying to share inputs.
    _, X_te_t, y_te_t, _, _ = _load_model(path_t)
    pred_t = model_t.predict(X_te_t)

    digits = np.arange(10)
    fig, axes = plt.subplots(len(digits), n_per_digit, figsize=(n_per_digit * 1.2, 11))
    for r, d in enumerate(digits):
        idx_d_n = np.where(y_te_n == d)[0][:n_per_digit]
        idx_d_t = np.where(y_te_t == d)[0][:n_per_digit]
        for c in range(n_per_digit):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(idx_d_n):
                ax.imshow(X_te_n[idx_d_n[c]].reshape(side, side), cmap="gray_r")
                pn = pred_n[idx_d_n[c]]
                pt = pred_t[idx_d_t[c]] if c < len(idx_d_t) else -1
                col = "black"
                # red text if any model got it wrong
                if pn != d or pt != d:
                    col = "tab:red"
                ax.set_title(f"N:{pn}  T:{pt}", fontsize=8, color=col)
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(f"{d}", rotation=0, ha="right", va="center", fontsize=12)

    fig.suptitle("MNIST test samples — N: Reservoir NN prediction, T: Baseline NN prediction "
                 "(red = at least one wrong)", fontsize=11)
    fig.tight_layout(); fig.subplots_adjust(top=0.95)
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "samples_grid.png"), dpi=130)
    plt.close(fig)
    print(f"Saved: {fig_dir}/samples_grid.png")


def plot_crossval(path_n: str, path_t: str, fig_dir: str, n_splits: int = 5) -> None:
    _, _, _, _, cfg_n = _load_model(path_n)
    _, _, _, _, cfg_t = _load_model(path_t)
    dataset_name = _dataset_name(cfg_n, cfg_t)

    # Dispatch on cfg type: T_matrix workflow (cfg_t["T_matrix_source"]) vs
    # direct-features workflow (cfg_t["input_source"]+input_key, e.g. for the
    # voltage_reservoir output saved by generate_iris_data.py).
    if "T_matrix_source" in cfg_t:
        # Build I directly from T_matrix.npz with the same pixel-sampling helper as
        # fun_training (so crossval mirrors the actual training pipeline). Skips
        # the precomputed T_matrix_dataset_*.npz entirely, avoiding the 110 GB
        # intermediate on MNIST.
        source = cfg_t["T_matrix_source"]
        t_path = os.path.join(source, "simulation_T", "T_matrix.npz")
        d = np.load(t_path)
        T_Ey, T_Ex, T_Ez = d["T_Ey"], d["T_Ex"], d["T_Ez"]
        built_with_cw = bool(d["use_cw"])    if "use_cw"    in d.files else False
        run_until     = float(d["run_until"]) if "run_until" in d.files else 0.0
        if not built_with_cw and run_until > 0:
            T_Ey, T_Ex, T_Ez = T_Ey/run_until, T_Ex/run_until, T_Ez/run_until

        X, y = _load_dataset_normalized(dataset_name)
        N_y_total = T_Ey.shape[0]
        n_pixels = list(cfg_t["layer_sizes"])[0]
        if n_pixels is None or n_pixels == N_y_total:
            n_pixels = N_y_total
            idx_px = np.arange(N_y_total)
        else:
            idx_px = _pick_detector_pixels(
                n_pixels=n_pixels, N_y_total=N_y_total,
                mode=cfg_t.get("pixel_sampling", "linspace"),
                seed=cfg_t.get("pixel_sampling_seed", 42),
                source_path=source,
                grid_override=cfg_t.get("pixel_grid"),
            )
        T_X = _apply_T_pixelwise((T_Ey, T_Ex, T_Ez), X, idx_px)
        I   = np.sum(np.abs(T_X) ** 2, axis=-1).astype(np.float32)
        I_min, I_max = I.min(axis=0), I.max(axis=0)
        I_norm = np.where(I_max > I_min, (I - I_min) / (I_max - I_min), 0.0)
        del T_X, T_Ey, T_Ex, T_Ez
    elif "input_source" in cfg_t:
        # Direct-features workflow: both X and the reservoir features come from
        # the same npz pointed at by cfg_t["input_source"]. cfg_n must also
        # point to the SAME npz (use input_key="X" for raw features there).
        src = cfg_t["input_source"]
        in_key = cfg_t.get("input_key", "I_out")
        lbl_key = cfg_t.get("label_key", "labels")
        npz_path = os.path.join(path_t, src) if not os.path.isabs(src) else src
        dd = np.load(npz_path, allow_pickle=True)
        I = np.asarray(dd[in_key], dtype=np.float32)
        y = np.asarray(dd[lbl_key], dtype=np.int64)
        # X for the "normal" branch comes from cfg_n's input_source/input_key on
        # the same data (e.g. raw X column of iris_dataset.npz).
        src_n = cfg_n.get("input_source", src)
        in_key_n = cfg_n.get("input_key", "X")
        npz_path_n = os.path.join(path_n, src_n) if not os.path.isabs(src_n) else src_n
        ddn = np.load(npz_path_n, allow_pickle=True)
        X = np.asarray(ddn[in_key_n], dtype=np.float32)
        # Min-max normalize both (mirrors train_voltage_reservoir).
        def _mm(A):
            a0, a1 = A.min(axis=0), A.max(axis=0)
            r = np.where(a1 > a0, a1 - a0, 1.0)
            return ((A - a0) / r).astype(np.float32)
        X = _mm(X); I_norm = _mm(I)
        n_pixels = I_norm.shape[1]
    else:
        raise KeyError(f"cfg_t must contain either 'T_matrix_source' or 'input_source'")

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

    for fold, (idx_tr, idx_te) in enumerate(kf.split(X, y)):
        model_n = _make_model(cfg_n, X.shape[1])
        model_n.fit(X[idx_tr], y[idx_tr], **fit_kwargs)
        acc_n_folds.append(model_n.score(X[idx_te], y[idx_te]))

        model_t = _make_model(cfg_t, I_norm.shape[1])
        model_t.fit(I_norm[idx_tr], y[idx_tr], **fit_kwargs)
        acc_t_folds.append(model_t.score(I_norm[idx_te], y[idx_te]))

        print(f"Fold {fold+1}/{n_splits}  reservoir={acc_n_folds[-1]:.3f}  baseline={acc_t_folds[-1]:.3f}")

    folds = np.arange(1, n_splits + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(folds, acc_n_folds, "o-", color="steelblue",  label=f"Reservoir NN   (mean={np.mean(acc_n_folds):.3f})")
    ax.plot(folds, acc_t_folds, "o-", color="darkorange", label=f"Baseline NN (mean={np.mean(acc_t_folds):.3f})")
    ax.axhline(float(np.mean(acc_n_folds)), color="steelblue",  linestyle="--", alpha=0.5)
    ax.axhline(float(np.mean(acc_t_folds)), color="darkorange", linestyle="--", alpha=0.5)
    ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{n_splits}-fold cross-validation  ({dataset_name})")
    ax.set_xticks(folds); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "crossval.png"), dpi=150)
    plt.close(fig)

    cv_stats = {
        "normal_NN":   {"mean": float(np.mean(acc_n_folds)), "std": float(np.std(acc_n_folds)),
                        "folds": [float(a) for a in acc_n_folds]},
        "T_matrix_NN": {"mean": float(np.mean(acc_t_folds)), "std": float(np.std(acc_t_folds)),
                        "folds": [float(a) for a in acc_t_folds]},
        "dataset": dataset_name,
        "pixel_sampling": cfg_t.get("pixel_sampling", "n/a"),
        "n_pixels": int(n_pixels),
    }
    with open(os.path.join(fig_dir, "crossval_stats.json"), "w") as f:
        json.dump(cv_stats, f, indent=4)

    print(f"Reservoir NN   — mean={np.mean(acc_n_folds):.3f} ± {np.std(acc_n_folds):.3f}")
    print(f"Baseline NN — mean={np.mean(acc_t_folds):.3f} ± {np.std(acc_t_folds):.3f}")
    print(f"Saved: {fig_dir}/crossval.png  +  crossval_stats.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",     default="data/NN_data/2_normal_model_mnist",
                        help="model_data.json folder for the raw-feature NN")
    parser.add_argument("--path_T",   default="data/NN_data/2_T_model_mnist",
                        help="model_data.json folder for the Baseline NN")
    parser.add_argument("--fig_dir",  default="data/NN_data/2_figures_mnist",
                        help="where to write the comparison plots")
    parser.add_argument("--n_splits", default=5, type=int,
                        help="number of CV folds; pass 0 to skip the cross-val plot")
    parser.add_argument("--skip-samples-grid", action="store_true",
                        help="skip the MNIST sample-grid figure")
    parser.add_argument("--skip-input-comparison", action="store_true",
                        help="skip the raw-vs-T-matrix input comparison figure")
    args = parser.parse_args()

    plot_loss(args.path, args.path_T, args.fig_dir)
    plot_accuracy(args.path, args.path_T, args.fig_dir)
    plot_confusion(args.path, args.path_T, args.fig_dir)
    plot_per_class_accuracy(args.path, args.path_T, args.fig_dir)
    if not args.skip_input_comparison:
        plot_input_comparison(args.path, args.path_T, args.fig_dir)
    if not args.skip_samples_grid:
        plot_samples_grid(args.path, args.path_T, args.fig_dir)
    if args.n_splits > 0:
        plot_crossval(args.path, args.path_T, args.fig_dir, n_splits=args.n_splits)
