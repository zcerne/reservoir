"""Train two MNIST classifier NNs — one LINEAR, one NONLINEAR — and classify the
test set. Same architecture (196→hidden→10 on 14×14 block-mean MNIST), differing
ONLY in the hidden activation (identity vs sigmoid), mirroring the 05_linearNN /
06_nonlinearNN reference convention. Reports test accuracy + per-class breakdown,
saves the models + a confusion-matrix figure.

  python scripts/train_mnist_classifiers.py --hidden 128 --epochs 40

Outputs (under --out, default data/NN_data/mnist_classifiers/):
  linear_mnist.pt / nonlinear_mnist.pt   (+ _history.json)
  confusion.png,  results.json
"""
from __future__ import annotations
import argparse, os, sys, json

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=128, help="hidden layer width")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--target_size", type=int, default=14, help="MNIST downsample (14→196 feat)")
    ap.add_argument("--nonlinear_act", default="sigmoid", choices=["sigmoid", "tanh", "relu"])
    ap.add_argument("--out", default=os.path.join(_HERE, "data", "NN_data", "mnist_classifiers"))
    args = ap.parse_args()

    import numpy as np
    from class_neural_network import DenseNN, _load_mnist_downsampled_normalized

    os.makedirs(args.out, exist_ok=True)
    feat = args.target_size ** 2
    print(f"[mnist] loading MNIST → {args.target_size}×{args.target_size}={feat} features …", flush=True)
    X, y = _load_mnist_downsampled_normalized(target_size=args.target_size)
    # canonical MNIST split: first 60k train, last 10k test
    Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]
    print(f"[mnist] train {Xtr.shape}  test {Xte.shape}  ({len(np.unique(y))} classes)", flush=True)

    layers = [feat, args.hidden, 10]
    results = {}
    for tag, act in (("linear", "linear"), ("nonlinear", args.nonlinear_act)):
        print(f"\n[{tag}] DenseNN {layers} activation={act}", flush=True)
        net = DenseNN(layers, activation=act)
        net.fit(Xtr, ytr, epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size, val_split=0.1, verbose=True)
        # ---- classify the held-out test set ----
        pred = net.predict(Xte)
        acc = float((pred == yte).mean())
        per_class = {int(c): float((pred[yte == c] == c).mean()) for c in range(10)}
        print(f"[{tag}] TEST accuracy = {acc:.4f}", flush=True)
        net.save(os.path.join(args.out, f"{tag}_mnist.pt"))
        results[tag] = {"activation": act, "test_acc": acc, "per_class_acc": per_class,
                        "layers": layers, "val_acc_final": net.history["val_acc"][-1]}

    # ---- confusion matrices side by side ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        for ax, tag in zip(axes, ("linear", "nonlinear")):
            net = DenseNN(layers, activation=("linear" if tag == "linear" else args.nonlinear_act))
            net.load(os.path.join(args.out, f"{tag}_mnist.pt"))
            cm = confusion_matrix(yte, net.predict(Xte))
            im = ax.imshow(cm, cmap="Blues"); ax.set_title(f"{tag}  (acc {results[tag]['test_acc']:.3f})")
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.set_xticks(range(10)); ax.set_yticks(range(10))
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("MNIST classification — linear vs nonlinear NN")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "confusion.png"), dpi=130, bbox_inches="tight")
        print(f"[mnist] confusion.png saved", flush=True)
    except Exception as e:
        print(f"[mnist] confusion plot skipped: {e}", flush=True)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[mnist] SUMMARY  linear={results['linear']['test_acc']:.4f}  "
          f"nonlinear={results['nonlinear']['test_acc']:.4f}  → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
