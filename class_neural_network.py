import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from class_simulation_T import SimulationT


def _load_iris_normalized():
    data = load_iris()
    X = data.data.astype(float)   # (150, 4)
    y = data.target                # (150,)  0=setosa 1=versicolor 2=virginica
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_norm, y


def _load_mnist_downsampled_normalized(target_size: int = 14):
    """Load full MNIST and block-mean-downsample each digit to (target_size,
    target_size). Returns (X, y) with X shape (70000, target_size**2) ∈ [0, 1]
    and y shape (70000,) ∈ {0..9}.

    Downsample factor `28 // target_size` must divide 28 exactly (so 14, 7, 4,
    2 are valid). Uses sklearn's openml fetcher (auto-caches in
    ~/scikit_learn_data/, first call downloads ~50 MB).
    """
    from sklearn.datasets import fetch_openml
    if 28 % target_size != 0:
        raise ValueError(f"target_size={target_size} must divide 28")
    factor = 28 // target_size

    cache = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = cache.data.astype(np.float32) / 255.0          # (70000, 784) in [0,1]
    y = cache.target.astype(int)
    X28 = X.reshape(-1, 28, 28)
    # Block-mean downsample: e.g. for factor=2 -> reshape(-1, 14, 2, 14, 2).mean((2,4))
    X_ds = X28.reshape(-1, target_size, factor, target_size, factor).mean(axis=(2, 4))
    X_flat = X_ds.reshape(-1, target_size * target_size)
    return X_flat, y


def _load_dataset_normalized(name: str):
    """Dispatch on dataset name. Returns (X, y) with X normalized to [0,1].

    Supported keywords:
        'iris'  -> 150 samples, 4 features, 3 classes
        'mnist' -> 70000 samples, 196 features (14×14 block-mean), 10 classes
    """
    name = name.lower().strip()
    if name == "iris":
        return _load_iris_normalized()
    if name == "mnist":
        return _load_mnist_downsampled_normalized(target_size=14)
    raise ValueError(f"unknown dataset '{name}' (use 'iris' or 'mnist')")


class DenseNN(nn.Module):
    def __init__(
        self,
        layer_sizes: list,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        device: str | None = None,
    ):
        """layer_sizes = [input_dim, hidden1, hidden2, ..., output_dim]

        device: "cuda" / "cpu" / "mps" / None. None auto-picks
        torch.cuda.is_available() → "cuda", else "cpu". Tensors fed to .fit /
        .predict / .score are moved to this device internally.
        """
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least [input_dim, output_dim]")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "linear": nn.Identity,
        }
        act_cls = activations[activation]

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(act_cls())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.history = {}
        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        val_split: float = 0.1,
        verbose: bool = True,
    ) -> dict:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        n_val = max(1, int(len(X_t) * val_split))
        idx = torch.randperm(len(X_t))
        X_val = X_t[idx[:n_val]].to(self.device)
        y_val = y_t[idx[:n_val]].to(self.device)
        X_tr,  y_tr  = X_t[idx[n_val:]], y_t[idx[n_val:]]

        # Pin host memory + non_blocking copies if on CUDA (a free speedup).
        pin = (self.device.type == "cuda")
        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=batch_size, shuffle=True, pin_memory=pin,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=pin)
                yb = yb.to(self.device, non_blocking=pin)
                optimizer.zero_grad()
                loss = criterion(self(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            scheduler.step()

            self.eval()
            with torch.no_grad():
                val_logits = self(X_val)
                val_loss = criterion(val_logits, y_val).item()
                val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
            self.train()

            history["train_loss"].append(train_loss / len(X_tr))
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"epoch {epoch+1:4d}/{epochs}  "
                    f"loss={history['train_loss'][-1]:.4f}  "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
                )

        self.history = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xb = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            logits = self(xb)
        return logits.argmax(1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xb = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            logits = self(xb)
        return torch.softmax(logits, dim=1).cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        if self.history:
            history_path = os.path.splitext(path)[0] + "_history.json"
            with open(history_path, "w") as f:
                json.dump(self.history, f)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
        history_path = os.path.splitext(path)[0] + "_history.json"
        if os.path.exists(history_path):
            with open(history_path) as f:
                self.history = json.load(f)
        return self



