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


class DenseNN(nn.Module):
    def __init__(
        self,
        layer_sizes: list,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        """layer_sizes = [input_dim, hidden1, hidden2, ..., output_dim]"""
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least [input_dim, output_dim]")
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
        X_val, y_val = X_t[idx[:n_val]], y_t[idx[:n_val]]
        X_tr, y_tr = X_t[idx[n_val:]], y_t[idx[n_val:]]

        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for xb, yb in loader:
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
            logits = self(torch.tensor(X, dtype=torch.float32))
        return logits.argmax(1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self(torch.tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        if self.history:
            history_path = os.path.splitext(path)[0] + "_history.json"
            with open(history_path, "w") as f:
                json.dump(self.history, f)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        history_path = os.path.splitext(path)[0] + "_history.json"
        if os.path.exists(history_path):
            with open(history_path) as f:
                self.history = json.load(f)
        return self



