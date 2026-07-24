"""
Phase 3b — Classical FiLM neural network (PyTorch).

Matches the classical branch of the paper (Sec. III C / Table II):
  - FiLM on epicenter coordinates (2 features)
  - MLP with 3 hidden layers × 100 units, ReLU, Dropout 0.5
  - 5 logits for next-adjacent-node classification
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .utils import (
    FILM_DIM,
    MAIN_DIM,
    MODEL_CHECKPOINT,
    N_OUTPUTS,
    ensure_dirs,
)


class FiLMGenerator(nn.Module):
    """
    FiLM generator: epicenter (x, y) → (γ, β) for feature-wise linear modulation.

    Paper: 'passing these coordinates into two fully-connected layers.
    The layers then become multiplicative and additive values for the
    main body of the neural network.'
    """

    def __init__(self, film_dim: int = FILM_DIM, hidden: int = 64, out_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(film_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim * 2),  # γ || β
        )
        self.out_dim = out_dim

    def forward(self, epi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gb = self.net(epi)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta


class ClassicalFiLMNetwork(nn.Module):
    """
    Classical FiLM NN — paper Sec. III C:

    MLP with three fully connected hidden layers (100 nodes), ReLU,
    Dropout 0.5, five output logits. FiLM modulates after the first
    hidden projection using epicenter coordinates.
    """

    def __init__(
        self,
        main_dim: int = MAIN_DIM,
        film_dim: int = FILM_DIM,
        hidden: int = 100,
        n_outputs: int = N_OUTPUTS,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.film = FiLMGenerator(film_dim=film_dim, hidden=64, out_dim=hidden)
        self.fc1 = nn.Linear(main_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_outputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 36) — Table I layout
        epi = x[:, :2]
        main = x[:, 2:]  # 34 features
        gamma, beta = self.film(epi)

        h = self.fc1(main)
        h = gamma * h + beta  # FiLM modulation
        h = self.dropout(F.relu(h))
        h = self.dropout(F.relu(self.fc2(h)))
        h = self.dropout(F.relu(self.fc3(h)))
        return self.out(h)  # 5 logits


def train_film_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[str] = None,
    checkpoint: Path = MODEL_CHECKPOINT,
) -> Tuple[ClassicalFiLMNetwork, Dict[str, float]]:
    """Train classical FiLM on Dijkstra labels; save checkpoint. Returns (model, metrics)."""
    ensure_dirs()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassicalFiLMNetwork().to(device)

    # Train/val split
    n = len(y)
    idx = np.random.default_rng(0).permutation(n)
    split = int(0.85 * n)
    tr, va = idx[:split], idx[split:]

    def loader(subset):
        # torch.tensor(np.asarray(...)) — Cloud-safe; avoid from_numpy ABI issues
        xb = torch.tensor(np.asarray(X[subset], dtype=np.float32), dtype=torch.float32)
        yb = torch.tensor(np.asarray(y[subset], dtype=np.int64), dtype=torch.long)
        return DataLoader(TensorDataset(xb, yb), batch_size=batch_size, shuffle=True)

    train_loader, val_loader = loader(tr), loader(va)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = None
    metrics: Dict[str, float] = {}

    print(f"[QuantumRelief] Training Classical FiLM on {device} ({epochs} epochs)…")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        model.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                vloss += loss.item() * len(yb)
                vcorrect += (logits.argmax(1) == yb).sum().item()
                vtotal += len(yb)

        train_acc = correct / max(total, 1)
        val_acc = vcorrect / max(vtotal, 1)
        vloss /= max(vtotal, 1)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"  epoch {epoch:3d}/{epochs}  "
                f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  val_loss={vloss:.4f}"
            )
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            metrics = {
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "val_loss": float(vloss),
                "best_epoch": float(epoch),
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "model_state": model.state_dict(),
        "metrics": metrics,
        "arch": {
            "main_dim": MAIN_DIM,
            "film_dim": FILM_DIM,
            "hidden": 100,
            "n_outputs": N_OUTPUTS,
        },
    }
    torch.save(payload, checkpoint)
    print(
        f"[QuantumRelief] Saved model → {checkpoint} "
        f"(val_acc={metrics.get('val_acc', 0):.3f})"
    )
    return model, metrics


def load_film_model(
    checkpoint: Path = MODEL_CHECKPOINT,
    device: Optional[str] = None,
) -> ClassicalFiLMNetwork:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassicalFiLMNetwork()
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
        model.load_state_dict(state)
        print(f"[QuantumRelief] Loaded FiLM checkpoint from {checkpoint}")
    else:
        print("[QuantumRelief] No checkpoint found — using randomly initialised weights.")
    model.to(device)
    model.eval()
    return model


def _to_float32_batch(x: np.ndarray) -> np.ndarray:
    """Copy-safe float32 array shaped (B, F) with finite values."""
    arr = np.array(x, dtype=np.float32, copy=True)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D features, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(arr)


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a CPU tensor to ndarray without requiring torch↔numpy ABI bridge."""
    t = t.detach().cpu()
    try:
        return np.asarray(t.numpy(), dtype=np.float32)
    except (RuntimeError, ImportError, TypeError, AttributeError) as exc:
        # PyTorch raises RuntimeError("Numpy is not available.") on bad Cloud installs
        msg = str(exc).lower()
        if "numpy" not in msg and not isinstance(exc, (ImportError, TypeError, AttributeError)):
            raise
        return np.asarray(t.tolist(), dtype=np.float32)


def predict_logits(model: nn.Module, x: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    """Run FiLM / hybrid forward; copy-safe tensor conversion (Cloud-safe)."""
    device = device or next(model.parameters()).device
    model.eval()
    arr = _to_float32_batch(x)
    with torch.no_grad():
        # Always copy via np.asarray → torch.tensor (never rely on from_numpy)
        t = torch.tensor(np.asarray(arr, dtype=np.float32), dtype=torch.float32, device=device)
        try:
            out = model(t)
        except RuntimeError as exc:
            # Hybrid / PennyLane path can fail if torch cannot talk to NumPy
            if "numpy" in str(exc).lower() and hasattr(model, "classical"):
                out = model.classical(t)
            else:
                raise RuntimeError(
                    f"Model forward failed ({type(exc).__name__}: {exc}). "
                    "Often caused by a NumPy/PyTorch ABI mismatch on Streamlit Cloud — "
                    "pin numpy==1.26.4 before torch==2.2.2 in requirements.txt."
                ) from exc
        logits = _tensor_to_numpy(out)
    return logits


def ensure_trained_model(
    epochs: int = 30,
    n_episodes: int = 60,
) -> Tuple[ClassicalFiLMNetwork, Dict[str, np.ndarray]]:
    """Load dataset + model; generate/train on first run if missing."""
    from .dataset_generation import generate_dataset, load_dataset
    from .utils import DATASET_PATH

    if not DATASET_PATH.exists():
        ds = generate_dataset(n_episodes=n_episodes)
    else:
        ds = load_dataset()

    if not MODEL_CHECKPOINT.exists():
        train_film_model(ds["X"], ds["y"], epochs=epochs)
    model = load_film_model()
    return model, ds


if __name__ == "__main__":
    model, ds = ensure_trained_model(epochs=20, n_episodes=40)
    logits = predict_logits(model, ds["X"][:3])
    print("sample logits:", logits)
