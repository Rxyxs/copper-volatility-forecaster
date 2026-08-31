"""PyTorch MLP forecaster for forward realized copper volatility.

Third, complementary modeling approach on top of the same lookahead-safe
feature/target pipeline used by CatBoost (`src/modeling.py`) and the
GARCH(1,1)/HAR-RV econometric baselines (`src/baselines.py`): a small
feed-forward network over the identical tabular feature set, trained with a
volatility-aware custom loss and evaluated across three activation
functions (ReLU, GELU, Swish/SiLU) on the same `TimeSeriesSplit` walk-forward
folds, so the three-way comparison (CatBoost / GARCH+HAR-RV / MLP) is
apples-to-apples: same features, same target, same folds.

Loss: `HuberRMSPELoss` combines a Huber term (robust to the rare large
volatility spikes injected in `src/data.py`'s GARCH-X process) with an
RMSPE-style relative term (volatility forecasts are evaluated relatively as
often as absolutely -- a miss of 0.001 matters a lot more in a low-vol
regime than a high-vol one).
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import TimeSeriesSplit
from torch import nn

SEED = 42
ACTIVATIONS = ("relu", "gelu", "swish")
HIDDEN_DIMS = (64, 32)
EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


def _activation_module(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return nn.SiLU()  # SiLU == Swish (x * sigmoid(x))
    raise ValueError(f"Unknown activation: {name}")


class HuberRMSPELoss(nn.Module):
    """Huber loss (robust to rare large-volatility spikes) plus an RMSPE-style
    relative term, since volatility forecasts matter proportionally: a fixed
    absolute error is a much larger relative miss in a low-vol regime than a
    high-vol one."""

    def __init__(self, delta: float = 0.01, rmspe_weight: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.rmspe_weight = rmspe_weight
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        huber_term = self.huber(preds, target)
        rel_sq = ((preds - target) / (target.abs() + self.eps)) ** 2
        rmspe_term = torch.sqrt(rel_sq.mean() + self.eps)
        return huber_term + self.rmspe_weight * rmspe_term


class MLPVolatilityForecaster(nn.Module):
    """Feed-forward network over tabular volatility features. Predicts a
    non-negative forward realized volatility via a Softplus output head."""

    def __init__(self, n_features: int, hidden_dims: tuple[int, ...] = HIDDEN_DIMS, activation: str = "relu"):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(_activation_module(activation))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Softplus())  # volatility is non-negative
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _standardize(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_val - mean) / std


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    activation: str = "relu",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    seed: int = SEED,
) -> dict:
    """Trains one `MLPVolatilityForecaster` and returns predictions, per-epoch
    train/val loss history, and wall-clock inference latency (used for the
    README's latency column)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, Xval = _standardize(X_train, X_val)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.float32)
    Xval_t = torch.tensor(Xval, dtype=torch.float32)
    yval_t = torch.tensor(y_val, dtype=torch.float32)

    model = MLPVolatilityForecaster(n_features=X_train.shape[1], activation=activation)
    criterion = HuberRMSPELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    n = Xtr_t.shape[0]
    train_losses: list[float] = []
    val_losses: list[float] = []

    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        train_losses.append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            val_preds = model(Xval_t)
            val_losses.append(criterion(val_preds, yval_t).item())

    model.eval()
    start_t = time.perf_counter()
    with torch.no_grad():
        final_val_preds = model(Xval_t).numpy()
    latency_ms_per_sample = ((time.perf_counter() - start_t) / max(len(y_val), 1)) * 1000.0

    return {
        "preds": final_val_preds,
        "train_loss_history": train_losses,
        "val_loss_history": val_losses,
        "latency_ms_per_sample": latency_ms_per_sample,
        "model": model,
    }


def _rmse(preds: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - actual) ** 2)))


def _mae(preds: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - actual)))


def run_activation_comparison(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_splits: int = 5,
    epochs: int = EPOCHS,
    seed: int = SEED,
    activations: tuple[str, ...] = ACTIVATIONS,
) -> dict:
    """Walk-forward (`TimeSeriesSplit`) comparison of the MLP across
    activation functions, on the identical folds used for CatBoost/GARCH/
    HAR-RV. Returns per-activation RMSE/MAE plus the last fold's loss curves
    and predictions (used for plotting) for the best-performing activation.
    """
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df.select(target_col).to_numpy().ravel().astype(np.float32)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = list(tscv.split(X))

    results: dict[str, dict] = {act: {"rmse": [], "mae": []} for act in activations}
    last_fold_detail: dict[str, dict] = {}

    for activation in activations:
        for fold_i, (train_idx, val_idx) in enumerate(folds, start=1):
            out = train_mlp(
                X[train_idx], y[train_idx], X[val_idx], y[val_idx], activation=activation, epochs=epochs, seed=seed
            )
            results[activation]["rmse"].append(_rmse(out["preds"], y[val_idx]))
            results[activation]["mae"].append(_mae(out["preds"], y[val_idx]))
            if fold_i == len(folds):
                last_fold_detail[activation] = {
                    "preds": out["preds"],
                    "actual": y[val_idx],
                    "train_loss_history": out["train_loss_history"],
                    "val_loss_history": out["val_loss_history"],
                    "latency_ms_per_sample": out["latency_ms_per_sample"],
                }
        print(
            f"[mlp:{activation}] RMSE mean={np.mean(results[activation]['rmse']):.6f} "
            f"MAE mean={np.mean(results[activation]['mae']):.6f}"
        )

    summary = {
        act: {
            "rmse_mean": float(np.mean(vals["rmse"])),
            "rmse_std": float(np.std(vals["rmse"])),
            "mae_mean": float(np.mean(vals["mae"])),
            "mae_std": float(np.std(vals["mae"])),
            "rmse_per_fold": vals["rmse"],
            "mae_per_fold": vals["mae"],
            "latency_ms_per_sample": last_fold_detail[act]["latency_ms_per_sample"],
        }
        for act, vals in results.items()
    }
    best_activation = min(summary, key=lambda a: summary[a]["rmse_mean"])
    return {
        "by_activation": summary,
        "best_activation": best_activation,
        "best_detail": last_fold_detail[best_activation],
        "all_detail": last_fold_detail,
    }
