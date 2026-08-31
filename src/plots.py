"""Explanatory plots for the three-way model comparison (CatBoost, GARCH/
HAR-RV baselines, PyTorch MLP): predicted-vs-actual, residual distribution,
and MLP loss curves by activation function. Written to `outputs/plots/`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots"

COLORS = {
    "catboost": "#d97706",
    "garch": "#2563eb",
    "har_rv": "#16a34a",
    "mlp": "#dc2626",
    "relu": "#2563eb",
    "gelu": "#dc2626",
    "swish": "#16a34a",
}


def _ensure_dir() -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR


def plot_predicted_vs_actual(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]], filename: str = "predicted_vs_actual.png"
) -> Path:
    """`predictions` maps model name -> (preds, actual) arrays (same actual
    across models, but kept per-model so callers can pass differently sized
    val slices)."""
    out_dir = _ensure_dir()
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5), squeeze=False)
    axes = axes[0]

    for ax, (name, (preds, actual)) in zip(axes, predictions.items()):
        color = COLORS.get(name, "#6b7280")
        ax.scatter(actual, preds, alpha=0.35, s=14, color=color, edgecolors="none")
        lo, hi = float(min(actual.min(), preds.min())), float(max(actual.max(), preds.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#6b7280", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Actual forward realized vol")
        ax.set_ylabel("Predicted forward realized vol")
        ax.grid(alpha=0.25)

    fig.suptitle("Predicted vs. actual forward realized volatility (last walk-forward fold)")
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_residual_distribution(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]], filename: str = "residual_distribution.png"
) -> Path:
    out_dir = _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (preds, actual) in predictions.items():
        residuals = preds - actual
        color = COLORS.get(name, "#6b7280")
        ax.hist(residuals, bins=40, alpha=0.45, label=name, color=color, density=True)
    ax.axvline(0.0, color="#111827", linewidth=1, linestyle="--")
    ax.set_xlabel("Residual (predicted - actual)")
    ax.set_ylabel("Density")
    ax.set_title("Residual distribution by model (last walk-forward fold)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_mlp_loss_curves(
    loss_histories: dict[str, dict[str, list[float]]], filename: str = "mlp_loss_curves.png"
) -> Path:
    """`loss_histories` maps activation name -> {"train_loss_history": [...],
    "val_loss_history": [...]}."""
    out_dir = _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    for activation, hist in loss_histories.items():
        color = COLORS.get(activation, "#6b7280")
        epochs = range(1, len(hist["train_loss_history"]) + 1)
        ax.plot(epochs, hist["train_loss_history"], color=color, linestyle="-", label=f"{activation} (train)")
        ax.plot(epochs, hist["val_loss_history"], color=color, linestyle="--", label=f"{activation} (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber + RMSPE loss")
    ax.set_title("MLP training curves by activation function")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
