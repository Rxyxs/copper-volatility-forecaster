"""Explanatory plots for the three-way model comparison (CatBoost, GARCH/
HAR-RV baselines, PyTorch MLP): predicted-vs-actual, residual distribution,
and MLP loss curves by activation function. Written to `outputs/plots/`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.animation import FuncAnimation

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


def _subsample(values: list[float], max_points: int = 50) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= max_points:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_points).round().astype(int)
    return arr[idx]


def plot_mlp_loss_curves_animated(
    loss_histories: dict[str, dict[str, list[float]]], filename: str = "mlp_loss_curves_animated.gif"
) -> Path:
    """Racing-line GIF version of `plot_mlp_loss_curves`: progressively draws
    the real train/val loss curves (subsampled if long) with a floating
    label at the advancing tip of each line showing the series name and
    current loss value. Uses the same `loss_histories` data as the static
    PNG -- no fabricated numbers."""
    out_dir = _ensure_dir()

    series = []
    for activation, hist in loss_histories.items():
        color = COLORS.get(activation, "#6b7280")
        series.append((f"{activation} (train)", _subsample(hist["train_loss_history"]), color, "-"))
        series.append((f"{activation} (val)", _subsample(hist["val_loss_history"]), color, "--"))

    n_frames = min(60, max(len(s[1]) for s in series))
    n_frames = max(n_frames, 2)

    all_x_max = max(len(s[1]) for s in series)
    all_y = np.concatenate([s[1] for s in series])
    y_lo, y_hi = float(all_y.min()), float(all_y.max())
    pad = (y_hi - y_lo) * 0.1 or 0.05

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6))
        lines = []
        labels = []
        for name, ys, color, style in series:
            (line,) = ax.plot([], [], color=color, linestyle=style, linewidth=2, label=name)
            lines.append(line)
            label = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=8,
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.85),
            )
            labels.append(label)

        ax.set_xlim(1, all_x_max)
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_xlabel("Epoch (subsampled)")
        ax.set_ylabel("Huber + RMSPE loss")
        ax.set_title("MLP training curves by activation function (animated)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)
        fig.tight_layout()

        def update(frame):
            step = frame + 1
            for (name, ys, color, style), line, label in zip(series, lines, labels):
                n = max(1, round(step / n_frames * len(ys)))
                n = min(n, len(ys))
                xs = np.arange(1, n + 1)
                line.set_data(xs, ys[:n])
                cur_x, cur_y = xs[-1], ys[n - 1]
                label.xy = (cur_x, cur_y)
                label.set_text(f"{name}: {cur_y:.4f}")
            return lines + labels

        ani = FuncAnimation(fig, update, frames=n_frames, interval=120, blit=False)
        path = out_dir / filename
        ani.save(path, writer="pillow")
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
