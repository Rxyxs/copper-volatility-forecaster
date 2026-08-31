import numpy as np
import torch

from src.data import generate_synthetic_copper_market
from src.deep_learning import (
    HuberRMSPELoss,
    MLPVolatilityForecaster,
    run_activation_comparison,
    train_mlp,
)
from src.features import build_features_and_target


def _sample_df():
    market = generate_synthetic_copper_market(n_days=900, seed=41)
    df, feature_cols, _ = build_features_and_target(market)
    return df, feature_cols


def test_mlp_forward_pass_is_nonnegative_and_correct_shape():
    model = MLPVolatilityForecaster(n_features=10, activation="gelu")
    x = torch.randn(16, 10)
    out = model(x)
    assert out.shape == (16,)
    assert (out >= 0).all()


def test_huber_rmspe_loss_is_zero_for_perfect_predictions():
    loss_fn = HuberRMSPELoss()
    target = torch.tensor([0.01, 0.02, 0.03])
    loss = loss_fn(target, target)
    assert loss.item() < 1e-3  # bounded below only by the loss's internal eps floor


def test_train_mlp_returns_predictions_of_correct_length_for_each_activation():
    df, feature_cols = _sample_df()
    X = df.select(feature_cols).to_numpy().astype("float32")
    y = df.select("target_fwd_realized_vol").to_numpy().ravel().astype("float32")
    split = int(len(X) * 0.8)

    for activation in ("relu", "gelu", "swish"):
        out = train_mlp(X[:split], y[:split], X[split:], y[split:], activation=activation, epochs=5)
        assert out["preds"].shape == y[split:].shape
        assert np.isfinite(out["preds"]).all()
        assert len(out["train_loss_history"]) == 5
        assert len(out["val_loss_history"]) == 5
        assert out["latency_ms_per_sample"] >= 0


def test_run_activation_comparison_covers_all_activations():
    df, feature_cols = _sample_df()
    result = run_activation_comparison(df, feature_cols, "target_fwd_realized_vol", n_splits=2, epochs=3)

    assert set(result["by_activation"].keys()) == {"relu", "gelu", "swish"}
    for activation, metrics in result["by_activation"].items():
        assert metrics["rmse_mean"] > 0
        assert len(metrics["rmse_per_fold"]) == 2
    assert result["best_activation"] in {"relu", "gelu", "swish"}
    assert "preds" in result["best_detail"]
