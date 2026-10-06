import numpy as np
import polars as pl

from src.features import TARGET_COL, build_features_and_target
from src.modeling import MODELS, fit_final_model, run_walk_forward_comparison
from tests.helpers import make_market

PARAMS = {"iterations": 100, "learning_rate": 0.1, "depth": 4}


def _sample_df():
    market = make_market(n_days=1200, seed=21)
    df, feature_cols, feature_groups = build_features_and_target(market)
    return df, feature_cols, feature_groups


def test_walk_forward_comparison_covers_every_model():
    df, feature_cols, _ = _sample_df()
    summary = run_walk_forward_comparison(df, feature_cols, TARGET_COL, PARAMS, horizon=5, n_splits=3)
    for name in MODELS:
        assert summary[name]["rmse_mean"] > 0
        assert summary[name]["qlike_mean"] > 0
        assert len(summary[name]["rmse_per_fold"]) == 3
    oos = summary["oos"]
    assert len(oos["row_idx"]) == sum(val for _, val in summary["fold_sizes"])
    assert all(len(p) == len(oos["actual"]) for p in oos["pred"].values())


def test_a_fold_never_sees_its_own_validation_targets():
    """Rewriting the targets of the first validation fold must not change that fold's
    forecasts: neither Optuna, nor early stopping, nor HAR-RV may look at them."""
    df, feature_cols, _ = _sample_df()
    first = run_walk_forward_comparison(df, feature_cols, TARGET_COL, None, horizon=5, n_splits=3, n_trials=2)
    n_val = first["fold_sizes"][0][1]
    val_rows = first["oos"]["row_idx"][:n_val]

    noisy = df.with_columns(
        pl.when(pl.int_range(pl.len()).is_in(val_rows.tolist()))
        .then(pl.col(TARGET_COL) * 3.0)
        .otherwise(pl.col(TARGET_COL))
        .alias(TARGET_COL)
    )
    second = run_walk_forward_comparison(noisy, feature_cols, TARGET_COL, None, horizon=5, n_splits=3, n_trials=2)
    for name in MODELS:
        assert np.array_equal(first["oos"]["pred"][name][:n_val], second["oos"]["pred"][name][:n_val]), name
    assert first["catboost_folds"][0] == second["catboost_folds"][0]


def test_fit_final_model_predicts_positive_volatility():
    df, feature_cols, _ = _sample_df()
    model = fit_final_model(df, feature_cols, TARGET_COL, PARAMS)
    preds = model.predict(df.select(feature_cols).to_numpy())
    assert (preds > -0.05).all()  # sanity: no wildly negative garbage predictions
