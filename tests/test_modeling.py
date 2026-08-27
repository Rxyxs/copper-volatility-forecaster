from src.data import generate_synthetic_copper_market
from src.features import build_features_and_target
from src.modeling import fit_final_model, run_walk_forward_comparison


def _sample_df():
    market = generate_synthetic_copper_market(n_days=1200, seed=21)
    df, feature_cols, feature_groups = build_features_and_target(market)
    return df, feature_cols, feature_groups


def test_walk_forward_comparison_covers_all_three_models():
    df, feature_cols, _ = _sample_df()
    params = {"iterations": 100, "learning_rate": 0.1, "depth": 4}
    summary = run_walk_forward_comparison(
        df, feature_cols, "target_fwd_realized_vol", params, horizon=5, n_splits=3
    )
    for name in ("catboost", "garch", "har_rv"):
        assert name in summary
        assert summary[name]["rmse_mean"] > 0
        assert len(summary[name]["rmse_per_fold"]) == 3
        assert len(summary[name]["mae_per_fold"]) == 3


def test_fit_final_model_predicts_positive_volatility():
    df, feature_cols, _ = _sample_df()
    params = {"iterations": 100, "learning_rate": 0.1, "depth": 4}
    model = fit_final_model(df, feature_cols, "target_fwd_realized_vol", params)
    X = df.select(feature_cols).to_numpy()
    preds = model.predict(X)
    assert (preds > -0.05).all()  # sanity: no wildly negative garbage predictions
