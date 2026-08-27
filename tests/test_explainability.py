import numpy as np

from src.data import generate_synthetic_copper_market
from src.explainability import compute_shap_values, global_importance_by_feature, global_importance_by_group
from src.features import build_features_and_target
from src.modeling import fit_final_model


def _fitted_model_and_data():
    market = generate_synthetic_copper_market(n_days=1000, seed=41)
    df, feature_cols, feature_groups = build_features_and_target(market)
    params = {"iterations": 100, "learning_rate": 0.1, "depth": 4}
    model = fit_final_model(df, feature_cols, "target_fwd_realized_vol", params)
    return model, df, feature_cols, feature_groups


def test_shap_values_shape_matches_features():
    model, df, feature_cols, _ = _fitted_model_and_data()
    _, shap_values, X_df = compute_shap_values(model, df, feature_cols)
    assert shap_values.shape == (df.height, len(feature_cols))
    assert X_df.shape == (df.height, len(feature_cols))


def test_global_importance_by_group_shares_sum_to_100():
    model, df, feature_cols, feature_groups = _fitted_model_and_data()
    _, shap_values, _ = compute_shap_values(model, df, feature_cols)
    group_importance = global_importance_by_group(shap_values, feature_cols, feature_groups)
    assert np.isclose(group_importance["share_pct"].sum(), 100.0, atol=0.1)
    assert set(group_importance["group"]) == set(feature_groups.keys())


def test_global_importance_by_feature_sorted_descending():
    model, df, feature_cols, _ = _fitted_model_and_data()
    _, shap_values, _ = compute_shap_values(model, df, feature_cols)
    importance = global_importance_by_feature(shap_values, feature_cols)
    values = importance["mean_abs_shap"].to_numpy()
    assert (values[:-1] >= values[1:]).all()
    assert set(importance["feature"]) == set(feature_cols)
