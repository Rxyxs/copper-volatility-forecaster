import numpy as np

from src.baselines import garch_predict_fold, har_rv_predict_fold
from src.data import generate_synthetic_copper_market
from src.features import build_features_and_target


def _sample_df():
    market = generate_synthetic_copper_market(n_days=1200, seed=11)
    df, feature_cols, _ = build_features_and_target(market)
    return df


def test_har_rv_predict_fold_returns_finite_predictions_of_correct_length():
    df = _sample_df()
    n = df.height
    train_idx = np.arange(0, int(n * 0.7))
    val_idx = np.arange(int(n * 0.7), n)

    preds = har_rv_predict_fold(df, train_idx, val_idx, "target_fwd_realized_vol")

    assert preds.shape == val_idx.shape
    assert np.isfinite(preds).all()


def test_garch_predict_fold_returns_finite_positive_predictions_of_correct_length():
    df = _sample_df()
    n = df.height
    train_idx = np.arange(0, int(n * 0.7))
    val_idx = np.arange(int(n * 0.7), n)
    log_returns = df.select("log_return").to_numpy().ravel()

    preds = garch_predict_fold(log_returns, train_idx, val_idx, horizon=5)

    assert preds.shape == val_idx.shape
    assert np.isfinite(preds).all()
    assert (preds > 0).all()


def test_garch_predict_fold_matches_catboost_information_cutoff():
    """Row i's target is realized vol over i+1..i+horizon. Both the ML
    features (via `.shift(1)`) and this GARCH forecast must be built from
    information through day i-1 only -- day i's own return must never
    affect the forecast for row i (it's not part of either model's input),
    while day i-1's return (available to both) must."""
    df = _sample_df()
    n = df.height
    train_idx = np.arange(0, int(n * 0.7))
    val_idx = np.arange(int(n * 0.7), n)
    log_returns = df.select("log_return").to_numpy().ravel()

    preds = garch_predict_fold(log_returns, train_idx, val_idx, horizon=5)

    corrupted_same_day = log_returns.copy()
    corrupted_same_day[val_idx[0]] = 10.0
    preds_same_day = garch_predict_fold(corrupted_same_day, train_idx, val_idx, horizon=5)
    assert preds[0] == preds_same_day[0], "GARCH forecast leaked information from day i itself"

    corrupted_prior_day = log_returns.copy()
    corrupted_prior_day[train_idx[-1]] = 10.0
    preds_prior_day = garch_predict_fold(corrupted_prior_day, train_idx, val_idx, horizon=5)
    assert preds[0] != preds_prior_day[0], "GARCH forecast ignored information from day i-1 (should use it)"
