from src.features import build_features_and_target
from src.tuning import run_optuna_search
from tests.helpers import make_market


def test_optuna_search_finds_valid_params():
    market = make_market(n_days=800, seed=31)
    df, feature_cols, _ = build_features_and_target(market)

    study = run_optuna_search(df, feature_cols, "target_fwd_realized_vol", n_trials=3, val_fraction=0.2)

    assert study.best_value > 0
    assert 3 <= study.best_params["depth"] <= 8
    assert 200 <= study.best_params["iterations"] <= 800


def test_split_for_early_stopping_keeps_the_last_fifth_as_holdout():
    from src.tuning import split_for_early_stopping

    assert split_for_early_stopping(1000) == 800
    assert split_for_early_stopping(10, fraction=0.3) == 7
