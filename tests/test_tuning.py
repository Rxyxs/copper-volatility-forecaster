from src.data import generate_synthetic_copper_market
from src.features import build_features_and_target
from src.tuning import run_optuna_search


def test_optuna_search_finds_valid_params():
    market = generate_synthetic_copper_market(n_days=800, seed=31)
    df, feature_cols, _ = build_features_and_target(market)

    study = run_optuna_search(df, feature_cols, "target_fwd_realized_vol", n_trials=3, val_fraction=0.2)

    assert study.best_value > 0
    assert 3 <= study.best_params["depth"] <= 8
    assert 200 <= study.best_params["iterations"] <= 800
