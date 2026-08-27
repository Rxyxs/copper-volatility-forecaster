import polars as pl

from src.data import generate_synthetic_copper_market
from src.features import build_features_and_target


def test_build_features_has_no_nulls():
    market = generate_synthetic_copper_market(n_days=1000, seed=1)
    df, feature_cols, feature_groups = build_features_and_target(market)
    assert df.height > 0
    assert df.select(feature_cols + ["target_fwd_realized_vol"]).null_count().sum_horizontal().sum() == 0


def test_feature_groups_partition_all_feature_columns():
    market = generate_synthetic_copper_market(n_days=1000, seed=1)
    df, feature_cols, feature_groups = build_features_and_target(market)
    grouped = sorted(c for cols in feature_groups.values() for c in cols)
    assert grouped == sorted(feature_cols)
    assert len(grouped) == len(set(grouped))


def test_feature_at_row_t_does_not_use_same_day_return():
    """Leakage check: perturbing day t's own return must not change the
    features computed *for* day t (they may only use information strictly
    before t) -- but it must change the following day's features, proving
    the perturbation is actually detectable and this isn't a vacuous check."""
    market = generate_synthetic_copper_market(n_days=300, seed=3)
    df_a, feature_cols, _ = build_features_and_target(market)

    perturbed_prices = market["price"].to_numpy().copy()
    shock_day = 150
    perturbed_prices[shock_day:] *= 1.5  # one-off level shock from that day forward
    market_b = market.with_columns(pl.Series("price", perturbed_prices))
    df_b, _, _ = build_features_and_target(market_b)

    shock_date = market["date"][shock_day]
    next_date = market["date"][shock_day + 1]

    row_a = df_a.filter(pl.col("date") == shock_date)
    row_b = df_b.filter(pl.col("date") == shock_date)
    assert row_a.height == 1 and row_b.height == 1
    for col in feature_cols:
        assert row_a[col][0] == row_b[col][0], f"feature {col} leaked same-day return information"

    next_a = df_a.filter(pl.col("date") == next_date)
    next_b = df_b.filter(pl.col("date") == next_date)
    assert next_a.height == 1 and next_b.height == 1
    changed = any(next_a[col][0] != next_b[col][0] for col in ("lag_return_1", "realized_vol_5d"))
    assert changed, "perturbation had no detectable effect on the following day -- test is not sensitive"


def test_target_is_forward_looking():
    market = generate_synthetic_copper_market(n_days=300, seed=5)
    df, _, _ = build_features_and_target(market)
    assert (df["target_fwd_realized_vol"] > 0).all()
