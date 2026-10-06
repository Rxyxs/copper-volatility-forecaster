import datetime as dt

import numpy as np
import polars as pl

from src.data import Market, add_log_returns, dxy_release_cutoff
from src.features import TARGET_COL, build_features_and_target
from tests.helpers import make_market


def _with(market: Market, **frames) -> Market:
    return Market(
        copper=frames.get("copper", market.copper),
        vix=frames.get("vix", market.vix),
        dxy=frames.get("dxy", market.dxy),
        report=market.report,
    )


def _shift_value_on(frame: pl.DataFrame, day: dt.date, factor: float) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.col("date") == day).then(pl.col("value") * factor).otherwise(pl.col("value")).alias("value")
    )


def test_build_features_has_no_nulls():
    df, feature_cols, _ = build_features_and_target(make_market(n_days=1000, seed=1))
    assert df.height > 0
    assert df.select(feature_cols + [TARGET_COL]).null_count().sum_horizontal().sum() == 0


def test_feature_groups_partition_all_feature_columns():
    df, feature_cols, feature_groups = build_features_and_target(make_market(n_days=1000, seed=1))
    grouped = sorted(c for cols in feature_groups.values() for c in cols)
    assert grouped == sorted(feature_cols)
    assert len(grouped) == len(set(grouped))
    assert set(feature_groups) == {"return", "macro", "calendar"}


def test_feature_at_row_t_does_not_use_same_day_return():
    """Perturbing day t's own return must not change the features computed *for* day t,
    but must change the following day's features (so the check is not vacuous)."""
    market = make_market(n_days=300, seed=3)
    df_a, feature_cols, _ = build_features_and_target(market)

    prices = market.copper["price"].to_numpy().copy()
    shock_day = 150
    prices[shock_day:] *= 1.5
    copper_b, _ = add_log_returns(market.copper.select("date").with_columns(pl.Series("price", prices)))
    df_b, _, _ = build_features_and_target(_with(market, copper=copper_b))

    shock_date = market.copper["date"][shock_day]
    next_date = market.copper["date"][shock_day + 1]
    row_a, row_b = df_a.filter(pl.col("date") == shock_date), df_b.filter(pl.col("date") == shock_date)
    assert row_a.height == 1 and row_b.height == 1
    for col in feature_cols:
        assert row_a[col][0] == row_b[col][0], f"feature {col} leaked same-day return information"

    next_a, next_b = df_a.filter(pl.col("date") == next_date), df_b.filter(pl.col("date") == next_date)
    assert any(next_a[c][0] != next_b[c][0] for c in ("lag_return_1", "realized_vol_5d"))


def test_vix_features_only_use_closes_from_before_the_row():
    market = make_market(n_days=400, seed=4)
    df_a, _, _ = build_features_and_target(market)
    day = df_a["date"][200]
    previous = market.vix.filter(pl.col("date") < day)["date"].max()

    df_same_day, _, _ = build_features_and_target(_with(market, vix=_shift_value_on(market.vix, day, 3.0)))
    df_previous, _, _ = build_features_and_target(_with(market, vix=_shift_value_on(market.vix, previous, 3.0)))

    row = lambda df: df.filter(pl.col("date") == day)  # noqa: E731
    assert row(df_same_day)["vix_level"][0] == row(df_a)["vix_level"][0], "the row saw its own day's VIX close"
    assert row(df_previous)["vix_level"][0] == 3.0 * row(df_a)["vix_level"][0]


def test_dxy_features_wait_for_the_weekly_release():
    market = make_market(n_days=400, seed=5)
    df_a, _, _ = build_features_and_target(market)
    day = df_a["date"][250]
    cutoff = dxy_release_cutoff(day - dt.timedelta(days=1))
    published = market.dxy.filter(pl.col("date") <= cutoff)["date"].max()
    unpublished = market.dxy.filter(pl.col("date") > cutoff)["date"].min()
    assert unpublished < day  # it existed by then, it just had not been released yet

    def row(dxy):
        df, _, _ = build_features_and_target(_with(market, dxy=dxy))
        return df.filter(pl.col("date") == day)["dxy_change_1d"][0]

    assert row(_shift_value_on(market.dxy, unpublished, 1.05)) == df_a.filter(pl.col("date") == day)["dxy_change_1d"][0]
    assert row(_shift_value_on(market.dxy, published, 1.05)) != df_a.filter(pl.col("date") == day)["dxy_change_1d"][0]


def test_target_is_realized_vol_of_the_next_five_returns():
    market = make_market(n_days=300, seed=6)
    df, _, _ = build_features_and_target(market)
    copper = market.copper
    day = df["date"][100]
    position = copper["date"].to_list().index(day)
    future = copper["log_return"].to_numpy()[position + 1 : position + 6]
    expected = np.sqrt(np.mean(future**2))
    assert np.isclose(df.filter(pl.col("date") == day)[TARGET_COL][0], expected, rtol=1e-12)


def test_rows_whose_windows_touch_a_hole_are_dropped():
    market = make_market(n_days=600, seed=7)
    copper = market.copper.with_columns(
        pl.when(pl.int_range(pl.len()) == 300).then(None).otherwise(pl.col("log_return")).alias("log_return")
    )
    df, _, _ = build_features_and_target(_with(market, copper=copper))
    hole_date = copper["date"][300]
    dates = df["date"].to_list()
    # no row in the 5 days before the hole (their target would include it) or within the
    # 60 rows after it (their trailing windows would)
    position = market.copper["date"].to_list()
    nearby = position[300 - 5 : 300 + 61]
    assert not set(nearby) & set(dates)
    assert hole_date not in dates
    assert df.height > 300
