"""Lookahead-safe features and the forward realized-volatility target.

Two timing rules, enforced here and tested in ``tests/test_features.py``:

1. **Copper features at row t only use returns up to t-1** (every rolling window runs over
   ``log_return.shift(1)``). The target at row t is realized volatility over t+1..t+5.
2. **Macro features are computed in each series' own calendar and joined by publication
   date.** Row t's forecast is made at the end of the previous calendar day, so it can use
   a VIX close dated up to t-1, and a dollar-index value only once the weekly H.10 release
   that contains it is out (:func:`src.data.dxy_release_cutoff`). A value older than
   ``MAX_STALENESS`` counts as missing instead of being carried forward indefinitely.

Realized volatility is the root mean square of daily log returns, without demeaning: over
five days the sample mean is noise, and it is the quantity a zero-mean GARCH forecasts.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from src.data import Market, dxy_release_cutoff

VOL_TARGET_HORIZON = 5  # predict realized vol over the NEXT 5 trading days
FEATURE_WINDOWS = (5, 10, 20, 60)
MACRO_WINDOWS = (5, 20)
MAX_STALENESS = {"vix": dt.timedelta(days=7), "dxy": dt.timedelta(days=7)}

# Canonical HAR-RV (Corsi 2009) horizons: daily / weekly / monthly.
HAR_WEEKLY_WINDOW = 5
HAR_MONTHLY_WINDOW = 22
HAR_COLUMNS = ["har_rv_daily", "har_rv_weekly", "har_rv_monthly"]
TARGET_COL = "target_fwd_realized_vol"


def _rms(expr: pl.Expr, window: int) -> pl.Expr:
    return expr.pow(2).rolling_mean(window_size=window).sqrt()


def _add_return_features(df: pl.DataFrame, windows: tuple[int, ...]) -> tuple[pl.DataFrame, list[str]]:
    past_return = pl.col("log_return").shift(1)  # never use today's own return
    exprs = []
    for w in windows:
        exprs.append(_rms(past_return, w).alias(f"realized_vol_{w}d"))
        exprs.append(past_return.rolling_mean(window_size=w).alias(f"mean_return_{w}d"))
    df = df.with_columns(exprs)
    df = df.with_columns([pl.col("log_return").shift(lag).alias(f"lag_return_{lag}") for lag in (1, 2, 3)])

    cols = [f"realized_vol_{w}d" for w in windows] + [f"mean_return_{w}d" for w in windows]
    cols += [f"lag_return_{lag}" for lag in (1, 2, 3)]
    return df, cols


def macro_features(
    series: pl.DataFrame, name: str, log_changes: bool, with_level: bool, windows: tuple[int, ...] = MACRO_WINDOWS
) -> tuple[pl.DataFrame, list[str]]:
    """Features of one macro series, one row per observation of that series.

    The VIX enters in points (its level is itself a volatility forecast for US equities);
    the dollar index in log changes, since its level drifts and says nothing about risk.
    """
    value = pl.col("value")
    change = (value.log() - value.log().shift(1)) if log_changes else (value - value.shift(1))
    change_col = f"{name}_change_1d"
    table = series.sort("date").with_columns(change.alias(change_col))
    exprs = []
    cols = [change_col]
    for w in windows:
        exprs.append(pl.col(change_col).rolling_std(window_size=w).alias(f"{name}_change_std_{w}d"))
        exprs.append(pl.col(change_col).abs().rolling_mean(window_size=w).alias(f"{name}_change_abs_mean_{w}d"))
        cols += [f"{name}_change_std_{w}d", f"{name}_change_abs_mean_{w}d"]
    table = table.with_columns(exprs)
    if with_level:
        table = table.with_columns(value.alias(f"{name}_level"))
        cols = [f"{name}_level"] + cols
    return table.select(["date"] + cols).drop_nulls(), cols


def _join_published(df: pl.DataFrame, table: pl.DataFrame, key: str, tolerance: dt.timedelta) -> pl.DataFrame:
    """For each row, the latest ``table`` row dated on or before ``df[key]``."""
    right = table.rename({"date": "_macro_date"}).sort("_macro_date")
    joined = df.sort(key).join_asof(
        right, left_on=key, right_on="_macro_date", strategy="backward", tolerance=tolerance
    )
    return joined.drop("_macro_date").sort("date")


def build_features_and_target(
    market: Market,
    horizon: int = VOL_TARGET_HORIZON,
    windows: tuple[int, ...] = FEATURE_WINDOWS,
) -> tuple[pl.DataFrame, list[str], dict[str, list[str]]]:
    """Returns ``(df, feature_cols, feature_groups)``; ``feature_groups`` tags each feature
    as "return", "macro" or "calendar" for :mod:`src.explainability`."""
    df = market.copper.sort("date")
    df, return_cols = _add_return_features(df, windows)

    # The forecast for row t is made at the end of the previous calendar day.
    df = df.with_columns((pl.col("date") - pl.duration(days=1)).alias("_origin"))
    df = df.with_columns(
        pl.col("_origin").map_elements(dxy_release_cutoff, return_dtype=pl.Date).alias("_dxy_cutoff")
    )
    vix_table, vix_cols = macro_features(market.vix, "vix", log_changes=False, with_level=True)
    dxy_table, dxy_cols = macro_features(market.dxy, "dxy", log_changes=True, with_level=False)
    df = _join_published(df, vix_table, "_origin", MAX_STALENESS["vix"])
    df = _join_published(df, dxy_table, "_dxy_cutoff", MAX_STALENESS["dxy"])
    macro_cols = vix_cols + dxy_cols

    df = df.with_columns([pl.col("date").dt.weekday().alias("day_of_week"), pl.col("date").dt.month().alias("month")])
    calendar_cols = ["day_of_week", "month"]

    # HAR-RV components, kept apart from the ML features so the baseline stays textbook.
    past_return = pl.col("log_return").shift(1)
    df = df.with_columns(
        [
            past_return.abs().alias("har_rv_daily"),
            _rms(past_return, HAR_WEEKLY_WINDOW).alias("har_rv_weekly"),
            _rms(past_return, HAR_MONTHLY_WINDOW).alias("har_rv_monthly"),
        ]
    )

    # Target: realized volatility over the NEXT `horizon` days (t+1 .. t+horizon).
    future_squared = pl.col("log_return").shift(-1).pow(2)
    df = df.with_columns(
        future_squared.rolling_mean(window_size=horizon).shift(-(horizon - 1)).sqrt().alias(TARGET_COL)
    )

    feature_cols = return_cols + macro_cols + calendar_cols
    feature_groups = {"return": return_cols, "macro": macro_cols, "calendar": calendar_cols}
    required = feature_cols + HAR_COLUMNS + [TARGET_COL, "log_return"]
    df = df.drop_nulls(subset=required).drop(["_origin", "_dxy_cutoff"])
    return df, feature_cols, feature_groups
