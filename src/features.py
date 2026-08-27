"""Lookahead-safe feature engineering for the copper volatility forecaster.

Lookahead-bias rule, enforced everywhere in this module: every feature at
row t is built from values shifted by at least 1, so it only ever sees data
strictly before day t. The target at row t is realized volatility computed
from returns at t+1..t+horizon (strictly future), which is fine for a
*target* but must never leak into a feature column.
"""

from __future__ import annotations

import polars as pl

VOL_TARGET_HORIZON = 5  # predict realized vol over the NEXT 5 trading days
FEATURE_WINDOWS = (5, 10, 20, 60)

# Canonical HAR-RV (Corsi 2009) horizons: daily / weekly / monthly.
HAR_DAILY_WINDOW = 1
HAR_WEEKLY_WINDOW = 5
HAR_MONTHLY_WINDOW = 22


def _add_return_features(df: pl.DataFrame, windows: tuple[int, ...]) -> tuple[pl.DataFrame, list[str]]:
    feature_exprs = []
    for w in windows:
        past_return = pl.col("log_return").shift(1)  # never use today's own return
        feature_exprs.append(past_return.rolling_std(window_size=w).alias(f"realized_vol_{w}d"))
        feature_exprs.append(past_return.rolling_mean(window_size=w).alias(f"mean_return_{w}d"))
    df = df.with_columns(feature_exprs)
    df = df.with_columns([pl.col("log_return").shift(lag).alias(f"lag_return_{lag}") for lag in (1, 2, 3)])

    cols = [f"realized_vol_{w}d" for w in windows] + [f"mean_return_{w}d" for w in windows]
    cols += [f"lag_return_{lag}" for lag in (1, 2, 3)]
    return df, cols


def _add_volume_features(df: pl.DataFrame, windows: tuple[int, ...] = (5, 20)) -> tuple[pl.DataFrame, list[str]]:
    df = df.with_columns((pl.col("volume").log().shift(1)).alias("log_volume_lag1"))
    feature_exprs = [pl.col("log_volume_lag1").alias("log_volume_lag1")]
    for w in windows:
        feature_exprs.append(
            pl.col("log_volume_lag1").rolling_mean(window_size=w).alias(f"log_volume_roll_mean_{w}d")
        )
        feature_exprs.append(
            pl.col("log_volume_lag1").rolling_std(window_size=w).alias(f"log_volume_roll_std_{w}d")
        )
    df = df.with_columns(feature_exprs)
    cols = ["log_volume_lag1"] + [f"log_volume_roll_mean_{w}d" for w in windows] + [
        f"log_volume_roll_std_{w}d" for w in windows
    ]
    return df, cols


def _add_macro_features(
    df: pl.DataFrame, col: str, windows: tuple[int, ...] = (5, 20)
) -> tuple[pl.DataFrame, list[str]]:
    change_col = f"{col}_change_1d"
    df = df.with_columns((pl.col(col) - pl.col(col).shift(1)).shift(1).alias(change_col))
    feature_exprs = [pl.col(change_col)]
    for w in windows:
        feature_exprs.append(pl.col(change_col).rolling_std(window_size=w).alias(f"{col}_roll_std_{w}d"))
        feature_exprs.append(pl.col(change_col).abs().rolling_mean(window_size=w).alias(f"{col}_roll_abs_mean_{w}d"))
    df = df.with_columns(feature_exprs)
    cols = [change_col] + [f"{col}_roll_std_{w}d" for w in windows] + [f"{col}_roll_abs_mean_{w}d" for w in windows]
    return df, cols


def build_features_and_target(
    df: pl.DataFrame,
    horizon: int = VOL_TARGET_HORIZON,
    windows: tuple[int, ...] = FEATURE_WINDOWS,
) -> tuple[pl.DataFrame, list[str], dict[str, list[str]]]:
    """Returns (df, all_feature_cols, feature_groups) where `feature_groups`
    tags each feature as "return", "volume", or "macro" -- used by
    `src.explainability` to aggregate SHAP importance by group."""
    df = df.sort("date").with_columns((pl.col("price").log() - pl.col("price").log().shift(1)).alias("log_return"))

    df, return_cols = _add_return_features(df, windows)
    df, volume_cols = _add_volume_features(df)
    df, usd_cols = _add_macro_features(df, "usd_index")
    df, risk_cols = _add_macro_features(df, "risk_proxy")
    macro_cols = usd_cols + risk_cols

    df = df.with_columns([pl.col("date").dt.weekday().alias("day_of_week"), pl.col("date").dt.month().alias("month")])
    calendar_cols = ["day_of_week", "month"]

    # HAR-RV components (Corsi 2009 canonical daily/weekly/monthly realized
    # vol), computed separately from the ML feature set above so the
    # econometric baseline stays textbook-faithful rather than entangled
    # with arbitrary ML window choices. Uses |return| as the daily RV proxy.
    df = df.with_columns(pl.col("log_return").shift(1).abs().alias("_har_daily"))
    df = df.with_columns(
        [
            pl.col("_har_daily").alias("har_rv_daily"),
            pl.col("_har_daily").rolling_mean(window_size=HAR_WEEKLY_WINDOW).alias("har_rv_weekly"),
            pl.col("_har_daily").rolling_mean(window_size=HAR_MONTHLY_WINDOW).alias("har_rv_monthly"),
        ]
    ).drop("_har_daily")
    har_cols = ["har_rv_daily", "har_rv_weekly", "har_rv_monthly"]

    # Target: realized volatility over the NEXT `horizon` days (strictly
    # future returns from t+1 .. t+horizon), aligned to row t.
    future_return = pl.col("log_return").shift(-1)
    df = df.with_columns(
        future_return.rolling_std(window_size=horizon).shift(-(horizon - 1)).alias("target_fwd_realized_vol")
    )

    feature_cols = return_cols + volume_cols + macro_cols + calendar_cols
    feature_groups = {
        "return": return_cols,
        "volume": volume_cols,
        "macro": macro_cols,
        "calendar": calendar_cols,
    }

    required_cols = feature_cols + har_cols + ["target_fwd_realized_vol"]
    df = df.drop_nulls(subset=required_cols)
    return df, feature_cols, feature_groups
