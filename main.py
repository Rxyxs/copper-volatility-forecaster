"""
Copper price volatility forecaster.

Predicts forward realized volatility of copper prices with CatBoost, validated
with strict walk-forward time-series cross-validation (TimeSeriesSplit).

Data note: this initial version generates a SYNTHETIC daily copper price series
(GARCH(1,1)-style volatility clustering) as a placeholder so the full pipeline
can be built and validated end to end. It is explicitly labeled as synthetic
below and is meant to be swapped for a real sourced series (e.g. LME/COMEX
copper futures) before any real forecasting is done on it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit

SEED = 42
N_DAYS = 3000
VOL_TARGET_HORIZON = 5          # predict realized vol over the NEXT 5 trading days
FEATURE_WINDOWS = (5, 10, 20, 60)
N_CV_SPLITS = 5


def generate_synthetic_copper_prices(n_days: int = N_DAYS, seed: int = SEED) -> pl.DataFrame:
    """Simulate a daily copper price series with GARCH(1,1) volatility clustering.

    SYNTHETIC DATA — not real copper prices. Used only to exercise the pipeline.
    """
    rng = np.random.default_rng(seed)

    omega, alpha, beta = 1e-6, 0.08, 0.90  # GARCH(1,1) params (alpha+beta < 1 => stationary)
    mu = 0.0002  # small daily drift

    variances = np.empty(n_days)
    returns = np.empty(n_days)
    variances[0] = omega / (1 - alpha - beta)
    returns[0] = mu + np.sqrt(variances[0]) * rng.standard_normal()

    for t in range(1, n_days):
        variances[t] = omega + alpha * returns[t - 1] ** 2 + beta * variances[t - 1]
        returns[t] = mu + np.sqrt(variances[t]) * rng.standard_normal()

    price0 = 4.00  # USD/lb, roughly realistic copper price scale
    prices = price0 * np.exp(np.cumsum(returns))

    start = dt.date(2015, 1, 1)
    end = start + dt.timedelta(days=n_days - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True)

    return pl.DataFrame({"date": dates, "price": prices})


def build_features_and_target(
    df: pl.DataFrame,
    horizon: int = VOL_TARGET_HORIZON,
    windows: tuple[int, ...] = FEATURE_WINDOWS,
) -> pl.DataFrame:
    """Build lookahead-safe features and the forward realized-volatility target.

    Lookahead-bias rule: every feature at row t is built from `log_return` values
    shifted by at least 1, so it only ever sees data strictly before day t. The
    target at row t is realized volatility computed from returns at t+1..t+horizon
    (strictly future), which is fine for a *target* but must never leak into a
    feature column.
    """
    df = df.sort("date").with_columns(
        (pl.col("price").log() - pl.col("price").log().shift(1)).alias("log_return")
    )

    feature_exprs = []
    for w in windows:
        past_return = pl.col("log_return").shift(1)  # never use today's own return
        feature_exprs.append(
            past_return.rolling_std(window_size=w).alias(f"realized_vol_{w}d")
        )
        feature_exprs.append(
            past_return.rolling_mean(window_size=w).alias(f"mean_return_{w}d")
        )

    df = df.with_columns(feature_exprs)

    # Lagged raw returns (t-1, t-2, t-3) as short-memory features.
    df = df.with_columns(
        [pl.col("log_return").shift(lag).alias(f"lag_return_{lag}") for lag in (1, 2, 3)]
    )

    # Calendar features are safe (known in advance, no leakage).
    df = df.with_columns(
        [
            pl.col("date").dt.weekday().alias("day_of_week"),
            pl.col("date").dt.month().alias("month"),
        ]
    )

    # Target: realized volatility over the NEXT `horizon` days (strictly future
    # returns from t+1 .. t+horizon), aligned to row t. This uses future data by
    # construction because it IS the forecast target, not a feature.
    future_return = pl.col("log_return").shift(-1)
    df = df.with_columns(
        future_return.rolling_std(window_size=horizon)
        .shift(-(horizon - 1))
        .alias("target_fwd_realized_vol")
    )

    feature_cols = [c for c in df.columns if c.startswith(("realized_vol_", "mean_return_", "lag_return_"))]
    feature_cols += ["day_of_week", "month"]

    df = df.drop_nulls(subset=feature_cols + ["target_fwd_realized_vol"])
    return df, feature_cols


def run_time_series_cv(
    df: pl.DataFrame, feature_cols: list[str], target_col: str, n_splits: int = N_CV_SPLITS
) -> None:
    """Strict walk-forward CV: TimeSeriesSplit guarantees every validation fold
    comes strictly after its training fold in time, so no future information
    ever informs a model evaluated on the past.
    """
    X = df.select(feature_cols).to_numpy()
    y = df.select(target_col).to_numpy().ravel()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmses = []
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_pool = Pool(X_train, y_train, feature_names=feature_cols)
        val_pool = Pool(X_val, y_val, feature_names=feature_cols)

        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=SEED,
            verbose=False,
        )
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

        preds = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))
        mae = float(np.mean(np.abs(preds - y_val)))
        fold_rmses.append(rmse)
        fold_maes.append(mae)

        print(
            f"[fold {fold}/{n_splits}] train={len(train_idx):>5d} val={len(val_idx):>5d} "
            f"RMSE={rmse:.6f} MAE={mae:.6f}"
        )

    print("-" * 60)
    print(f"CV mean RMSE: {np.mean(fold_rmses):.6f} (+/- {np.std(fold_rmses):.6f})")
    print(f"CV mean MAE : {np.mean(fold_maes):.6f} (+/- {np.std(fold_maes):.6f})")


def main() -> None:
    print("Generating synthetic copper price series (GARCH(1,1) volatility)...")
    prices = generate_synthetic_copper_prices()

    print("Building lookahead-safe features and forward-volatility target...")
    df, feature_cols = build_features_and_target(prices)
    print(f"Rows after feature/target construction: {df.height}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    print(f"\nRunning TimeSeriesSplit CV ({N_CV_SPLITS} folds)...")
    run_time_series_cv(df, feature_cols, target_col="target_fwd_realized_vol")


if __name__ == "__main__":
    main()
