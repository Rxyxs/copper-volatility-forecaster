"""Econometric and naive baselines for forward realized-volatility forecasting.

All of them are evaluated on the *identical* `TimeSeriesSplit` folds used for CatBoost (see
`src/modeling.py`), with the same target and the same information cutoff (returns through
day i-1 for row i):

    - **Naive (persistence)**: the next five days will be as volatile as the last five. The
      bar any model has to clear before it is worth anything.
    - **EWMA** (RiskMetrics, lambda = 0.94): an exponentially weighted variance with no
      estimated parameters, the industry default for daily risk.
    - **HAR-RV** (Corsi, 2009): OLS of forward realized vol on the canonical daily / weekly /
      monthly realized-vol components; **HAR-X** adds the VIX level and the dollar index's
      20-day volatility, two regressors fixed before seeing any result.
    - **GARCH(1,1)**: fit on training-fold returns only (`arch` package, zero-mean, normal
      errors, no macro inputs), then a genuine walk-forward, non-refitting multi-step
      forecast for every day in the validation fold via `last_obs`/`start`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from arch import arch_model

from src.features import HAR_COLUMNS

EWMA_LAMBDA = 0.94
EWMA_WARMUP = 20  # returns used to seed the recursion; they sit inside the first training fold


def naive_predict_fold(df: pl.DataFrame, val_idx: np.ndarray) -> np.ndarray:
    """Realized volatility of the last five returns (t-5 .. t-1)."""
    return df["realized_vol_5d"].to_numpy()[val_idx]


def ewma_forecast(log_returns: np.ndarray, lam: float = EWMA_LAMBDA, warmup: int = EWMA_WARMUP) -> np.ndarray:
    """RiskMetrics volatility for every row: sigma2[i] = lam*sigma2[i-1] + (1-lam)*r[i-1]**2.

    Nothing is estimated, so it runs over the whole series at once; row i only uses
    returns through i-1. The forecast is flat over the horizon (EWMA has no mean
    reversion), so it is directly comparable with the 5-day target.
    """
    squared = np.asarray(log_returns, dtype=float) ** 2
    variance = np.empty_like(squared)
    variance[0] = squared[:warmup].mean()
    for i in range(1, len(squared)):
        variance[i] = lam * variance[i - 1] + (1.0 - lam) * squared[i - 1]
    return np.sqrt(variance)


def har_rv_predict_fold(
    df: pl.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    target_col: str,
    columns: list[str] | None = None,
) -> np.ndarray:
    """OLS on the HAR components (or on ``columns``, for the HAR-X variant with macro inputs)."""
    X = df.select(columns or HAR_COLUMNS).to_numpy()
    y = df.select(target_col).to_numpy().ravel()

    X_train = sm.add_constant(X[train_idx], has_constant="add")
    X_val = sm.add_constant(X[val_idx], has_constant="add")

    model = sm.OLS(y[train_idx], X_train).fit()
    preds = model.predict(X_val)
    return np.asarray(preds)


def garch_predict_fold(
    log_returns: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, horizon: int
) -> np.ndarray:
    """Fits GARCH(1,1) on `log_returns[train_idx]` only, then produces a
    genuine walk-forward h-step-ahead variance forecast for every day in
    `val_idx`, using the `arch` package's `last_obs`/`start` mechanism
    (fixed parameters estimated on train only).

    Information-cutoff alignment with the ML feature set matters here: the
    CatBoost/HAR-RV features for row i are built from `log_return.shift(1)`,
    i.e. they only ever see returns up to day i-1. `arch`'s rolling forecast
    at origin o uses the *actual* realized return at day o itself (to
    update the conditional-variance state before projecting forward) --
    verified empirically, not assumed, since it's easy to get backwards.
    To give GARCH the same information cutoff as row i's features (data
    through i-1, not i), the origin used for row i's forecast is **i-1**,
    and since the target at row i is realized vol over days i+1..i+horizon,
    that means requesting an (horizon+1)-step-ahead forecast from origin
    i-1 and using steps h=2..(horizon+1) -- h=1 (day i itself) is discarded,
    since day i is not part of the target window.
    """
    import pandas as pd

    train_end = int(train_idx[-1])
    fold_end = int(val_idx[-1])
    # Slice to this fold's train+val span only -- passing the full series
    # would make `forecast(start=...)` keep producing forecasts past the
    # end of this fold's validation set, misaligning it with `val_idx`.
    scaled_returns = pd.Series(log_returns[: fold_end + 1] * 100.0, index=pd.RangeIndex(fold_end + 1))

    am = arch_model(scaled_returns, mean="Zero", vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(last_obs=train_end + 1, disp="off")

    forecast = res.forecast(horizon=horizon + 1, start=train_end, reindex=False)
    # Origins are train_end .. fold_end; row i's target needs origin i-1,
    # so keep origins train_end .. fold_end-1 (== val_idx - 1).
    variances_scaled = forecast.variance.loc[train_end : fold_end - 1].to_numpy()
    variances = variances_scaled[:, 1:] / (100.0**2)  # drop h.1 (day i itself), keep h.2..h.(horizon+1)

    vol_equivalent = np.sqrt(variances.mean(axis=1))
    return vol_equivalent
