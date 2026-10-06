"""Forecast-evaluation metrics for volatility.

Realized volatility over five days is a noisy measure of the true (unobservable) volatility,
so the loss used to rank models matters. RMSE on volatility is reported because it is easy
to read, but **QLIKE** (Patton, 2011) is the one that ranks forecasts the same way the true
variance would despite that noise, and it punishes under-forecasting risk harder than
over-forecasting it. Differences between two models are tested with **Diebold-Mariano**,
using a Newey-West variance because the 5-day targets of consecutive rows overlap.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

VOL_FLOOR = 1e-6  # a forecast of zero volatility has an infinite QLIKE loss


def rmse(preds: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(preds) - np.asarray(actual)) ** 2)))


def mae(preds: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(preds) - np.asarray(actual))))


def qlike_losses(pred_vol: np.ndarray, actual_vol: np.ndarray, floor: float = VOL_FLOOR) -> np.ndarray:
    """Per-row QLIKE on variances, normalized so a perfect forecast scores 0:
    ``rv/h - log(rv/h) - 1`` with ``h = pred**2`` and ``rv = actual**2``."""
    h = np.maximum(np.asarray(pred_vol, dtype=float), floor) ** 2
    rv = np.maximum(np.asarray(actual_vol, dtype=float), floor) ** 2
    ratio = rv / h
    return ratio - np.log(ratio) - 1.0


def qlike(pred_vol: np.ndarray, actual_vol: np.ndarray) -> float:
    return float(np.mean(qlike_losses(pred_vol, actual_vol)))


def squared_errors(preds: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return (np.asarray(preds, dtype=float) - np.asarray(actual, dtype=float)) ** 2


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, max_lags: int) -> dict:
    """Tests whether model A's mean loss differs from model B's.

    A negative statistic means A has the lower loss. ``max_lags`` should be at least
    ``horizon - 1`` when targets overlap.
    """
    diff = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    fit = sm.OLS(diff, np.ones_like(diff)).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    return {
        "mean_loss_difference": float(diff.mean()),
        "statistic": float(fit.tvalues[0]),
        "p_value": float(fit.pvalues[0]),
        "n": int(diff.size),
    }
