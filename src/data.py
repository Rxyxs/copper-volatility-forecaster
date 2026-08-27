"""Synthetic copper market data generator (price, volume, macro proxies).

SYNTHETIC DATA -- not real LME/COMEX prices. Generates a daily series with a
*documented, injected* causal structure so that the macro/volume SHAP
explainability results (see `src/explainability.py`) can be validated
against a known ground truth instead of assumed correct:

    - Copper log-returns follow a GARCH(1,1) process (volatility clustering).
    - The conditional variance is a **GARCH-X** (exogenous-augmented) process:
      the prior day's absolute USD-index shock and risk-proxy shock both feed
      the variance equation with a one-day lag. Both macro series are
      genuine, injected *leading indicators* of copper volatility, not
      decorative columns.
    - `usd_index` is a mean-reverting (OU-style) process, independent of the
      copper return innovation, representing a dollar-strength proxy
      (commodities are typically dollar-denominated, so USD shocks are a
      textbook driver of commodity volatility).
    - `risk_proxy` is a mean-reverting process with occasional upward jumps
      (Poisson-triggered), representing a global risk-off / demand-shock
      proxy (copper is a growth-cyclical "Dr. Copper" commodity).
    - `volume` is tied to the underlying conditional volatility path plus a
      day-of-week seasonality effect and idiosyncratic noise -- volume
      leading/coinciding with volatility clustering is a standard market
      microstructure stylized fact.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

SEED = 42
N_DAYS = 3000
PRICE_0 = 4.00  # USD/lb, roughly realistic copper price scale

# GARCH(1,1) core parameters (alpha + beta < 1 => stationary).
GARCH_OMEGA = 1e-6
GARCH_ALPHA = 0.08
GARCH_BETA = 0.90
DRIFT_MU = 0.0002

# GARCH-X exogenous coefficients: how much yesterday's *raw* macro shock
# (in the macro series' own units, e.g. USD-index points) feeds today's
# conditional variance (in daily log-return^2 units). Sized so a typical
# (1-sigma) USD shock adds roughly 3-4% of the unconditional variance, and a
# risk-proxy jump (rare, large) adds a genuine multi-sigma vol spike -- the
# copper-specific GARCH dynamics (alpha/beta) still dominate day to day;
# macro is a real but secondary, occasionally dominant driver.
GAMMA_USD = 1.0e-5
GAMMA_RISK = 0.6e-5

# Hard ceiling on conditional variance, expressed as a multiple of the
# unconditional (typical) variance -- a standard simulation safeguard for
# exogenous-augmented GARCH so that a rare cluster of large macro shocks
# can't compound into a runaway/overflowing price path. In practice this
# ceiling is only ever approached, never binding on typical days.
MAX_VARIANCE_MULTIPLE = 40.0

# USD index (OU/AR(1) mean-reverting process).
USD_LONG_RUN = 100.0
USD_KAPPA = 0.02
USD_SIGMA = 0.35

# Global risk/demand proxy (OU + occasional upward jumps).
RISK_LONG_RUN = 20.0
RISK_KAPPA = 0.03
RISK_SIGMA = 0.60
RISK_JUMP_PROB = 0.01
RISK_JUMP_SCALE = 4.0

# Volume: tied to conditional volatility + weekday seasonality + noise.
VOLUME_BASE = 12_000.0
VOLUME_VOL_GAMMA = 1.2
VOLUME_NOISE_SIGMA = 0.18
VOLUME_WEEKDAY_EFFECT = {1: 0.02, 2: 0.05, 3: 0.04, 4: 0.00, 5: -0.10, 6: 0.0, 7: 0.0}  # polars weekday: Mon=1..Sun=7


def generate_synthetic_copper_market(n_days: int = N_DAYS, seed: int = SEED) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    variances = np.empty(n_days)
    returns = np.empty(n_days)
    usd_index = np.empty(n_days)
    risk_proxy = np.empty(n_days)
    usd_shock = np.zeros(n_days)
    risk_shock = np.zeros(n_days)
    volume = np.empty(n_days)

    typical_var = GARCH_OMEGA / (1 - GARCH_ALPHA - GARCH_BETA)
    max_variance = MAX_VARIANCE_MULTIPLE * typical_var

    variances[0] = typical_var
    returns[0] = DRIFT_MU + np.sqrt(variances[0]) * rng.standard_normal()
    usd_index[0] = USD_LONG_RUN
    risk_proxy[0] = RISK_LONG_RUN

    for t in range(1, n_days):
        variances[t] = min(
            max_variance,
            GARCH_OMEGA
            + GARCH_ALPHA * returns[t - 1] ** 2
            + GARCH_BETA * variances[t - 1]
            + GAMMA_USD * usd_shock[t - 1] ** 2
            + GAMMA_RISK * risk_shock[t - 1] ** 2,
        )
        returns[t] = DRIFT_MU + np.sqrt(variances[t]) * rng.standard_normal()

        usd_shock[t] = USD_SIGMA * rng.standard_normal()
        usd_index[t] = usd_index[t - 1] + USD_KAPPA * (USD_LONG_RUN - usd_index[t - 1]) + usd_shock[t]

        jump = RISK_JUMP_SCALE * rng.exponential() if rng.random() < RISK_JUMP_PROB else 0.0
        risk_innovation = RISK_SIGMA * rng.standard_normal() + jump
        risk_shock[t] = risk_innovation
        risk_proxy[t] = max(
            1.0, risk_proxy[t - 1] + RISK_KAPPA * (RISK_LONG_RUN - risk_proxy[t - 1]) + risk_innovation
        )

    prices = PRICE_0 * np.exp(np.cumsum(returns))

    typical_vol_scale = np.sqrt(typical_var)
    start = dt.date(2015, 1, 1)
    dates = pl.date_range(start=start, end=start + dt.timedelta(days=n_days - 1), interval="1d", eager=True)
    weekdays = dates.dt.weekday().to_numpy()
    for t in range(n_days):
        weekday_effect = VOLUME_WEEKDAY_EFFECT.get(int(weekdays[t]), 0.0)
        # Clipped so that even an extreme vol spike produces a large but
        # bounded volume surge (a few hundred x normal), not an overflow.
        vol_ratio = np.clip(np.sqrt(variances[t]) / typical_vol_scale - 1.0, -1.0, 4.0)
        volume[t] = VOLUME_BASE * np.exp(
            VOLUME_VOL_GAMMA * vol_ratio + weekday_effect + VOLUME_NOISE_SIGMA * rng.standard_normal()
        )

    return pl.DataFrame(
        {
            "date": dates,
            "price": prices,
            "volume": volume,
            "usd_index": usd_index,
            "risk_proxy": risk_proxy,
        }
    )
