"""A small synthetic market with the same schema as `src.data.load_market`, for unit tests.

The results in the README come from the real series. This only exists so the tests run
offline and fast: GARCH(1,1) copper returns on business days, a VIX that reacts to them
and a random-walk dollar index.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from src.data import Market, add_log_returns


def business_days(start: dt.date, n: int) -> list[dt.date]:
    span = np.arange(np.datetime64(start), np.datetime64(start) + int(n * 1.5) + 10)
    return span[np.is_busday(span)][:n].astype("datetime64[D]").tolist()


def make_market(n_days: int = 1000, seed: int = 0, start: dt.date = dt.date(2015, 1, 5)) -> Market:
    rng = np.random.default_rng(seed)
    dates = business_days(start, n_days)

    omega, alpha, beta = 2e-6, 0.06, 0.92
    variance = omega / (1 - alpha - beta)
    returns = np.empty(n_days)
    for t in range(n_days):
        returns[t] = np.sqrt(variance) * rng.standard_normal()
        variance = omega + alpha * returns[t] ** 2 + beta * variance
    price = 3.0 * np.exp(np.cumsum(returns))

    vix = np.empty(n_days)
    vix[0] = 18.0
    for t in range(1, n_days):
        vix[t] = vix[t - 1] + 0.05 * (18.0 - vix[t - 1]) + 300.0 * abs(returns[t]) - 3.0 + rng.normal(0, 0.6)
    vix = np.clip(vix, 9.0, None)
    dxy = 110.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n_days)))

    copper = pl.DataFrame({"date": dates, "price": price}, schema={"date": pl.Date, "price": pl.Float64})
    copper, report = add_log_returns(copper)
    schema = {"date": pl.Date, "value": pl.Float64}
    return Market(
        copper=copper,
        vix=pl.DataFrame({"date": dates, "value": vix}, schema=schema),
        dxy=pl.DataFrame({"date": dates, "value": dxy}, schema=schema),
        report=report,
    )
