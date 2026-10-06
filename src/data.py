"""Parses and cleans the raw copper, VIX and dollar-index files (offline, pure functions).

The copper series (mindicador.cl, ``libra_cobre``) has three properties that matter for a
volatility model, all checked here rather than assumed:

- **Duplicate days.** Identical copies are collapsed (there is one in the sample); a day with
  two *different* prices raises, because there is no way to know which one is right.
- **It follows the Chilean calendar.** On Chilean holidays the London Metal Exchange trades
  but the series has no value, so the next return spans more than one business day. 95% of
  the returns span exactly one; the ones that span 2 to 4 are kept as they are. Many of
  those holidays are London holidays too, so dividing every multi-day return by the square
  root of the business days elapsed would be wrong: it leaves them *less* volatile than an
  ordinary day (0.94% vs 1.30% daily standard deviation). Doing it right would need the LME
  holiday calendar, which is not in the data.
- **Two real holes**: 7 business days around Christmas 2014 and 18 in December 2017. A
  return across ``MAX_SESSIONS_PER_RETURN`` or more business days is not a daily return,
  so it is set to null and no feature, target or model ever uses it.

Each copper value is dated by mindicador with the day it is published for: a download made
on 2026-10-06 already contains a price dated 2026-10-07. The timing rules in
:mod:`src.features` hold under either reading of that date.

FRED writes missing observations (US holidays) as ``.``; those rows are dropped.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from src.sources import COPPER_FIRST_YEAR, RAW_DIR, copper_path, fred_path

# The analysis is frozen at this date so that re-running it reproduces the README even
# after the sources publish new data. Move it forward (and re-run) to update the results.
SAMPLE_END = dt.date(2026, 10, 2)
MAX_SESSIONS_PER_RETURN = 5
PRICE_BOUNDS_USD_PER_LB = (0.5, 20.0)


@dataclass
class Market:
    """The three cleaned series. ``copper`` has ``date, price, sessions, log_return``
    (``log_return`` is null across a hole); ``vix`` and ``dxy`` have ``date, value``."""

    copper: pl.DataFrame
    vix: pl.DataFrame
    dxy: pl.DataFrame
    report: dict = field(default_factory=dict)


def parse_mindicador(payloads: Iterable[dict]) -> tuple[pl.DataFrame, dict]:
    """``date, price`` from mindicador's yearly JSON payloads, validated."""
    values: dict[dt.date, list[float]] = {}
    raw_rows = 0
    for payload in payloads:
        for item in payload.get("serie", []):
            day = dt.date.fromisoformat(item["fecha"][:10])
            values.setdefault(day, []).append(float(item["valor"]))
            raw_rows += 1

    conflicts = sorted(d for d, v in values.items() if len(set(v)) > 1)
    if conflicts:
        raise ValueError(f"copper: days with two different prices: {conflicts[:5]}")
    weekend = sorted(d for d in values if d.weekday() >= 5)
    if weekend:
        raise ValueError(f"copper: prices dated on a weekend (a date-parsing problem): {weekend[:5]}")
    lo, hi = PRICE_BOUNDS_USD_PER_LB
    out_of_range = sorted((d, v[0]) for d, v in values.items() if not lo < v[0] < hi)
    if out_of_range:
        raise ValueError(f"copper: prices outside {lo}-{hi} USD/lb (wrong unit?): {out_of_range[:5]}")

    days = sorted(values)
    df = pl.DataFrame(
        {"date": days, "price": [values[d][0] for d in days]},
        schema={"date": pl.Date, "price": pl.Float64},
    )
    return df, {"raw_rows": raw_rows, "duplicates_collapsed": raw_rows - len(days)}


def add_log_returns(
    copper: pl.DataFrame, max_sessions: int = MAX_SESSIONS_PER_RETURN
) -> tuple[pl.DataFrame, dict]:
    """Adds ``sessions`` (business days since the previous price) and ``log_return``,
    nulled across a hole of ``max_sessions`` or more business days."""
    copper = copper.sort("date")
    dates = copper["date"].to_numpy().astype("datetime64[D]")
    sessions = np.zeros(len(dates), dtype=np.int64)
    sessions[1:] = np.busday_count(dates[:-1], dates[1:])

    log_price = np.log(copper["price"].to_numpy())
    returns = np.full(len(dates), np.nan)
    returns[1:] = np.diff(log_price)
    holes = np.flatnonzero(sessions >= max_sessions)
    returns[holes] = np.nan

    out = copper.with_columns(
        pl.Series("sessions", sessions),
        pl.Series("log_return", returns).fill_nan(None),
    )
    report = {
        "returns_by_business_days": {int(k): int(v) for k, v in sorted(Counter(sessions[1:]).items())},
        "holes": [
            {
                "from": str(copper["date"][int(i) - 1]),
                "to": str(copper["date"][int(i)]),
                "business_days": int(sessions[i]),
            }
            for i in holes
        ],
    }
    return out, report


def parse_fred_csv(text: str) -> pl.DataFrame:
    """``date, value`` from a FRED ``fredgraph.csv`` download, without the missing days."""
    reader = csv.reader(io.StringIO(text))
    next(reader)  # header: observation_date,<SERIES>
    rows = [(dt.date.fromisoformat(day), float(value)) for day, value in reader if value not in ("", ".")]
    return pl.DataFrame(rows, schema={"date": pl.Date, "value": pl.Float64}, orient="row").sort("date")


def dxy_release_cutoff(origin: dt.date) -> dt.date:
    """Last dollar-index date already published by the end of ``origin``.

    The broad dollar index comes out once a week in the Federal Reserve's H.10 release, on
    Monday afternoon, with data through the previous Friday. A Monday holiday moves the
    release to Tuesday, so a week only counts as published from its Tuesday: the cutoff is
    the Friday before the most recent Tuesday on or before ``origin`` (4 to 10 days back).
    """
    days_since_tuesday = (origin.weekday() - 1) % 7
    return origin - dt.timedelta(days=days_since_tuesday + 4)


def load_market(raw_dir: Path = RAW_DIR, end: dt.date = SAMPLE_END) -> Market:
    payloads = []
    for year in range(COPPER_FIRST_YEAR, end.year + 1):
        path = copper_path(year, raw_dir)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run `python main.py --download` first")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    copper, report = parse_mindicador(payloads)
    copper = copper.filter(pl.col("date") <= end)
    copper, returns_report = add_log_returns(copper)

    def fred(series: str) -> pl.DataFrame:
        path = fred_path(series, raw_dir)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run `python main.py --download` first")
        return parse_fred_csv(path.read_text(encoding="utf-8")).filter(pl.col("date") <= end)

    vix, dxy = fred("VIXCLS"), fred("DTWEXBGS")
    report.update(returns_report)
    report.update(
        {
            "sample_end": str(end),
            "copper_first": str(copper["date"].min()),
            "copper_last": str(copper["date"].max()),
            "copper_days": copper.height,
            "vix_days": vix.height,
            "dxy_days": dxy.height,
        }
    )
    return Market(copper=copper, vix=vix, dxy=dxy, report=report)
