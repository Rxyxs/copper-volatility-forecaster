import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from src.data import add_log_returns, dxy_release_cutoff, load_market, parse_fred_csv, parse_mindicador
from src.sources import copper_path, fred_path


def _payload(*rows):
    return {"codigo": "libra_cobre", "serie": [{"fecha": f"{d}T03:00:00.000Z", "valor": v} for d, v in rows]}


def test_parse_mindicador_collapses_identical_duplicates():
    df, report = parse_mindicador([_payload(("2024-01-02", 3.9), ("2024-01-03", 3.8), ("2024-01-03", 3.8))])
    assert df.height == 2
    assert report["duplicates_collapsed"] == 1
    assert df["date"].to_list() == [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]


def test_parse_mindicador_raises_on_conflicting_duplicates():
    with pytest.raises(ValueError, match="two different prices"):
        parse_mindicador([_payload(("2024-01-03", 3.8), ("2024-01-03", 3.7))])


def test_parse_mindicador_rejects_weekend_dates():
    with pytest.raises(ValueError, match="weekend"):
        parse_mindicador([_payload(("2024-01-06", 3.8))])  # a Saturday


def test_parse_mindicador_rejects_prices_in_another_unit():
    with pytest.raises(ValueError, match="USD/lb"):
        parse_mindicador([_payload(("2024-01-02", 8400.0))])  # USD per tonne, not per pound


def test_add_log_returns_nulls_only_returns_across_a_hole():
    days = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 5), dt.date(2024, 1, 15), dt.date(2024, 1, 16)]
    copper = pl.DataFrame({"date": days, "price": [4.0, 4.1, 4.0, 4.4, 4.3]})
    out, report = add_log_returns(copper, max_sessions=5)

    assert out["sessions"].to_list() == [0, 1, 2, 6, 1]
    returns = out["log_return"].to_list()
    assert returns[0] is None  # nothing before the first price
    assert returns[2] == pytest.approx(np.log(4.0 / 4.1))  # a holiday in between: kept
    assert returns[3] is None  # 6 business days: a hole, not a daily return
    assert returns[4] == pytest.approx(np.log(4.3 / 4.4))
    assert report["holes"] == [{"from": "2024-01-05", "to": "2024-01-15", "business_days": 6}]


def test_parse_fred_csv_drops_missing_days():
    text = "observation_date,VIXCLS\n2024-12-24,15.8\n2024-12-25,.\n2024-12-26,14.7\n"
    df = parse_fred_csv(text)
    assert df["date"].to_list() == [dt.date(2024, 12, 24), dt.date(2024, 12, 26)]
    assert df["value"].to_list() == [15.8, 14.7]


@pytest.mark.parametrize(
    "origin, expected",
    [
        (dt.date(2026, 9, 29), dt.date(2026, 9, 25)),  # Tuesday: last week is out
        (dt.date(2026, 10, 2), dt.date(2026, 9, 25)),  # Friday of the same week
        (dt.date(2026, 9, 28), dt.date(2026, 9, 18)),  # Monday: today's release may slip to Tuesday
        (dt.date(2026, 9, 27), dt.date(2026, 9, 18)),  # Sunday
    ],
)
def test_dxy_release_cutoff_examples(origin, expected):
    assert dxy_release_cutoff(origin) == expected


def test_dxy_release_cutoff_is_always_a_friday_at_least_four_days_back():
    for offset in range(60):
        origin = dt.date(2026, 1, 1) + dt.timedelta(days=offset)
        cutoff = dxy_release_cutoff(origin)
        assert cutoff.weekday() == 4
        assert 4 <= (origin - cutoff).days <= 10


def test_load_market_reads_the_cached_files(tmp_path):
    copper_path(2012, tmp_path).write_text(json.dumps(_payload(("2012-12-27", 3.6), ("2012-12-28", 3.6))))
    copper_path(2013, tmp_path).write_text(json.dumps(_payload(("2013-01-02", 3.7), ("2013-01-03", 3.65))))
    fred_path("VIXCLS", tmp_path).write_text("observation_date,VIXCLS\n2012-12-28,22.7\n2013-01-02,14.7\n")
    fred_path("DTWEXBGS", tmp_path).write_text("observation_date,DTWEXBGS\n2012-12-28,98.0\n2013-01-04,97.5\n")

    market = load_market(tmp_path, end=dt.date(2013, 1, 2))

    assert market.copper["date"].max() == dt.date(2013, 1, 2)  # nothing after the sample end
    assert market.copper.height == 3
    assert market.dxy.height == 1
    assert market.report["copper_days"] == 3


def test_load_market_fails_loudly_without_the_raw_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="--download"):
        load_market(tmp_path, end=dt.date(2013, 1, 2))
