"""Downloads the raw public series the project runs on and caches them in ``data/raw/``.

Two free sources, no API keys:

- **mindicador.cl** -- daily copper price (``libra_cobre``, US dollars per pound), one JSON
  file per year. The series starts on 2012-10-05; earlier years come back empty.
- **FRED** (Federal Reserve Bank of St. Louis) -- the CBOE VIX (``VIXCLS``) and the Federal
  Reserve's nominal broad U.S. dollar index (``DTWEXBGS``), as CSV.

Closed years of the copper series are downloaded once. The current year and the two FRED
files are downloaded again only with ``refresh=True``. Nothing here cleans or reshapes the
data: that is :mod:`src.data`, which works offline on these files so it can be unit-tested.
A failed or empty download raises -- there is no synthetic fallback.
"""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MINDICADOR_URL = "https://mindicador.cl/api/libra_cobre/{year}"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
COPPER_FIRST_YEAR = 2012
FRED_SERIES = ("VIXCLS", "DTWEXBGS")
USER_AGENT = "copper-volatility-forecaster (github.com/Rxyxs)"
TIMEOUT_S = 60
PAUSE_S = 0.3  # free public API: one request at a time


def copper_path(year: int, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"libra_cobre_{year}.json"


def fred_path(series: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"fred_{series}.csv"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return response.read()


def download_all(raw_dir: Path = RAW_DIR, refresh: bool = False, today: dt.date | None = None) -> list[Path]:
    """Downloads every raw file that is missing (plus the ones that can still change, if
    ``refresh``) and returns the paths written."""
    today = today or dt.date.today()
    raw_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for year in range(COPPER_FIRST_YEAR, today.year + 1):
        path = copper_path(year, raw_dir)
        if path.exists() and not (refresh and year == today.year):
            continue
        payload = json.loads(_get(MINDICADOR_URL.format(year=year)))
        if not payload.get("serie") and year < today.year:
            raise RuntimeError(f"mindicador returned no copper prices for {year}")
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        written.append(path)
        time.sleep(PAUSE_S)

    for series in FRED_SERIES:
        path = fred_path(series, raw_dir)
        if path.exists() and not refresh:
            continue
        content = _get(FRED_URL.format(series=series))
        if not content.startswith(b"observation_date") and not content.startswith(b"DATE"):
            raise RuntimeError(f"FRED did not return a CSV for {series}")
        path.write_bytes(content)
        written.append(path)

    return written
