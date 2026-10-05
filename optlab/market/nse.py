"""NSE end-of-day F&O data (the "bhavcopy"), downloaded and normalised.

One zip per trading day, published in the evening IST at a fixed URL. A 404 means there is no file
for that date (weekend, holiday, or not published yet), which callers treat as "skip", not as an
error. NSE's archive host rejects requests without a browser-like User-Agent.

Normalised columns:
    trade_date, underlying, instrument ("option" | "future"), expiry, strike, option_type
    ("call" | "put" | None), close, settle, underlying_price, open_interest, volume (contracts),
    n_trades, lot_size
"""

import io
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
INDICES_URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{day:%d%m%Y}.csv"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

INSTRUMENTS = {"IDO": "option", "STO": "option", "IDF": "future", "STF": "future"}
OPTION_TYPES = {"CE": "call", "PE": "put"}


class NotPublished(Exception):
    """No file for this date: a weekend, a market holiday, or not out yet."""


class BadResponse(Exception):
    """NSE answered 200 but not with the file (e.g. an HTML error page)."""


def _get(url: str, session: requests.Session | None, retries: int = 3, timeout: float = 30) -> bytes:
    http = session or requests
    for attempt in range(retries):
        try:
            response = http.get(url, headers=HEADERS, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries - 1:
                raise
        else:
            if response.status_code == 404:
                raise NotPublished(url)
            if response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()
                return response.content
            if attempt == retries - 1:
                response.raise_for_status()
        time.sleep(2 * 3**attempt)  # 2s, 6s, ...
    raise AssertionError("unreachable")


def _bhavcopy_cache(cache_dir: Path, day: date) -> Path:
    return cache_dir / f"fo_{day:%Y%m%d}.csv.zip"


def _vix_cache(cache_dir: Path, day: date) -> Path:
    return cache_dir / f"ind_close_all_{day:%Y%m%d}.csv"


def is_cached(day: date, cache_dir: Path | None) -> bool:
    """Whether both of the day's files are already cached (so loading it makes no requests)."""
    return (
        bool(cache_dir) and _bhavcopy_cache(cache_dir, day).exists() and _vix_cache(cache_dir, day).exists()
    )


def download(day: date, cache_dir: Path | None = None, session: requests.Session | None = None) -> bytes:
    """The raw bhavcopy zip for one day, from the cache when it has it."""
    cached = _bhavcopy_cache(cache_dir, day) if cache_dir else None
    if cached and cached.exists():
        return cached.read_bytes()
    raw = _get(URL.format(day=day), session)
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise BadResponse(f"bhavcopy for {day} is not a zip ({len(raw)} bytes): {raw[:80]!r}")
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(raw)
    return raw


def parse(raw_zip: bytes, underlyings=None) -> pd.DataFrame:
    """Options and futures rows from a bhavcopy zip, optionally for some underlyings only."""
    df = pd.read_csv(io.BytesIO(raw_zip), compression="zip")
    df = df[df["FinInstrmTp"].isin(INSTRUMENTS)]
    if underlyings is not None:
        df = df[df["TckrSymb"].isin(underlyings)]
    expiry = df["FininstrmActlXpryDt"].fillna(df["XpryDt"])
    out = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(df["TradDt"]).dt.date,
            "underlying": df["TckrSymb"].astype(str),
            "instrument": df["FinInstrmTp"].map(INSTRUMENTS),
            "expiry": pd.to_datetime(expiry).dt.date,
            "strike": pd.to_numeric(df["StrkPric"], errors="coerce"),
            "option_type": df["OptnTp"].map(OPTION_TYPES),
            "close": pd.to_numeric(df["ClsPric"], errors="coerce"),
            "settle": pd.to_numeric(df["SttlmPric"], errors="coerce"),
            "underlying_price": pd.to_numeric(df["UndrlygPric"], errors="coerce"),
            "open_interest": pd.to_numeric(df["OpnIntrst"], errors="coerce").fillna(0).astype("int64"),
            "volume": pd.to_numeric(df["TtlTradgVol"], errors="coerce").fillna(0).astype("int64"),
            "n_trades": pd.to_numeric(df["TtlNbOfTxsExctd"], errors="coerce").fillna(0).astype("int64"),
            "lot_size": pd.to_numeric(df["NewBrdLotQty"], errors="coerce").fillna(0).astype("int64"),
        }
    )
    return out.reset_index(drop=True)


def load(day: date, underlyings=None, cache_dir: Path | None = None, session=None) -> pd.DataFrame:
    return parse(download(day, cache_dir, session), underlyings)


def india_vix(day: date, session: requests.Session | None = None, cache_dir: Path | None = None) -> float:
    """India VIX close for the day as a decimal (0.1446 for 14.46), or NaN if NSE has no file."""
    cached = _vix_cache(cache_dir, day) if cache_dir else None
    if cached and cached.exists():
        raw = cached.read_bytes()
    else:
        try:
            raw = _get(INDICES_URL.format(day=day), session)
        except NotPublished:
            return float("nan")
        if not raw.startswith(b"Index Name"):
            raise BadResponse(f"index file for {day} isn't the expected CSV: {raw[:80]!r}")
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(raw)
    indices = pd.read_csv(io.BytesIO(raw))
    vix = indices[indices["Index Name"].str.strip().str.upper() == "INDIA VIX"]
    return float(vix["Closing Index Value"].iloc[0]) / 100 if len(vix) else float("nan")
