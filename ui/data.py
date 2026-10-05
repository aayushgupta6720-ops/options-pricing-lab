"""Where the app gets market data: a local dataset if there is one, else the market-data branch.

Locally, `data/` is a worktree of the market-data branch; when it holds a summary, it's the only
source (a file missing there is missing, not fetched from GitHub). On Render there's no such
folder, so files come from GitHub's raw file host. Both locations can be overridden with
MARKET_DATA_DIR / MARKET_DATA_URL.

Caching: the summary is cached for an hour, and each chain file is cached under the dataset
version (the latest trade date in that summary). So once the summary shows a new day, chains are
fetched afresh rather than served from an older cache entry, and the request carries the version
as a query string so GitHub's CDN doesn't hand back a stale copy either.
"""

import io
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

from optlab import config
from optlab.market.store import SUMMARY, as_dates

LOCAL_DIR = Path(os.environ.get("MARKET_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
REMOTE_URL = os.environ.get(
    "MARKET_DATA_URL", "https://raw.githubusercontent.com/aayushgupta6720-ops/options-pricing-lab/market-data"
)
UNDERLYINGS = config.UNDERLYINGS


class DataUnavailable(Exception):
    """The dataset couldn't be reached (network down, GitHub unavailable)."""


def _read(relative: str, version: str | None = None) -> pd.DataFrame | None:
    if (LOCAL_DIR / SUMMARY).exists():
        path = LOCAL_DIR / relative
        return pd.read_parquet(path) if path.exists() else None
    url = f"{REMOTE_URL}/{relative}" + (f"?v={version}" if version else "")
    try:
        response = requests.get(url, timeout=15)
    except requests.RequestException as error:
        raise DataUnavailable(f"couldn't reach {REMOTE_URL}") from error
    if response.status_code == 404:
        return None
    if not response.ok:
        raise DataUnavailable(f"{relative}: HTTP {response.status_code}")
    return pd.read_parquet(io.BytesIO(response.content))


@st.cache_data(ttl=3600, show_spinner=False)
def summary() -> pd.DataFrame:
    df = _read(SUMMARY)
    if df is None:
        return pd.DataFrame(columns=["trade_date", "underlying", "spot"])
    return as_dates(df, "trade_date").sort_values(["underlying", "trade_date"]).reset_index(drop=True)


def version() -> str:
    s = summary()
    return str(s["trade_date"].max()) if len(s) else "empty"


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=24)
def _chain_month(underlying: str, month: str, version: str) -> pd.DataFrame:
    df = _read(f"chains/{underlying}/{month}.parquet", version)
    return pd.DataFrame() if df is None else as_dates(df, "trade_date", "expiry")


def chain(underlying: str, day: date) -> pd.DataFrame:
    month = _chain_month(underlying, f"{day:%Y-%m}", version())
    if month.empty:
        return month
    return month[month["trade_date"] == day].reset_index(drop=True)


def rows_for(underlying: str) -> pd.DataFrame:
    """The summary rows for one underlying on days it traded, oldest first."""
    s = summary()
    return s[(s["underlying"] == underlying) & s["spot"].notna()].reset_index(drop=True)


def days_for(underlying: str) -> list[date]:
    return sorted(rows_for(underlying)["trade_date"], reverse=True)


def latest_chain(underlying: str, lookback: int = 5) -> tuple[date | None, pd.DataFrame]:
    """The most recent day with a non-empty chain (normally the latest day), and that chain."""
    for day in days_for(underlying)[:lookback]:
        ch = chain(underlying, day)
        if len(ch):
            return day, ch
    return None, pd.DataFrame()


def load_or_stop(fn, *args):
    """Call a data function; if the dataset can't be reached, say so on the page and stop."""
    try:
        return fn(*args)
    except DataUnavailable as error:
        st.error(
            f"The market data can't be reached right now ({error}). Try again in a minute; "
            "the Pricing lab pages work without it."
        )
        st.stop()


def market_option(underlying: str = "NIFTY", min_days: int = 20) -> dict | None:
    """Inputs for the at-the-money option on the first expiry at least `min_days` out, latest day.

    The dividend yield is backed out of the forward: F = S exp((r - q) T).
    """
    day, ch = latest_chain(underlying)
    ch = ch[ch["T"] * 365 >= min_days] if len(ch) else ch
    if ch.empty:
        return None
    ch = ch[ch["expiry"] == ch["expiry"].min()]
    atm = ch.iloc[(ch["log_moneyness"]).abs().argsort().iloc[0]]
    S, F, T, r = float(atm["spot"]), float(atm["forward"]), float(atm["T"]), float(atm["r"])
    return {
        "label": f"{underlying} {atm['strike']:,.0f} {atm['option_type']}, {atm['expiry']:%d %b %Y}",
        "S": S,
        "K": float(atm["strike"]),
        "days": round(T * 365),
        "sigma": float(atm["iv"]),
        "r": r,
        "q": r - np.log(F / S) / T,
        "kind": str(atm["option_type"]),
        "market_price": float(atm["close"]),
        "as_of": day,
        "lot_size": int(atm["lot_size"]),
    }
