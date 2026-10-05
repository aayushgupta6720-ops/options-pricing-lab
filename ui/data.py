"""Where the app gets market data: a local dataset if there is one, else the market-data branch.

Locally, `data/` is a worktree of the market-data branch. On Render there's no such folder, so
files are fetched from GitHub's raw file host and cached for an hour (the dataset changes once a
day). Both locations can be overridden with MARKET_DATA_DIR / MARKET_DATA_URL.
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


def _read(relative: str) -> pd.DataFrame | None:
    local = LOCAL_DIR / relative
    if local.exists():
        return pd.read_parquet(local)
    response = requests.get(f"{REMOTE_URL}/{relative}", timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return pd.read_parquet(io.BytesIO(response.content))


@st.cache_data(ttl=3600, show_spinner=False)
def summary() -> pd.DataFrame:
    df = _read(SUMMARY)
    if df is None:
        return pd.DataFrame(columns=["trade_date", "underlying"])
    return as_dates(df, "trade_date").sort_values(["underlying", "trade_date"]).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=24)
def _chain_month(underlying: str, month: str) -> pd.DataFrame:
    df = _read(f"chains/{underlying}/{month}.parquet")
    return pd.DataFrame() if df is None else as_dates(df, "trade_date", "expiry")


def chain(underlying: str, day: date) -> pd.DataFrame:
    month = _chain_month(underlying, f"{day:%Y-%m}")
    if month.empty:
        return month
    return month[month["trade_date"] == day].reset_index(drop=True)


def days_for(underlying: str) -> list[date]:
    s = summary()
    return sorted(s.loc[s["underlying"] == underlying, "trade_date"], reverse=True)


def market_option(underlying: str = "NIFTY", min_days: int = 20) -> dict | None:
    """Inputs for the at-the-money call on the first expiry at least `min_days` out, latest day.

    The dividend yield is backed out of the forward: F = S exp((r - q) T).
    """
    days = days_for(underlying)
    if not days:
        return None
    ch = chain(underlying, days[0])
    ch = ch[ch["T"] * 365 >= min_days]
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
        "as_of": days[0],
        "lot_size": int(atm["lot_size"]),
    }


def lot_size(underlying: str) -> int | None:
    days = days_for(underlying)
    if not days:
        return None
    ch = chain(underlying, days[0])
    return int(ch["lot_size"].iloc[0]) if len(ch) else None


UNDERLYINGS = config.UNDERLYINGS
