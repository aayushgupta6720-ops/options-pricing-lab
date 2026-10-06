"""Where the app gets market data: a local dataset if there is one, else the market-data branch.

Locally, `data/` is a worktree of the market-data branch; when it holds a summary, it's the only
source (a file missing there is missing, not fetched from GitHub). On Render there's no such
folder, so files come from GitHub's raw file host. Both locations can be overridden with
MARKET_DATA_DIR / MARKET_DATA_URL.

Caching: the summary is cached for an hour, and each chain file is cached under the dataset
version (the latest trade date in that summary). So once the summary shows a new day, chains are
fetched afresh rather than served from an older cache entry, and the request carries the version
as a query string so GitHub's CDN doesn't hand back a stale copy either.

The dataset is end-of-day. Live prices (optlab/market/live.py) are cached for a minute and only
used when they're later than its last close; LIVE_PRICES=0 turns them off.
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
from optlab.market import live
from optlab.market.store import (
    HESTON_FITS,
    ROUGH_EXPIRIES,
    ROUGH_FITS,
    ROUGH_QUOTES,
    SABR_FITS,
    SUMMARY,
    as_dates,
)
from optlab.models.heston import HestonParams
from optlab.models.rough_bergomi import ForwardVariance, RoughBergomiParams

LOCAL_DIR = Path(os.environ.get("MARKET_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
REMOTE_URL = os.environ.get(
    "MARKET_DATA_URL", "https://raw.githubusercontent.com/aayushgupta6720-ops/options-pricing-lab/market-data"
)
UNDERLYINGS = config.UNDERLYINGS
LIVE_PRICES = os.environ.get("LIVE_PRICES", "1") != "0"
MAX_LIVE_MOVE = 0.2  # NSE halts index trading at a 20% move, so anything further is a bad price


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


@st.cache_resource(ttl=6 * 3600, show_spinner=False, max_entries=8)
def _fits(relative: str, version: str) -> pd.DataFrame:
    """One shared copy per table and dataset version (cache_resource doesn't copy on every access the
    way cache_data does; the quote-level table is ~20 MB). Callers filter it and never modify it."""
    df = _read(relative, version)
    if df is None:
        return pd.DataFrame(columns=["trade_date", "underlying"])
    df = as_dates(df, "trade_date", "expiry")
    for column in ("underlying", "option_type"):
        if column in df:
            df[column] = df[column].astype("category")
    return df


def heston_fits(underlying: str) -> pd.DataFrame:
    """Every successful daily Heston fit for one underlying, oldest first."""
    fits = _fits(HESTON_FITS, version())
    if fits.empty:
        return fits
    fits = fits[(fits["underlying"] == underlying) & fits["fitted"].astype(bool)]
    return fits.sort_values("trade_date").reset_index(drop=True)


def heston_params(row) -> HestonParams:
    return HestonParams(row["v0"], row["kappa"], row["theta"], row["xi"], row["rho"])


def sabr_fits(underlying: str, day: date) -> pd.DataFrame:
    fits = _fits(SABR_FITS, version())
    if fits.empty:
        return fits
    return fits[(fits["underlying"] == underlying) & (fits["trade_date"] == day)].reset_index(drop=True)


def rough_fits(underlying: str = "NIFTY") -> pd.DataFrame:
    """Every successful daily rough Bergomi fit, oldest first."""
    fits = _fits(ROUGH_FITS, version())
    if fits.empty:
        return fits
    fits = fits[(fits["underlying"] == underlying) & fits["fitted"].astype(bool)]
    return fits.sort_values("trade_date").reset_index(drop=True)


def rough_model(row) -> tuple[RoughBergomiParams, ForwardVariance]:
    xi = ForwardVariance(np.asarray(row["xi_times"], dtype=float), np.asarray(row["xi_values"], dtype=float))
    return RoughBergomiParams(row["H"], row["eta"], row["rho"]), xi


def rough_expiries(underlying: str = "NIFTY") -> pd.DataFrame:
    table = _fits(ROUGH_EXPIRIES, version())
    return table[table["underlying"] == underlying].reset_index(drop=True) if len(table) else table


def rough_quotes(underlying: str, day: date) -> pd.DataFrame:
    table = _fits(ROUGH_QUOTES, version())
    if table.empty:
        return table
    return table[(table["underlying"] == underlying) & (table["trade_date"] == day)].reset_index(drop=True)


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


@st.cache_data(ttl=60, show_spinner=False)
def _live_quote(underlying: str) -> live.Quote | None:
    return live.latest(underlying)


def live_spot(underlying: str, close_day: date, close: float) -> live.Quote | None:
    """The live price, if it's later than the dataset's last close: during the session, and after it
    until that evening's data lands. None otherwise, when no source answers, or when the price is
    implausibly far from the close."""
    if not LIVE_PRICES:
        return None
    quote = _live_quote(underlying)
    if quote is None or quote.time.date() <= close_day or abs(np.log(quote.price / close)) > MAX_LIVE_MOVE:
        return None
    return quote


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
    """Inputs for the at-the-money option on the first expiry at least `min_days` out.

    At the live price when there's one later than the dataset's last close (live_spot), else at
    that close, with the option's closing price to compare against. The dividend yield is backed out
    of the close's forward: F = S exp((r - q) T). With a live price, the strike is the listed one
    nearest the live forward and the vol is the close's smile at that moneyness, so the at-the-money
    vol moves along with the market (sticky moneyness).
    """
    day, ch = latest_chain(underlying)
    if ch.empty:
        return None
    quote = live_spot(underlying, day, float(ch["spot"].iloc[0]))
    today = quote.time.date() if quote else day
    ch = ch[ch["expiry"].map(lambda e: (e - today).days) >= min_days]
    if ch.empty:
        return None
    ch = ch[ch["expiry"] == ch["expiry"].min()]
    expiry = ch["expiry"].iloc[0]
    S, F, T, r = (float(ch[c].iloc[0]) for c in ("spot", "forward", "T", "r"))
    q = r - np.log(F / S) / T
    common = {"r": r, "q": q, "as_of": day, "lot_size": int(ch["lot_size"].iloc[0])}
    if quote is None:
        atm = ch.iloc[ch["log_moneyness"].abs().argsort().iloc[0]]
        label = f"{underlying} {atm['strike']:,.0f} {atm['option_type']}, {expiry:%d %b %Y}"
        return common | {
            "label": label,
            "S": S,
            "K": float(atm["strike"]),
            "days": round(T * 365),
            "sigma": float(atm["iv"]),
            "kind": str(atm["option_type"]),
            "market_price": float(atm["close"]),
            "note": f"Loaded {label} (close ₹{atm['close']:,.2f} on {day:%d %b %Y}).",
        }
    days = (expiry - today).days
    F = quote.price * np.exp((r - q) * days / 365)
    strikes = np.sort(ch["strike"].unique())
    K = float(strikes[np.abs(strikes - F).argmin()])
    kind = "call" if K >= F else "put"
    smile = ch.sort_values("log_moneyness")
    label = f"{underlying} {K:,.0f} {kind}, {expiry:%d %b %Y}"
    return common | {
        "label": label,
        "S": quote.price,
        "K": K,
        "days": days,
        "sigma": float(np.interp(np.log(K / F), smile["log_moneyness"], smile["iv"])),
        "kind": kind,
        "live": quote,
        "note": f"Loaded {label} at {underlying} {quote.price:,.2f}, live from {quote.source} at "
        f"{quote.time:%H:%M} IST on {quote.time:%d %b}. Vol from the {day:%d %b} close's smile.",
    }
