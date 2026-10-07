"""The latest index level, during market hours or since the last close: NSE's own, else Yahoo's.

NSE's website API gives index levels (one request covers NIFTY and BANKNIFTY), but its bot
protection turns some clients away; Yahoo Finance's chart API is the fallback. Yahoo rate-limits
quickly (HTTP 429). So a source that refuses or fails is left alone for a while, and callers cache
what comes back (the app asks at most once a minute per underlying).

Stocks aren't covered: NSE refuses scripts its single-stock quotes, and Yahoo answered the app's
server (on Render, in the US) with 429 from its first request, so RELIANCE stays at its last close.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import monotonic
from urllib.parse import quote as url_quote

import requests

from optlab.market.nse import HEADERS

IST = timezone(timedelta(hours=5, minutes=30))  # India has no daylight saving
NSE_URL = "https://www.nseindia.com/api/allIndices"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"
NSE_INDICES = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK"}
YAHOO_SYMBOLS = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
COVERED = frozenset(NSE_INDICES) | frozenset(YAHOO_SYMBOLS)
TIMEOUT = 4  # seconds: a page waits this long for a source when the cache is empty
REST = 600  # seconds to leave a source alone after it fails
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Quote:
    underlying: str
    price: float
    time: datetime  # when the source last updated the price, in IST
    source: str


class NotCovered(Exception):
    """The source doesn't price this underlying (not a failure of the source)."""


def _json(url: str, referer: str | None = None) -> dict:
    headers = {**HEADERS, "Accept": "application/json"} | ({"Referer": referer} if referer else {})
    response = requests.get(url, headers=headers, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def from_nse(underlying: str) -> Quote:
    if underlying not in NSE_INDICES:
        raise NotCovered(f"NSE's index feed has no {underlying}")
    data = _json(NSE_URL, referer="https://www.nseindia.com/")
    row = next((r for r in data["data"] if r.get("index") == NSE_INDICES[underlying]), None)
    if row is None:
        raise KeyError(f"{NSE_INDICES[underlying]} missing from NSE's index feed")
    time = datetime.strptime(data["timestamp"], "%d-%b-%Y %H:%M").replace(tzinfo=IST)
    return Quote(underlying, float(row["last"]), time, "NSE")


def from_yahoo(underlying: str) -> Quote:
    if underlying not in YAHOO_SYMBOLS:
        raise NotCovered(f"no Yahoo symbol for {underlying}")
    meta = _json(YAHOO_URL.format(symbol=url_quote(YAHOO_SYMBOLS[underlying])))["chart"]["result"][0]["meta"]
    time = datetime.fromtimestamp(meta["regularMarketTime"], IST)
    return Quote(underlying, float(meta["regularMarketPrice"]), time, "Yahoo Finance")


SOURCES = {"NSE": from_nse, "Yahoo Finance": from_yahoo}
_resting: dict[str, float] = {}  # source -> monotonic time it may be asked again


def latest(underlying: str) -> Quote | None:
    """The first price a source gives, trying them in order; None if none does."""
    for name, fetch in SOURCES.items():
        if _resting.get(name, 0.0) > monotonic():
            continue
        try:
            quote = fetch(underlying)
        except NotCovered:
            continue
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
            # refused (403, 429), down, or answered with something else (an HTML error page)
            _resting[name] = monotonic() + REST
            log.warning(
                "live price: %s failed for %s (%s); resting it for %ds", name, underlying, error, REST
            )
            continue
        if quote.price > 0:
            return quote
    return None
