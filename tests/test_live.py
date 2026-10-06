"""Live prices: NSE's index feed first, Yahoo's chart API as the fallback. No network: requests.get
is replaced by canned answers."""

from datetime import datetime

import pytest
import requests

from optlab.market import live

WHEN = datetime(2026, 10, 7, 11, 42, tzinfo=live.IST)
NSE = {
    "timestamp": "07-Oct-2026 11:42",
    "data": [{"index": "NIFTY 50", "last": 22850.5}, {"index": "NIFTY BANK", "last": 55300.0}],
}
YAHOO = {
    "chart": {"result": [{"meta": {"regularMarketPrice": 1350.2, "regularMarketTime": WHEN.timestamp()}}]}
}


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if isinstance(self.body, str):  # e.g. an HTML "Access Denied" page
            raise requests.exceptions.JSONDecodeError("Expecting value", self.body, 0)
        return self.body


@pytest.fixture
def web(monkeypatch):
    """routes maps a host to its answer; calls records every URL asked for."""
    routes, calls, now = {}, [], [1000.0]

    def get(url, headers=None, timeout=None):
        calls.append(url)
        for host, answer in routes.items():
            if host in url:
                return answer
        raise requests.ConnectionError(url)

    monkeypatch.setattr(live.requests, "get", get)
    monkeypatch.setattr(live, "_resting", {})
    monkeypatch.setattr(live, "monotonic", lambda: now[0])
    return routes, calls, now


def test_indices_come_from_nse_in_ist(web):
    routes, calls, _ = web
    routes["nseindia.com"] = Response(NSE)
    quote = live.latest("NIFTY")
    assert (quote.price, quote.time, quote.source) == (22850.5, WHEN, "NSE")
    assert live.latest("BANKNIFTY").price == 55300.0
    assert not any("yahoo" in url for url in calls)


def test_stocks_come_from_yahoo_without_asking_nse(web):
    routes, calls, _ = web
    routes["yahoo.com"] = Response(YAHOO)
    quote = live.latest("RELIANCE")
    assert (quote.price, quote.time, quote.source) == (1350.2, WHEN, "Yahoo Finance")
    assert "RELIANCE.NS" in calls[0] and len(calls) == 1
    assert "NSE" not in live._resting  # not covering a stock isn't a failure


@pytest.mark.parametrize(
    "refusal", [Response("<HTML>Access Denied</HTML>", 403), Response("<HTML>busy</HTML>")]
)
def test_a_refusing_source_falls_back_and_rests(web, refusal):
    routes, calls, now = web
    routes["nseindia.com"], routes["yahoo.com"] = refusal, Response(YAHOO)
    assert live.latest("NIFTY").source == "Yahoo Finance"
    calls.clear()
    assert live.latest("NIFTY").source == "Yahoo Finance"
    assert not any("nseindia" in url for url in calls)  # resting
    now[0] += live.REST + 1
    routes["nseindia.com"] = Response(NSE)
    assert live.latest("NIFTY").source == "NSE"


def test_nothing_answers(web):
    routes, _, _ = web
    routes["yahoo.com"] = Response("Too Many Requests", 429)
    assert live.latest("NIFTY") is None  # NSE unreachable, Yahoo rate-limited
    assert live.latest("RELIANCE") is None
    assert live.latest("FINNIFTY") is None  # no source covers it
