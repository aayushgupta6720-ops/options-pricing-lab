from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from optlab import config
from optlab.market import nse, store
from optlab.market.chain import CHAIN_COLUMNS, build_chain
from optlab.market.forwards import forwards, parity_forward
from optlab.models import black_scholes
from optlab.surface import atm_at_tenor, atm_vol, daily_summary, expiry_metrics, smile_grid, vol_at_delta

FIXTURES = Path(__file__).parent / "fixtures"
TRADE_DATE = date(2026, 10, 1)


@pytest.fixture(scope="module")
def raw():
    """A real bhavcopy subset: NIFTY and RELIANCE, three expiries, plus one BANKNIFTY future."""
    return nse.parse((FIXTURES / "fo_20261001_subset.csv.zip").read_bytes())


def synthetic_day(forward, r, vol_of_strike, expiries_days=(30,), spot=None, n_trades=100):
    """nse.parse()-shaped rows priced exactly by Black-76 with a known forward and smile."""
    rows = []
    for days in expiries_days:
        T = days / 365
        for K in np.arange(0.7, 1.31, 0.02) * forward:
            sigma = vol_of_strike(np.log(K / forward))
            for kind in ("call", "put"):
                price = float(black_scholes.price(forward, K, T, r, sigma, r, kind))
                rows.append(
                    {
                        "trade_date": TRADE_DATE,
                        "underlying": "TEST",
                        "instrument": "option",
                        "expiry": TRADE_DATE + timedelta(days=days),
                        "strike": round(K, 6),
                        "option_type": kind,
                        "close": price,
                        "settle": price,
                        "underlying_price": spot or forward,
                        "open_interest": 1000,
                        "volume": 1000,
                        "n_trades": n_trades,
                        "lot_size": 50,
                    }
                )
    return pd.DataFrame(rows)


# --- nse.py ---------------------------------------------------------------------------------


def test_parse_normalises_the_bhavcopy(raw):
    assert set(raw["underlying"]) == {"NIFTY", "RELIANCE", "BANKNIFTY"}
    assert set(raw["instrument"]) == {"option", "future"}
    options = raw[raw["instrument"] == "option"]
    assert set(options["option_type"]) == {"call", "put"}
    assert raw.loc[raw["instrument"] == "future", "option_type"].isna().all()
    assert raw["trade_date"].eq(TRADE_DATE).all()
    assert date(2026, 10, 27) in set(raw["expiry"])
    nifty = raw[raw["underlying"] == "NIFTY"]
    assert nifty["underlying_price"].iloc[0] == pytest.approx(22421.95)
    assert nifty["lot_size"].iloc[0] == 65


def test_parse_can_filter_underlyings():
    only = nse.parse((FIXTURES / "fo_20261001_subset.csv.zip").read_bytes(), ["RELIANCE"])
    assert set(only["underlying"]) == {"RELIANCE"}


def test_india_vix_reads_the_index_file(monkeypatch):
    monkeypatch.setattr(
        nse, "_get", lambda url, session: (FIXTURES / "ind_close_all_01102026.csv").read_bytes()
    )
    assert nse.india_vix(TRADE_DATE) == pytest.approx(0.1446)


def test_india_vix_is_nan_when_there_is_no_file(monkeypatch):
    def missing(url, session):
        raise nse.NotPublished(url)

    monkeypatch.setattr(nse, "_get", missing)
    assert np.isnan(nse.india_vix(TRADE_DATE))


def test_download_uses_the_cache(tmp_path, monkeypatch):
    calls = []
    body = (FIXTURES / "fo_20261001_subset.csv.zip").read_bytes()
    monkeypatch.setattr(nse, "_get", lambda url, session: calls.append(url) or body)
    assert nse.download(TRADE_DATE, tmp_path) == body
    assert nse.download(TRADE_DATE, tmp_path) == body
    assert len(calls) == 1
    assert "20261001" in calls[0]


# --- forwards.py ----------------------------------------------------------------------------


def test_parity_forward_recovers_a_known_forward():
    day = synthetic_day(forward=20000.0, r=0.06, vol_of_strike=lambda k: 0.15 - 0.2 * k)
    assert parity_forward(day, 30 / 365, 0.06) == pytest.approx(20000.0, rel=1e-9)


def test_parity_forward_matches_the_futures(raw):
    nifty = raw[raw["underlying"] == "NIFTY"]
    fwd = forwards(nifty, config.RISK_FREE_RATE).set_index("expiry")
    for expiry in (date(2026, 10, 27), date(2026, 11, 23)):
        row = fwd.loc[expiry]
        assert row["forward_source"] == "parity"
        assert row["forward"] == pytest.approx(row["future_close"], rel=10e-4)  # within 10 bp


def test_forward_falls_back_to_the_future_when_options_are_illiquid(raw):
    thin = raw[raw["underlying"] == "RELIANCE"].copy()
    thin.loc[thin["instrument"] == "option", "n_trades"] = 0
    fwd = forwards(thin, config.RISK_FREE_RATE).set_index("expiry")
    assert (fwd["forward_source"] == "future").all()
    assert fwd.loc[date(2026, 10, 27), "forward"] == fwd.loc[date(2026, 10, 27), "future_close"]


# --- chain.py -------------------------------------------------------------------------------


def test_build_chain_recovers_the_smile_it_was_priced_with():
    def smile(k):
        return 0.15 - 0.3 * k + 0.5 * k**2

    day = synthetic_day(forward=20000.0, r=config.RISK_FREE_RATE, vol_of_strike=smile, expiries_days=(30, 90))
    chain = build_chain(day, "TEST")
    assert list(chain.columns) == CHAIN_COLUMNS
    np.testing.assert_allclose(chain["forward"], 20000.0, rtol=1e-9)
    np.testing.assert_allclose(chain["iv"], smile(chain["log_moneyness"]), rtol=1e-6)


def test_build_chain_keeps_only_clean_out_of_the_money_quotes(raw):
    chain = build_chain(raw, "NIFTY")
    assert len(chain) > 100
    puts, calls = chain[chain["option_type"] == "put"], chain[chain["option_type"] == "call"]
    assert (puts["strike"] < puts["forward"]).all()
    assert (calls["strike"] >= calls["forward"]).all()
    assert (chain["n_trades"] >= config.MIN_TRADES).all()
    assert (chain["close"] >= config.MIN_PRICE).all()
    assert (chain["T"] * 365 >= config.MIN_DAYS_TO_EXPIRY).all()
    assert chain["iv"].between(0.03, 1.5).all()
    assert ((chain["delta"] > 0) == (chain["option_type"] == "call")).all()


def test_build_chain_for_a_missing_underlying_is_empty(raw):
    assert build_chain(raw, "FINNIFTY").empty


def test_build_chain_drops_untraded_quotes():
    day = synthetic_day(forward=1000.0, r=0.05, vol_of_strike=lambda k: 0.2, n_trades=config.MIN_TRADES - 1)
    chain = build_chain(day, "TEST", r=0.05)
    assert chain.empty and list(chain.columns) == CHAIN_COLUMNS
    assert np.isnan(daily_summary(chain)["atm_iv_30d"])


# --- surface.py -----------------------------------------------------------------------------


def test_flat_vol_gives_flat_metrics():
    day = synthetic_day(forward=5000.0, r=0.05, vol_of_strike=lambda k: 0.22, expiries_days=(10, 40, 100))
    chain = build_chain(day, "TEST", r=0.05)
    metrics = expiry_metrics(chain)
    np.testing.assert_allclose(metrics["atm_iv"], 0.22, rtol=1e-6)
    np.testing.assert_allclose(metrics["skew_25d"], 0.0, atol=1e-6)
    summary = daily_summary(chain)
    for d in config.TENORS:
        assert summary[f"atm_iv_{d}d"] == pytest.approx(0.22, rel=1e-6)


def test_tenor_interpolation_is_linear_in_total_variance():
    metrics = pd.DataFrame({"T": [30 / 365, 90 / 365], "atm_iv": [0.10, 0.20]})
    expected_variance = 0.10**2 * 30 / 365 + (0.20**2 * 90 / 365 - 0.10**2 * 30 / 365) * 0.5
    assert atm_at_tenor(metrics, 60) == pytest.approx(np.sqrt(expected_variance / (60 / 365)))


def test_tenor_extrapolation_is_bounded():
    metrics = pd.DataFrame({"T": [30 / 365, 60 / 365], "atm_iv": [0.15, 0.16]})
    assert atm_at_tenor(metrics, 20) == pytest.approx(0.15)  # 10 days short of the first expiry: flat
    assert np.isnan(atm_at_tenor(metrics, 7))  # 23 days short: no honest value
    assert atm_at_tenor(metrics, 70) == pytest.approx(0.16)
    assert np.isnan(atm_at_tenor(metrics, 90))


def test_real_day_summary_is_sane(raw):
    summary = daily_summary(build_chain(raw, "NIFTY"))
    # India VIX closed at 14.46 that day; 30-day ATM vol should sit just under it.
    assert 0.11 < summary["atm_iv_30d"] < 0.1446
    assert summary["skew_25d_30d"] > 0  # puts richer than calls


def test_smile_grid_is_nan_outside_the_quoted_range():
    day = synthetic_day(forward=1000.0, r=0.05, vol_of_strike=lambda k: 0.2, expiries_days=(30,))
    grid = smile_grid(build_chain(day, "TEST", r=0.05), np.array([-0.05, 0.0, 0.05, 0.39]))
    assert grid.shape == (1, 4)
    np.testing.assert_allclose(grid.iloc[0, :3], 0.2, rtol=1e-6)
    assert np.isnan(grid.iloc[0, 3])


# --- store.py -------------------------------------------------------------------------------


def _summary_row(day, underlying, spot):
    return {"trade_date": day, "underlying": underlying, "spot": spot, "atm_iv_30d": 0.15}


def test_save_is_idempotent_and_reads_back(tmp_path, raw):
    chain = build_chain(raw, "NIFTY")
    for _ in range(2):
        store.save(tmp_path, [chain], [_summary_row(TRADE_DATE, "NIFTY", 22421.95)])
    back = store.read_chain(tmp_path, "NIFTY", TRADE_DATE)
    assert len(back) == len(chain)
    assert back["expiry"].iloc[0] == chain["expiry"].iloc[0]
    assert len(store.read_summary(tmp_path)) == 1
    assert store.ingested(tmp_path) == {(TRADE_DATE, "NIFTY")}
    assert store.read_chain(tmp_path, "NIFTY", date(2026, 10, 2)).empty


def test_realized_vol_matches_the_definition(tmp_path):
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 30)
    spots = 100 * np.exp(np.concatenate([[0], np.cumsum(returns)]))
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(len(spots))]
    store.save(tmp_path, [], [_summary_row(d, "X", s) for d, s in zip(days, spots, strict=True)])
    summary = store.read_summary(tmp_path)
    assert summary["rv_20d"].iloc[:20].isna().all()
    assert summary["rv_20d"].iloc[-1] == pytest.approx(np.std(returns[-20:], ddof=1) * np.sqrt(252))


def test_standardised_smile_grid_lines_up_expiries():
    # Flat vol: one standard deviation out is a different strike for each expiry, same IV.
    day = synthetic_day(forward=1000.0, r=0.05, vol_of_strike=lambda k: 0.2, expiries_days=(10, 90))
    grid = smile_grid(build_chain(day, "TEST", r=0.05), np.array([-1.0, 0.0, 1.0]), standardised=True)
    assert grid.shape == (2, 3)
    np.testing.assert_allclose(grid.to_numpy(), 0.2, rtol=1e-6)


def test_bonus_issue_is_not_a_price_move(tmp_path):
    # 1:1 bonus on day 25: the price halves and the lot doubles. On day 40 NSE resizes the lot
    # with no price change. Realized vol should look like the price never jumped.
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, 59)
    spots = 1000 * np.exp(np.concatenate([[0], np.cumsum(returns)]))
    lots = np.full(60, 250)
    spots[25:] /= 2
    lots[25:] = 500
    lots[40:] = 400
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(60)]
    rows = [
        {"trade_date": d, "underlying": "X", "spot": s, "lot_size": int(n)}
        for d, s, n in zip(days, spots, lots, strict=True)
    ]
    store.save(tmp_path, [], rows)
    rv = store.read_summary(tmp_path)["rv_20d"]
    expected_last = np.std(returns[-20:], ddof=1) * np.sqrt(252)
    assert rv.iloc[-1] == pytest.approx(expected_last)
    assert rv.max() < 0.3  # unadjusted, the halving alone would push it past 300%
    adjusted = store.adjusted_log_returns(pd.Series(spots), pd.Series(lots))
    np.testing.assert_allclose(adjusted.iloc[1:], returns, atol=1e-12)


def _quotes(k, iv, delta=None, T=0.25):
    k = np.asarray(k, dtype=float)
    return pd.DataFrame(
        {
            "log_moneyness": k,
            "iv": iv,
            "T": T,
            "delta": delta if delta is not None else np.where(k < 0, -0.3, 0.3),
            "option_type": np.where(k < 0, "put", "call"),
        }
    )


def test_atm_is_not_interpolated_across_a_wide_gap():
    # RELIANCE, 9 Apr 2025, June expiry: nearest put at k=-0.183 (35.2%), nearest call at +0.079
    # (26.1%), ~1.85 SD apart. Interpolating gave 28.9% against ~24.5% on the days around it.
    wide = _quotes([-0.25, -0.183, 0.079], [0.38, 0.352, 0.261], T=0.21)
    assert np.isnan(atm_vol(wide))
    close = _quotes([-0.04, -0.01, 0.02, 0.05], [0.16, 0.15, 0.145, 0.15], T=0.21)
    assert atm_vol(close) == pytest.approx(0.15 + (0.145 - 0.15) * (0.01 / 0.03))


def test_25_delta_vol_ignores_how_noisy_deltas_sort():
    k = np.array([-0.20, -0.15, -0.12, -0.10, -0.05])
    iv = np.array([0.22, 0.20, 0.19, 0.185, 0.17])
    # The -0.12 point's delta is out of order (noisy price): -0.24 sits between its neighbours'
    # strikes but on the "wrong" side of -0.25.
    noisy = np.array([-0.08, -0.18, -0.24, -0.23, -0.40])
    value = vol_at_delta(_quotes(k, iv, noisy), -0.25)
    assert 0.17 < value < 0.19
    assert value == vol_at_delta(_quotes(k[::-1], iv[::-1], noisy[::-1]), -0.25)


def test_resaving_a_day_with_no_quotes_removes_the_old_ones(tmp_path, raw):
    store.save(tmp_path, [build_chain(raw, "NIFTY")], [_summary_row(TRADE_DATE, "NIFTY", 22421.95)])
    assert len(store.read_chain(tmp_path, "NIFTY", TRADE_DATE)) > 100
    store.save(tmp_path, [pd.DataFrame(columns=CHAIN_COLUMNS)], [_summary_row(TRADE_DATE, "NIFTY", 22421.95)])
    assert store.read_chain(tmp_path, "NIFTY", TRADE_DATE).empty
    assert not store.chain_path(tmp_path, "NIFTY", "2026-10").exists()


class FakeSession:
    """Answers each GET with the next status code in the list."""

    def __init__(self, *statuses, content=b"ok"):
        self.statuses, self.content, self.calls = list(statuses), content, 0

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        response = requests.Response()
        response.status_code, response._content, response.url = self.statuses.pop(0), self.content, url
        return response


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(nse.time, "sleep", lambda seconds: None)


def test_get_treats_404_as_not_published(no_sleep):
    with pytest.raises(nse.NotPublished):
        nse._get("https://example/x", FakeSession(404))


def test_get_fails_fast_on_403(no_sleep):
    session = FakeSession(403)
    with pytest.raises(requests.HTTPError):
        nse._get("https://example/x", session)
    assert session.calls == 1  # blocked: retrying won't help


def test_get_retries_rate_limits_and_server_errors(no_sleep):
    session = FakeSession(429, 503, 200)
    assert nse._get("https://example/x", session) == b"ok"
    assert session.calls == 3
    with pytest.raises(requests.HTTPError):
        nse._get("https://example/x", FakeSession(500, 502, 503))


def test_download_refuses_a_non_zip_and_caches_nothing(tmp_path, no_sleep):
    session = FakeSession(200, content=b"<html>Access denied</html>")
    with pytest.raises(nse.BadResponse):
        nse.download(TRADE_DATE, tmp_path, session)
    assert not list(tmp_path.iterdir())


def test_india_vix_is_cached(tmp_path, monkeypatch):
    calls = []
    body = (FIXTURES / "ind_close_all_01102026.csv").read_bytes()
    monkeypatch.setattr(nse, "_get", lambda url, session: calls.append(url) or body)
    assert nse.india_vix(TRADE_DATE, cache_dir=tmp_path) == pytest.approx(0.1446)
    assert nse.india_vix(TRADE_DATE, cache_dir=tmp_path) == pytest.approx(0.1446)
    assert len(calls) == 1
