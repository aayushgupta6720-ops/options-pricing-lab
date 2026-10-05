from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from optlab import config
from optlab.market import nse, store
from optlab.market.chain import CHAIN_COLUMNS, build_chain
from optlab.market.forwards import forwards, parity_forward
from optlab.models import black_scholes
from optlab.surface import atm_at_tenor, daily_summary, expiry_metrics, smile_grid

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
    monkeypatch.setattr(nse, "_get", lambda url, session: calls.append(url) or b"zip-bytes")
    assert nse.download(TRADE_DATE, tmp_path) == b"zip-bytes"
    assert nse.download(TRADE_DATE, tmp_path) == b"zip-bytes"
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
    assert store.ingested_days(tmp_path) == {TRADE_DATE}
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
