from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from optlab import variance
from optlab.market import nse
from optlab.market.chain import build_chain
from optlab.models import black_scholes
from optlab.surface import expiry_metrics

FIXTURES = Path(__file__).parent / "fixtures"
DAY, F, R = date(2026, 10, 1), 22500.0, 0.055


def flat_chain(vol_of_days: dict, width_sd=6.0, n=161) -> pd.DataFrame:
    rows = []
    for days, vol in vol_of_days.items():
        T = days / 365
        K = F * np.exp(np.linspace(-width_sd, width_sd, n) * vol * np.sqrt(T))
        kind = np.where(K < F, "put", "call")
        rows.append(
            pd.DataFrame(
                {
                    "expiry": DAY + timedelta(days=days),
                    "T": T,
                    "r": R,
                    "strike": K,
                    "option_type": kind,
                    "close": black_scholes.price(F, K, T, R, vol, R, kind),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_replicates_black_scholes_variance():
    for days, vol in ((7, 0.12), (30, 0.18), (180, 0.25)):
        quotes = flat_chain({days: vol})
        assert np.sqrt(variance.variance_swap(quotes)) == pytest.approx(vol, rel=0.005), days


def test_needs_both_sides_and_enough_strikes():
    quotes = flat_chain({30: 0.2})
    assert np.isnan(variance.variance_swap(quotes[quotes["option_type"] == "put"]))
    assert np.isnan(variance.variance_swap(quotes.iloc[::40]))


def test_forward_variance_reproduces_the_term_structure():
    chain = flat_chain({7: 0.20, 30: 0.15, 90: 0.18})
    table = variance.term_structure(chain)
    curve = variance.forward_variance(chain)
    for T, total in zip(table["T"], table["total"], strict=True):
        assert curve.total(T) == pytest.approx(total, rel=1e-9)
    # total variance 0.2^2*7/365 then 0.15^2*30/365: the second step adds only a little variance
    assert np.sqrt(curve.values[1]) < 0.15 < np.sqrt(curve.values[2])


def test_falling_total_variance_is_mademonotone():
    np.testing.assert_allclose(
        variance.monotone(np.array([1.0, 3.0, 2.0, 4.0]), np.ones(4)), [1, 2.5, 2.5, 4]
    )
    chain = flat_chain({7: 0.40, 9: 0.10})  # total variance would fall from day 7 to day 9
    assert (np.diff(variance.term_structure(chain)["total"]) >= 0).all()
    assert variance.forward_variance(chain).values.min() >= variance.MIN_FORWARD_VARIANCE


def test_fixed_tenor_interpolation():
    table = variance.term_structure(flat_chain({20: 0.15, 40: 0.15}))
    assert variance.vs_vol_at(table, 30) == pytest.approx(0.15, rel=0.005)
    assert np.isnan(variance.vs_vol_at(table, 60))


def test_real_nifty_day_sits_near_india_vix():
    chain = build_chain(nse.parse((FIXTURES / "fo_20261001_subset.csv.zip").read_bytes()), "NIFTY")
    table = variance.term_structure(chain)
    vs30 = variance.vs_vol_at(table, 30)
    assert vs30 == pytest.approx(0.1446, abs=0.01)  # India VIX closed at 14.46 that day
    atm = expiry_metrics(chain).set_index("expiry")["atm_iv"]
    assert (table["vs_vol"].to_numpy() > table["expiry"].map(atm).to_numpy()).all()  # skew premium
