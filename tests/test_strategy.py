import numpy as np
import pytest

from optlab import strategy
from optlab.models import black_scholes
from optlab.strategy import Leg


def test_presets_place_strikes_around_the_money():
    condor = strategy.preset_legs("Iron condor", 22500, 200)
    assert [(leg.side, leg.kind, leg.strike) for leg in condor] == [
        (1, "put", 22100),
        (-1, "put", 22300),
        (-1, "call", 22700),
        (1, "call", 22900),
    ]


def test_straddle_breakevens_are_strike_plus_and_minus_the_premium():
    legs = [Leg(1, "call", 100), Leg(1, "put", 100)]
    net = strategy.premium(legs, [4.0, 3.0])
    spots = np.linspace(50, 150, 1001)
    assert strategy.breakevens(spots, strategy.payoff(legs, spots) - net) == pytest.approx([93.0, 107.0])


def test_iron_condor_is_bounded_on_both_sides():
    legs = strategy.preset_legs("Iron condor", 100, 5)  # long 90P, short 95P, short 105C, long 110C
    net = strategy.premium(legs, [0.5, 1.5, 1.4, 0.4])  # a credit of 2.0
    best, worst = strategy.extremes(legs, net)
    assert best == pytest.approx(2.0)
    assert worst == pytest.approx(-(5 - 2.0))


def test_short_put_worst_case_is_spot_going_to_zero():
    legs = [Leg(-1, "put", 100)]
    best, worst = strategy.extremes(legs, strategy.premium(legs, [3.0]))
    assert best == pytest.approx(3.0)
    assert worst == pytest.approx(-97.0)


def test_naked_calls_are_unlimited():
    assert strategy.extremes([Leg(1, "call", 100)], 5.0)[0] == np.inf
    assert strategy.extremes([Leg(-1, "call", 100)], -5.0)[1] == -np.inf
    # A call spread is capped even though it has calls in it.
    assert np.isfinite(strategy.extremes([Leg(1, "call", 100), Leg(-1, "call", 110)], 4.0)).all()


def test_value_matches_black_76_and_collapses_to_payoff_at_expiry():
    legs = [Leg(1, "call", 100, lots=2), Leg(-1, "put", 95)]
    spots = np.array([90.0, 100.0, 110.0])
    ratio, T, r, vols = 1.01, 0.25, 0.06, [0.2, 0.25]
    expected = 2 * black_scholes.price(spots * ratio, 100, T, r, 0.2, r, "call") - black_scholes.price(
        spots * ratio, 95, T, r, 0.25, r, "put"
    )
    np.testing.assert_allclose(strategy.value(legs, spots, ratio, T, r, vols), expected)
    np.testing.assert_allclose(strategy.value(legs, spots, ratio, 0.0, r, vols), strategy.payoff(legs, spots))


def test_position_delta_matches_a_finite_difference():
    legs = strategy.preset_legs("Bull call spread", 100, 5)
    ratio, T, r, vols = 1.005, 0.1, 0.06, [0.22, 0.2]
    g = strategy.greeks(legs, 100.0, ratio, T, r, vols)
    h = 0.01
    up, down = (strategy.value(legs, [100.0 + d], ratio, T, r, vols)[0] for d in (h, -h))
    assert g["delta"] == pytest.approx((up - down) / (2 * h), rel=1e-5)


def test_selling_the_inner_strikes_of_a_condor_collects_a_credit():
    legs = strategy.preset_legs("Iron condor", 26300, 500)
    vols = [0.116, 0.103, 0.089, 0.093]  # a real NIFTY smile: puts richer than calls
    prices = strategy.leg_prices(legs, 26203.0, 1.0046, 18 / 365, 0.055, vols)
    assert all(p > 0 for p in prices)
    net = strategy.premium(legs, prices)
    assert net < 0  # credit
    best, worst = strategy.extremes(legs, net)
    assert best == pytest.approx(-net)
    assert worst == pytest.approx(-(500 + net))
