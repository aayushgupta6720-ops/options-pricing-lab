import numpy as np
import pytest

from optlab.implied_vol import implied_vol
from optlab.models import black_scholes


def test_round_trip_over_a_grid():
    K = np.linspace(60, 160, 41)[:, None, None]
    T = np.array([2 / 365, 0.1, 0.5, 2.0])[None, :, None]
    sigma = np.array([0.05, 0.2, 0.6, 1.5])[None, None, :]
    for kind in ("call", "put"):
        price = black_scholes.price(100.0, K, T, 0.07, sigma, 0.01, kind)
        recovered = implied_vol(price, 100.0, K, T, 0.07, 0.01, kind)
        # With almost no time value the price barely moves with sigma, so implied vol is only
        # pinned down to a few digits (far out of the money or deep in it, short-dated, low vol).
        # Only check prices with at least 0.1% of spot in time value.
        intrinsic = black_scholes.price(100.0, K, T, 0.07, 1e-4, 0.01, kind)
        meaningful = (price - intrinsic) > 0.1
        assert meaningful.mean() > 0.5  # guard against the check passing vacuously
        np.testing.assert_allclose(
            np.broadcast_to(recovered, price.shape)[meaningful],
            np.broadcast_to(sigma, price.shape)[meaningful],
            rtol=1e-6,
        )


def test_mixed_calls_and_puts_in_one_call():
    kinds = np.array(["call", "put", "put", "call"])
    K = np.array([90.0, 90.0, 110.0, 110.0])
    sigma = np.array([0.15, 0.25, 0.35, 0.45])
    price = black_scholes.price(100.0, K, 0.5, 0.05, sigma, 0.0, kinds)
    np.testing.assert_allclose(implied_vol(price, 100.0, K, 0.5, 0.05, 0.0, kinds), sigma, rtol=1e-8)


def test_scalar_in_scalar_out():
    price = black_scholes.price(42.0, 40.0, 0.5, 0.10, 0.20)
    assert implied_vol(price, 42.0, 40.0, 0.5, 0.10) == pytest.approx(0.20, rel=1e-9)
    assert np.ndim(implied_vol(price, 42.0, 40.0, 0.5, 0.10)) == 0


def test_prices_outside_no_arbitrage_bounds_are_nan():
    # Below intrinsic, above the spot (call), at expiry, and missing.
    prices = np.array([1.0, 150.0, 5.0, np.nan])
    T = np.array([0.5, 0.5, 0.0, 0.5])
    out = implied_vol(prices, 100.0, 90.0, T, 0.05)
    assert np.isnan(out).all()


def test_american_implied_vol_round_trip():
    from optlab.contracts import OptionSpec
    from optlab.implied_vol import american_implied_vol
    from optlab.models import binomial

    spec = OptionSpec(50.0, 55.0, 0.75, 0.08, 0.3, kind="put", style="american")
    price = binomial.price_spec(spec, steps=300)
    assert american_implied_vol(price, spec) == pytest.approx(0.3, abs=1e-6)
    assert np.isnan(american_implied_vol(1.0, spec))  # below the exercise value of 5
