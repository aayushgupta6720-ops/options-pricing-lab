import numpy as np
import pytest

from optlab import exotics as ex
from optlab import local_vol as lv
from optlab.exotics import Exotic
from optlab.models import heston
from optlab.models.heston import HestonParams

S, T, R, Q = 100.0, 22 / 365, 0.055, 0.012
NIFTY_LIKE = HestonParams(0.02, 3.0, 0.025, 0.5, -0.6)


def test_a_flat_implied_surface_has_a_flat_local_vol():
    times = np.linspace(1 / 365, 0.5, 20)
    surface = lv.from_total_variance(0.04 * times[:, None] * np.ones((1, lv.Y_GRID.size)), times)
    np.testing.assert_allclose(surface.vols, 0.2, rtol=1e-9)
    assert surface(0.1, 0.03) == pytest.approx(0.2)


def test_local_vol_built_from_heston_reprices_hestons_vanilla_options():
    surface = lv.from_heston(NIFTY_LIKE, T)
    stats = lv.stats_local_vol_numba(S, T, R, Q, surface, 120_000, 22, substeps=4, seed=1)
    F, disc = S * np.exp((R - Q) * T), np.exp(-R * T)
    for k, kind in ((0.95, "put"), (1.0, "call"), (1.05, "call")):
        payoff = disc * np.maximum((stats.final - k * F) * (1 if kind == "call" else -1), 0)
        error = payoff.std() / np.sqrt(payoff.size)
        assert (
            abs(payoff.mean() - float(heston.price(F, k * F, T, R, NIFTY_LIKE, kind))) < 4 * error + 0.005
        ), k


def test_skew_shows_up_as_higher_local_vol_below_the_money():
    surface = lv.from_heston(NIFTY_LIKE, T)
    assert surface(T, -0.05) > surface(T, 0.0) > surface(T, 0.05)  # negative spot-vol correlation


def test_under_a_flat_surface_a_barrier_prices_as_black_scholes_does():
    flat = lv.LocalVolSurface(np.array([0.0, 1.0]), np.array([-1.0, 1.0]), np.full((2, 2), 0.2))
    barrier = Exotic("barrier", "call", 100.0, 110.0, "out", fixings=30)
    price = ex.price_from_stats(
        barrier, lv.stats_local_vol_numba(S, 0.25, R, Q, flat, 60_000, 30, seed=3, barrier=110.0), R, 0.25
    )
    exact = ex.barrier_bs(S, 100.0, 110.0, 0.25, R, Q, 0.2, "call", "out")
    assert abs(price.price - exact) < 4 * price.std_error + 0.02
