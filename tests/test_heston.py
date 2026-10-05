import numpy as np
import pytest
from scipy.integrate import quad

from optlab.models import black_scholes, heston
from optlab.models.heston import HestonParams

# Fang & Oosterlee (2008), the COS-method paper's Heston test: S = K = 100, T = 1, r = q = 0.
FANG_OOSTERLEE = HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
NIFTYISH = HestonParams(v0=0.02, kappa=3.0, theta=0.03, xi=0.8, rho=-0.7)


def test_matches_the_published_reference_price():
    assert heston.price_spot(100.0, 100.0, 1.0, 0.0, 0.0, FANG_OOSTERLEE) == pytest.approx(
        5.785155450, abs=1e-7
    )


def test_characteristic_function_is_a_martingale_density():
    for T in (0.01, 1.0, 5.0):
        assert heston.char_fn(0.0, T, NIFTYISH) == pytest.approx(1.0)
        assert heston.char_fn(-1j, T, NIFTYISH) == pytest.approx(1.0)  # E[S_T / F] = 1


def _quad_call(F, K, T, r, p):
    x = np.log(F / K)

    def integrand(u):
        return (np.exp(1j * u * x) * heston.char_fn(u - 0.5j, T, p) / (u * u + 0.25)).real

    value, _ = quad(integrand, 0, np.inf, limit=2000, epsabs=1e-13, epsrel=1e-12)
    return np.exp(-r * T) * (F - np.sqrt(F * K) / np.pi * value)


@pytest.mark.parametrize("days", [2, 30, 730])
def test_fast_integration_matches_adaptive_quadrature(days):
    T, F = days / 365, 22500.0
    K = F * np.exp(np.linspace(-0.4, 0.4, 9))
    fast = heston.price(F, K, T, 0.055, NIFTYISH)
    slow = [_quad_call(F, k, T, 0.055, NIFTYISH) for k in K]
    np.testing.assert_allclose(fast, slow, atol=1e-6)


def test_put_call_parity():
    K = np.linspace(18000, 27000, 19)
    F, T, r = 22500.0, 0.25, 0.055
    calls = heston.price(F, K, T, r, NIFTYISH, "call")
    puts = heston.price(F, K, T, r, NIFTYISH, "put")
    np.testing.assert_allclose(calls - puts, np.exp(-r * T) * (F - K), atol=1e-8)


def test_reduces_to_black_scholes_without_vol_of_vol():
    # Constant variance (v0 = theta) and xi -> 0: the price error shrinks like xi^2 when rho = 0.
    K = np.linspace(60, 160, 21)
    flat = HestonParams(0.04, 2.0, 0.04, 1e-3, 0.0)
    np.testing.assert_allclose(
        heston.price_spot(100.0, K, 1.0, 0.05, 0.01, flat),
        black_scholes.price(100.0, K, 1.0, 0.05, 0.2, 0.01),
        atol=1e-5,
    )


def test_negative_correlation_makes_downside_strikes_dearer():
    from optlab.implied_vol import implied_vol

    F, T, r = 100.0, 0.5, 0.0
    K = np.array([80.0, 100.0, 120.0])
    vols = implied_vol(heston.price(F, K, T, r, NIFTYISH, "call"), F, K, T, r, r, "call")
    assert vols[0] > vols[1] > vols[2]


def test_monte_carlo_agrees_with_the_formula():
    terminal = heston.simulate(100.0, 1.0, 0.0, 0.0, FANG_OOSTERLEE, n_paths=100_000, n_steps=200, rng=1)
    for K in (80.0, 100.0, 120.0):
        payoff = np.maximum(terminal - K, 0.0)
        estimate, error = payoff.mean(), payoff.std(ddof=1) / np.sqrt(payoff.size)
        exact = heston.price_spot(100.0, K, 1.0, 0.0, 0.0, FANG_OOSTERLEE)
        assert abs(estimate - exact) < 3 * error + 0.01, K  # 0.01 allows for time-stepping bias


def test_at_expiry_the_price_is_the_payoff():
    np.testing.assert_allclose(
        heston.price(100.0, [90.0, 110.0], 0.0, 0.05, NIFTYISH, ["call", "put"]), [10.0, 10.0]
    )


def test_feller_ratio_and_validation():
    assert FANG_OOSTERLEE.feller_ratio == pytest.approx(2 * 1.5768 * 0.0398 / 0.5751**2)
    with pytest.raises(ValueError):
        HestonParams(0.04, 2.0, 0.04, 0.5, -1.0)
    with pytest.raises(ValueError):
        heston.price(100.0, 100.0, 1.0, 0.0, NIFTYISH, "CE")
