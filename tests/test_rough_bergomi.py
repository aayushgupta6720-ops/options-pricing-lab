import numpy as np
import pytest

from optlab.implied_vol import implied_vol
from optlab.models import rough_bergomi as rb
from optlab.models.rough_bergomi import ForwardVariance, RoughBergomiParams

ROUGH = RoughBergomiParams(H=0.1, eta=1.9, rho=-0.7)


@pytest.fixture(scope="module")
def normals():
    return rb.draw(20_000, 30 / 365, rng=1)


def test_forward_is_a_martingale(normals):
    sim = rb.paths(normals, ROUGH, ForwardVariance.flat(0.04))
    x = np.exp(sim.log_x[:, -1])
    assert abs(x.mean() - 1) < 3 * x.std() / np.sqrt(x.size)


def test_expected_variance_follows_the_forward_variance_curve(normals):
    curve = ForwardVariance(np.array([10 / 365, np.inf]), np.array([0.02, 0.05]))
    # A milder eta: at 1.9, V is so heavy-tailed that 20,000 paths average it to only ~2%.
    sim = rb.paths(normals, RoughBergomiParams(0.1, 1.0, -0.7), curve)
    for day in (3, 7, 20, 29):
        i = day * rb.STEPS_PER_DAY
        assert sim.variance[:, i].mean() == pytest.approx(float(curve(sim.times[i])), rel=0.04), day
    assert curve.total(30 / 365) == pytest.approx(0.02 * 10 / 365 + 0.05 * 20 / 365)


@pytest.mark.parametrize("H", [0.05, 0.1, 0.3])
def test_volterra_process_has_variance_t_to_the_2h(normals, H):
    _, Y = rb.volterra(normals, H)
    for day in (1, 7, 29):
        i = day * rb.STEPS_PER_DAY
        assert Y[:, i].var() == pytest.approx((i * normals.dt) ** (2 * H), rel=0.04), (H, day)


def test_at_one_half_it_is_a_brownian_motion(normals):
    dW, Y = rb.volterra(normals, 0.5)
    np.testing.assert_allclose(Y[:, 1:], np.cumsum(dW, axis=1), atol=1e-12)


def _atm_skews(normals, params, days):
    sim = rb.paths(normals, params, ForwardVariance.flat(0.15**2))
    K, kinds = np.exp([-0.01, 0.01]), np.array(["put", "call"])
    out = []
    for d in days:
        v = implied_vol(rb.price(1.0, K, d / 365, 0.0, sim, kinds), 1.0, K, d / 365, 0.0, 0.0, kinds)
        out.append(abs(v[1] - v[0]) / 0.02)
    return np.array(out)


def test_rough_skew_follows_a_power_law_and_classical_skew_does_not(normals):
    days = np.array([3, 7, 14, 28])
    rough = np.polyfit(np.log(days), np.log(_atm_skews(normals, ROUGH, days)), 1)[0]
    classical = np.polyfit(
        np.log(days), np.log(_atm_skews(normals, RoughBergomiParams(0.5, 1.0, -0.7), days)), 1
    )[0]
    assert -0.45 < rough < -0.28  # theory: H - 1/2 = -0.4 as T -> 0
    assert abs(classical) < 0.08  # a classical model's skew levels off at short maturities


def test_put_call_parity_and_grid_checks(normals):
    sim = rb.paths(normals, ROUGH, ForwardVariance.flat(0.04))
    K, T = np.array([90.0, 100.0, 110.0]), 14 / 365
    calls = rb.price(100.0, K, T, 0.06, sim, "call")
    puts = rb.price(100.0, K, T, 0.06, sim, "put")
    np.testing.assert_allclose(calls - puts, np.exp(-0.06 * T) * (100.0 - K), atol=1e-10)
    with pytest.raises(ValueError, match="grid"):
        rb.price(100.0, 100.0, 14.1 / 365, 0.06, sim)


def test_same_random_numbers_give_a_smooth_price_in_the_parameters(normals):
    curve = ForwardVariance.flat(0.03)

    def put(z, eta):
        return float(
            rb.price(1.0, 0.98, 14 / 365, 0.0, rb.paths(z, RoughBergomiParams(0.1, eta, -0.7), curve), "put")
        )

    bumped = abs(put(normals, 1.600001) - put(normals, 1.6))
    fresh = abs(put(rb.draw(20_000, 30 / 365, rng=2), 1.6) - put(normals, 1.6))
    assert bumped < 0.01 * fresh  # a tiny bump moves the price far less than Monte Carlo noise would


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        RoughBergomiParams(0.6, 1.0, -0.5)
    with pytest.raises(ValueError):
        RoughBergomiParams(0.1, 1.0, -1.0)
