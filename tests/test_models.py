import numpy as np
import pytest

from optlab.contracts import OptionSpec
from optlab.greeks import finite_difference
from optlab.models import binomial, black_scholes, monte_carlo
from optlab.models.registry import MODELS, models_for

# Hull, Options, Futures and Other Derivatives, Example 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5.
HULL = dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20)


def test_black_scholes_matches_hull_example():
    assert black_scholes.price(**HULL, kind="call") == pytest.approx(4.76, abs=0.005)
    assert black_scholes.price(**HULL, kind="put") == pytest.approx(0.81, abs=0.005)


def test_black_scholes_greeks_match_hull_example():
    # Hull Example 19.1 (S=49, K=50, r=5%, sigma=20%, 20 weeks): delta 0.522, gamma 0.066,
    # vega 12.1, theta -4.31 per year, rho 8.91.
    g = black_scholes.greeks(49.0, 50.0, 0.3846, 0.05, 0.20)
    assert g["delta"] == pytest.approx(0.522, abs=0.001)
    assert g["gamma"] == pytest.approx(0.066, abs=0.001)
    assert g["vega"] == pytest.approx(12.1, abs=0.05)
    assert g["theta"] == pytest.approx(-4.31, abs=0.01)
    assert g["rho"] == pytest.approx(8.91, abs=0.01)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(q):
    K = np.linspace(20, 80, 13)
    call = black_scholes.price(50.0, K, 0.75, 0.06, 0.3, q, "call")
    put = black_scholes.price(50.0, K, 0.75, 0.06, 0.3, q, "put")
    np.testing.assert_allclose(call - put, 50.0 * np.exp(-q * 0.75) - K * np.exp(-0.06 * 0.75), atol=1e-10)


def test_black_scholes_broadcasts_over_mixed_kinds():
    kinds = np.array(["call", "put", "call"])
    prices = black_scholes.price(42.0, 40.0, 0.5, 0.10, 0.20, 0.0, kinds)
    assert prices.shape == (3,)
    assert prices[0] == prices[2] == pytest.approx(4.76, abs=0.005)
    assert prices[1] == pytest.approx(0.81, abs=0.005)


def test_black_scholes_rejects_unknown_kind():
    with pytest.raises(ValueError):
        black_scholes.price(42.0, 40.0, 0.5, 0.10, 0.20, kind="CE")


def test_black_scholes_degenerate_limits():
    # At expiry: intrinsic value.
    assert black_scholes.price(110.0, 100.0, 0.0, 0.05, 0.2, kind="call") == pytest.approx(10.0)
    assert black_scholes.price(110.0, 100.0, 0.0, 0.05, 0.2, kind="put") == pytest.approx(0.0)
    # Zero vol: discounted payoff on the forward, F = S exp((r - q) T).
    forward = 100.0 * np.exp(0.05)
    assert black_scholes.price(100.0, 100.0, 1.0, 0.05, 0.0) == pytest.approx(
        np.exp(-0.05) * (forward - 100.0)
    )
    assert black_scholes.price(100.0, 100.0, 1.0, 0.05, 0.0, kind="put") == pytest.approx(0.0)
    g = black_scholes.greeks(100.0, 100.0, 1.0, 0.05, 0.0)
    assert g["gamma"] == 0.0 and g["vega"] == 0.0 and g["delta"] == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [80.0, 100.0, 125.0])
def test_analytic_greeks_match_finite_differences(kind, K):
    spec = OptionSpec(S=100.0, K=K, T=0.8, r=0.05, sigma=0.25, q=0.02, kind=kind)
    analytic = black_scholes.greeks_spec(spec)
    numeric = finite_difference(black_scholes.price_spec, spec)
    for name in analytic:
        assert numeric[name] == pytest.approx(analytic[name], rel=1e-3, abs=1e-4), name


@pytest.mark.parametrize("kind", ["call", "put"])
def test_binomial_converges_to_black_scholes(kind):
    exact = black_scholes.price(**HULL, kind=kind)
    errors = [abs(binomial.price(**HULL, kind=kind, steps=n) - exact) for n in (50, 500, 5000)]
    assert errors[2] < 2e-3
    assert errors[2] < errors[1] < errors[0]


def test_american_call_without_dividends_equals_european():
    american = binomial.price(**HULL, kind="call", style="american", steps=800)
    european = binomial.price(**HULL, kind="call", style="european", steps=800)
    assert american == pytest.approx(european, abs=1e-12)


def test_american_put_is_worth_more_than_european():
    american = binomial.price(**HULL, kind="put", style="american", steps=800)
    european = binomial.price(**HULL, kind="put", style="european", steps=800)
    assert american > european + 0.01


def test_american_put_matches_hull_five_step_tree():
    # Hull Example 21.1: S=K=50, r=10%, sigma=40%, T=5 months, 5 steps -> $4.49.
    value = binomial.price(50.0, 50.0, 5 / 12, 0.10, 0.40, kind="put", style="american", steps=5)
    assert value == pytest.approx(4.49, abs=0.005)


def test_binomial_zero_vol_american_put_exercises_at_once():
    # Deep in the money with no volatility and r > 0: exercising now beats waiting.
    assert binomial.price(50.0, 100.0, 1.0, 0.05, 0.0, kind="put", style="american") == pytest.approx(50.0)
    assert binomial.price(50.0, 100.0, 1.0, 0.05, 0.0, kind="put") == pytest.approx(100 * np.exp(-0.05) - 50)


def test_binomial_rejects_too_few_steps_for_the_drift():
    with pytest.raises(ValueError, match="more steps"):
        binomial.price(100.0, 100.0, 10.0, 0.30, 0.01, steps=2)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_monte_carlo_lands_within_three_standard_errors(kind):
    result = monte_carlo.price(**HULL, kind=kind, n_paths=200_000, seed=1)
    assert abs(result.price - black_scholes.price(**HULL, kind=kind)) < 3 * result.std_error
    lo, hi = result.ci95
    assert lo < result.price < hi


def test_monte_carlo_error_shrinks_like_one_over_root_n():
    small = monte_carlo.price(**HULL, n_paths=10_000, seed=2).std_error
    large = monte_carlo.price(**HULL, n_paths=160_000, seed=2).std_error
    assert large == pytest.approx(small / 4, rel=0.1)


def test_monte_carlo_is_reproducible_with_a_seed():
    assert monte_carlo.price(**HULL, seed=3) == monte_carlo.price(**HULL, seed=3)


def test_gbm_paths_are_a_martingale_after_discounting():
    paths = monte_carlo.gbm_paths(100.0, 1.0, 0.05, 0.3, q=0.01, n_paths=200_000, n_steps=12, rng=4)
    assert paths.shape == (200_000, 13)
    assert np.all(paths[:, 0] == 100.0)
    discounted = paths[:, -1] * np.exp(-(0.05 - 0.01))
    assert discounted.mean() == pytest.approx(100.0, rel=3 * 0.3 / np.sqrt(200_000))


def test_every_registered_model_prices_the_hull_example():
    spec = OptionSpec(**HULL)
    for model in models_for(spec):
        assert model.price(spec) == pytest.approx(4.76, abs=0.01), model.name


def test_only_tree_models_price_american_options():
    american = OptionSpec(**HULL, kind="put", style="american")
    assert [m.name for m in models_for(american)] == ["Binomial (CRR)"]
    assert set(MODELS) >= {"Black-Scholes", "Binomial (CRR)", "Monte Carlo"}
    with pytest.raises(ValueError):
        black_scholes.price_spec(american)


def test_option_spec_validates_inputs():
    with pytest.raises(ValueError):
        OptionSpec(S=100, K=100, T=1, r=0.05, sigma=0.2, kind="CE")
    with pytest.raises(ValueError):
        OptionSpec(S=-1, K=100, T=1, r=0.05, sigma=0.2)
    with pytest.raises(ValueError):
        OptionSpec(S=100, K=100, T=-1, r=0.05, sigma=0.2)
