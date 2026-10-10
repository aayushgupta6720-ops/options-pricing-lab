import numpy as np
import pytest

from optlab import exotics as ex
from optlab.exotics import Exotic
from optlab.models import black_scholes, heston
from optlab.models.heston import HestonParams
from optlab.models.rough_bergomi import ForwardVariance, RoughBergomiParams

S, T, R, Q, SIGMA, N = 100.0, 0.5, 0.05, 0.02, 0.25, 50


@pytest.fixture(scope="module")
def bs_paths():
    return ex.paths_black_scholes(S, T, R, Q, SIGMA, 100_000, N, rng=1)


def mc_price(exotic, paths, rng=2):
    log_s, step_var = paths
    return ex.price_from_stats(exotic, ex.stats_from_paths(log_s, step_var, rng, exotic.barrier), R, T)


@pytest.mark.parametrize(
    "exotic",
    [
        Exotic("geometric_asian", "call", 100.0, fixings=N),
        Exotic("geometric_asian", "put", 105.0, fixings=N),
        Exotic("lookback", "call", fixings=N),
        Exotic("lookback", "put", fixings=N),
        Exotic("barrier", "call", 100.0, 90.0, "out", N),
        Exotic("barrier", "call", 90.0, 95.0, "in", N),
        Exotic("barrier", "call", 100.0, 120.0, "out", N),
        Exotic("barrier", "put", 100.0, 110.0, "in", N),
        Exotic("barrier", "put", 110.0, 105.0, "out", N),
        Exotic("barrier", "put", 100.0, 85.0, "in", N),
    ],
    ids=lambda e: f"{e.product}-{e.kind}-{e.knock or ''}-{e.barrier or ''}",
)
def test_monte_carlo_matches_the_closed_form(bs_paths, exotic):
    mc = mc_price(exotic, bs_paths)
    exact = ex.closed_form_bs(exotic, S, T, R, Q, SIGMA)
    assert abs(mc.price - exact) < 4 * mc.std_error + 1e-3, (mc, exact)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("barrier", [80.0, 95.0, 105.0, 125.0])
@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
def test_knock_in_plus_knock_out_is_the_vanilla(kind, barrier, strike):
    vanilla = black_scholes.price(S, strike, T, R, SIGMA, Q, kind)
    both = sum(ex.barrier_bs(S, strike, barrier, T, R, Q, SIGMA, kind, knock) for knock in ("in", "out"))
    assert both == pytest.approx(float(vanilla), abs=1e-10)


def test_a_barrier_nobody_reaches_changes_nothing():
    vanilla = float(black_scholes.price(S, 100.0, T, R, SIGMA, Q, "call"))
    assert ex.barrier_bs(S, 100.0, 1.0, T, R, Q, SIGMA, "call", "out") == pytest.approx(vanilla, abs=1e-8)
    assert ex.barrier_bs(S, 100.0, 1.0, T, R, Q, SIGMA, "call", "in") == pytest.approx(0.0, abs=1e-8)


def test_lookbacks_are_worth_more_than_at_the_money_vanillas():
    assert ex.lookback_bs(S, T, R, Q, SIGMA, "call") > black_scholes.price(S, S, T, R, SIGMA, Q, "call")
    assert ex.lookback_bs(S, T, R, Q, SIGMA, "put") > black_scholes.price(S, S, T, R, SIGMA, Q, "put")
    # zero carry is a removable singularity in the formula
    assert ex.lookback_bs(S, T, 0.03, 0.03, SIGMA) == pytest.approx(
        ex.lookback_bs(S, T, 0.03, 0.0300001, SIGMA), rel=1e-4
    )


def test_numba_kernel_agrees_with_numpy(bs_paths):
    for exotic in (
        Exotic("asian", "call", 100.0, fixings=N),
        Exotic("barrier", "call", 100.0, 90.0, "out", N),
    ):
        stats = ex.stats_black_scholes_numba(S, T, R, Q, SIGMA, 100_000, N, seed=3, barrier=exotic.barrier)
        fast, slow = ex.price_from_stats(exotic, stats, R, T), mc_price(exotic, bs_paths)
        assert abs(fast.price - slow.price) < 4 * np.hypot(fast.std_error, slow.std_error)


def test_heston_paths_and_kernel_agree_and_reduce_to_black_scholes():
    flat = HestonParams(SIGMA**2, 2.0, SIGMA**2, 1e-3, 0.0)  # constant variance: Black-Scholes
    exotic = Exotic("barrier", "call", 100.0, 90.0, "out", 25)
    exact = ex.barrier_bs(S, 100.0, 90.0, T, R, Q, SIGMA, "call", "out")
    stats = ex.stats_heston_numba(S, T, R, Q, flat, 60_000, 25, seed=4, barrier=90.0)
    fast = ex.price_from_stats(exotic, stats, R, T)
    log_s, w = heston.simulate_paths(S, T, R, Q, flat, 60_000, 25, rng=5)
    slow = ex.price_from_stats(exotic, ex.stats_from_paths(log_s, w, 6, 90.0), R, T)
    for estimate in (fast, slow):
        assert abs(estimate.price - exact) < 4 * estimate.std_error + 0.01


def test_rough_bergomi_paths_reduce_to_black_scholes():
    # H = 1/2, no vol of vol: variance stays at xi0.
    log_s, w = ex.paths_rough_bergomi(
        S, 30 / 365, R, Q, RoughBergomiParams(0.5, 0.0, 0.0), ForwardVariance.flat(SIGMA**2), 40_000, rng=7
    )
    exotic = Exotic("lookback", "call", fixings=30)
    mc = ex.price_from_stats(exotic, ex.stats_from_paths(log_s, w, 8), R, 30 / 365)
    assert abs(mc.price - ex.lookback_bs(S, 30 / 365, R, Q, SIGMA)) < 4 * mc.std_error + 0.01


def test_variance_reduction_methods_agree_and_reduce_variance():
    rows = {r["method"]: r for r in ex.variance_reduction(S, 100.0, T, R, Q, SIGMA, 50, 32_000)}
    plain = rows["Plain Monte Carlo"]
    for row in rows.values():
        assert abs(row["price"] - plain["price"]) < 4 * np.hypot(row["std_error"], plain["std_error"])
    assert rows["Control variate (geometric Asian)"]["variance_reduction"] > 100
    assert (
        rows["Sobol + PCA + control variate"]["variance_reduction"]
        > rows["Control variate (geometric Asian)"]["variance_reduction"]
    )


def test_a_control_that_never_varies_is_left_out_instead_of_giving_nan():
    # 4,000 paths in 16 Sobol sets of 250, 5% out of the money over 22 days: some sets have no
    # geometric payoff at all, and the control's variance is zero.
    rows = {
        r["method"]: r for r in ex.variance_reduction(24_900, 26_145, 22 / 365, 0.055, 0.012, 0.14, 22, 4_000)
    }
    assert all(np.isfinite(r["price"]) and np.isfinite(r["std_error"]) for r in rows.values())


def test_the_control_variate_factor_holds_at_any_path_count_but_sobol_grows():
    def factors(n):
        return {
            r["method"]: r["variance_reduction"]
            for r in ex.variance_reduction(S, 100.0, T, R, Q, SIGMA, 50, n, seed=21)
        }

    small, large = factors(4_000), factors(64_000)
    cv = "Control variate (geometric Asian)"
    assert 0.6 < large[cv] / small[cv] < 1.6
    assert large["Sobol + PCA paths"] > 4 * small["Sobol + PCA paths"]


def test_invalid_exotics_are_rejected():
    with pytest.raises(ValueError):
        Exotic("barrier", "call", 100.0)
    with pytest.raises(ValueError):
        Exotic("asian", "call")


def test_chunked_rough_bergomi_gives_the_same_answer():
    params, xi = RoughBergomiParams(0.1, 1.5, -0.7), ForwardVariance.flat(SIGMA**2)
    exotic = Exotic("barrier", "call", 100.0, 92.0, "out", 20)
    T20 = 20 / 365
    whole = ex.price_from_stats(
        exotic,
        ex.stats_rough_bergomi(S, T20, R, Q, params, xi, 24_000, rng=1, barrier=92.0, chunk=24_000),
        R,
        T20,
    )
    chunked = ex.price_from_stats(
        exotic,
        ex.stats_rough_bergomi(S, T20, R, Q, params, xi, 24_000, rng=1, barrier=92.0, chunk=5_000),
        R,
        T20,
    )
    assert chunked.n_paths == 24_000
    assert abs(whole.price - chunked.price) < 4 * np.hypot(whole.std_error, chunked.std_error)


def test_warm_up_compiles_without_error():
    ex.warm_up()
