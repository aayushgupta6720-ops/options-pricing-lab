"""Path-dependent options by Monte Carlo, under Black-Scholes, Heston or rough Bergomi.

Products, all monitored at n equally spaced fixing dates after today:
    asian            payoff on the arithmetic average of the fixings
    geometric_asian  payoff on the geometric average (has a closed form under Black-Scholes)
    barrier          knock-in or knock-out, up or down; continuously monitored
    lookback         floating strike: S_T - min S (call) or max S - S_T (put), continuously monitored

Continuous monitoring from discrete paths. Between two fixings the log-price is treated as a
Brownian bridge with that interval's variance, which is exact under Black-Scholes and an
approximation under stochastic volatility (it uses the interval's realised variance):
- a barrier's survival probability over an interval with both ends on the safe side is
  1 - exp(-2 (x0 - b)(x1 - b) / w); a knock-out pays the vanilla payoff times the product of these
  (a conditional expectation, smoother than sampling crossings), and a knock-in the rest.
- an interval's minimum (maximum) is sampled exactly from its bridge distribution:
  (x0 + x1 -/+ sqrt((x1 - x0)^2 - 2 w ln U)) / 2.

Paths come from `paths_*` as (log_s, step_variance) on the fixing grid. Under Black-Scholes and
Heston, `stats_*_numba` simulate and reduce each path on the fly in compiled code instead, never
storing a whole path, which keeps memory flat however many fixings there are.

Closed forms (Black-Scholes, continuous carry b = r - q): discretely monitored geometric Asian;
the eight standard barriers (Haug, The Complete Guide to Option Pricing Formulas, 2007);
Goldman-Sosin-Gatto floating lookbacks.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import ndtr

Product = Literal["asian", "geometric_asian", "barrier", "lookback"]


@dataclass(frozen=True)
class Exotic:
    product: Product
    kind: Literal["call", "put"] = "call"
    strike: float | None = None  # not used by floating lookbacks
    barrier: float | None = None
    knock: Literal["in", "out"] | None = None
    fixings: int = 50

    def __post_init__(self):
        if self.product == "barrier" and (self.barrier is None or self.knock not in ("in", "out")):
            raise ValueError("a barrier option needs a barrier level and knock='in' or 'out'")
        if self.product != "lookback" and self.strike is None:
            raise ValueError(f"a {self.product} option needs a strike")


@dataclass(frozen=True)
class MCPrice:
    price: float
    std_error: float
    n_paths: int


# --- Path statistics ------------------------------------------------------------------------


@dataclass(frozen=True)
class PathStats:
    """Per-path quantities every payoff here is built from (all in price terms, not logs)."""

    final: np.ndarray
    average: np.ndarray  # arithmetic mean of the fixings (excluding today)
    geometric: np.ndarray
    minimum: np.ndarray  # continuous-monitoring extremes (bridge-sampled), including today's spot
    maximum: np.ndarray
    survival: np.ndarray  # probability the path never touched the barrier (1 if no barrier)


def stats_from_paths(
    log_s: np.ndarray, step_var: np.ndarray, rng=None, barrier: float | None = None
) -> PathStats:
    rng = np.random.default_rng(rng)
    x0, x1 = log_s[:, :-1], log_s[:, 1:]
    w = np.maximum(step_var, 1e-300)
    jump = (x1 - x0) ** 2
    lows = 0.5 * (x0 + x1 - np.sqrt(jump - 2 * w * np.log(rng.random(w.shape))))
    highs = 0.5 * (x0 + x1 + np.sqrt(jump - 2 * w * np.log(rng.random(w.shape))))
    survival = np.ones(len(log_s))
    if barrier is not None:
        b = np.log(barrier)
        down = barrier < np.exp(log_s[0, 0])
        gap0, gap1 = (x0 - b, x1 - b) if down else (b - x0, b - x1)
        safe = (gap0 > 0) & (gap1 > 0)
        survival = np.where(safe, 1 - np.exp(-2 * gap0 * gap1 / w), 0.0).prod(axis=1)
    s = np.exp(log_s)
    return PathStats(
        final=s[:, -1],
        average=s[:, 1:].mean(axis=1),
        geometric=np.exp(log_s[:, 1:].mean(axis=1)),
        minimum=np.exp(lows.min(axis=1)),
        maximum=np.exp(highs.max(axis=1)),
        survival=survival,
    )


def payoffs(exotic: Exotic, stats: PathStats) -> np.ndarray:
    sign = 1.0 if exotic.kind == "call" else -1.0
    if exotic.product == "asian":
        return np.maximum(sign * (stats.average - exotic.strike), 0.0)
    if exotic.product == "geometric_asian":
        return np.maximum(sign * (stats.geometric - exotic.strike), 0.0)
    if exotic.product == "lookback":
        return stats.final - stats.minimum if exotic.kind == "call" else stats.maximum - stats.final
    vanilla = np.maximum(sign * (stats.final - exotic.strike), 0.0)
    return vanilla * (stats.survival if exotic.knock == "out" else 1.0 - stats.survival)


def price_from_stats(exotic: Exotic, stats: PathStats, r: float, T: float) -> MCPrice:
    discounted = np.exp(-r * T) * payoffs(exotic, stats)
    return MCPrice(
        float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(discounted.size)), discounted.size
    )


# --- Paths under each model (numpy) ---------------------------------------------------------


def paths_black_scholes(S, T, r, q, sigma, n_paths, n_fixings, rng=None, antithetic=True):
    rng = np.random.default_rng(rng)
    dt = T / n_fixings
    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((half, n_fixings))
    if antithetic:
        z = np.concatenate([z, -z])
    steps = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_s = np.concatenate([np.full((len(z), 1), np.log(S)), np.log(S) + np.cumsum(steps, axis=1)], axis=1)
    return log_s, np.full(steps.shape, sigma**2 * dt)


def paths_rough_bergomi(S, T, r, q, params, xi, n_paths, rng=None):
    """Daily fixings (T must be a whole number of days) from the rough Bergomi simulator."""
    from optlab.models import rough_bergomi

    normals = rough_bergomi.draw(n_paths, T, rng=rng)
    sim = rough_bergomi.paths(normals, params, xi)
    per_day = rough_bergomi.STEPS_PER_DAY
    days = int(round(T * 365))
    idx = np.arange(days + 1) * per_day
    t = sim.times[idx]
    log_s = np.log(S) + (r - q) * t[None, :] + sim.log_x[:, idx]
    v_dt = sim.variance[:, :-1] * normals.dt
    step_var = v_dt[:, : days * per_day].reshape(len(v_dt), days, per_day).sum(axis=2)
    return log_s, step_var


def stats_rough_bergomi(S, T, r, q, params, xi, n_paths, rng=None, barrier=None, chunk=2_000) -> PathStats:
    """Path statistics under rough Bergomi, simulated `chunk` paths at a time.

    A rough Bergomi path needs its whole history (the Volterra integral, an FFT convolution), so
    simulating every path at once costs memory in proportion to paths x steps: about 280 MB for
    20,000 paths over 22 days. Chunks keep the peak to a tenth of that for the same answer.
    """
    rng = np.random.default_rng(rng)
    parts = []
    for start in range(0, n_paths, chunk):
        size = min(chunk, n_paths - start)
        seed = int(rng.integers(2**32))
        log_s, w = paths_rough_bergomi(S, T, r, q, params, xi, size, rng=seed)
        parts.append(stats_from_paths(log_s, w, seed + 1, barrier))
        del log_s, w
    return PathStats(
        *(np.concatenate([getattr(p, f) for p in parts]) for f in PathStats.__dataclass_fields__)
    )


# --- Fused kernels (numba): simulate and reduce each path without storing it ----------------


def _numba():
    import numba

    return numba


_KERNELS: dict = {}


def _kernels():
    """Compile on first use, so pages that never price an exotic never import numba."""
    if _KERNELS:
        return _KERNELS
    numba = _numba()

    @numba.njit(cache=True)
    def bridge_update(x0, x1, w, b, down, has_barrier, lo, hi, survive):
        jump = (x1 - x0) ** 2
        low = 0.5 * (x0 + x1 - np.sqrt(jump - 2 * w * np.log(np.random.random())))
        high = 0.5 * (x0 + x1 + np.sqrt(jump - 2 * w * np.log(np.random.random())))
        lo = min(lo, low)
        hi = max(hi, high)
        if has_barrier:
            g0 = x0 - b if down else b - x0
            g1 = x1 - b if down else b - x1
            survive *= 1 - np.exp(-2 * g0 * g1 / w) if (g0 > 0 and g1 > 0) else 0.0
        return lo, hi, survive

    @numba.njit(cache=True)
    def gbm(n_paths, n_fixings, x_start, drift, sigma, dt, b, down, has_barrier, seed):
        np.random.seed(seed)
        out = np.empty((6, n_paths))
        w = sigma * sigma * dt
        for p in range(n_paths):
            x, total, log_total, lo, hi, survive = x_start, 0.0, 0.0, x_start, x_start, 1.0
            for _ in range(n_fixings):
                x1 = x + drift * dt + sigma * np.sqrt(dt) * np.random.standard_normal()
                lo, hi, survive = bridge_update(x, x1, w, b, down, has_barrier, lo, hi, survive)
                x = x1
                total += np.exp(x)
                log_total += x
            out[0, p], out[1, p], out[2, p] = np.exp(x), total / n_fixings, np.exp(log_total / n_fixings)
            out[3, p], out[4, p], out[5, p] = np.exp(lo), np.exp(hi), survive
        return out

    @numba.njit(cache=True)
    def heston(
        n_paths,
        n_fixings,
        substeps,
        x_start,
        carry,
        v0,
        kappa,
        theta,
        xi,
        rho,
        dt,
        b,
        down,
        has_barrier,
        seed,
    ):
        np.random.seed(seed)
        out = np.empty((6, n_paths))
        tilt = np.sqrt(1 - rho * rho)
        for p in range(n_paths):
            x, v, total, log_total, lo, hi, survive = x_start, v0, 0.0, 0.0, x_start, x_start, 1.0
            for _ in range(n_fixings):
                x0, w = x, 0.0
                for _ in range(substeps):
                    z1 = np.random.standard_normal()
                    z2 = rho * z1 + tilt * np.random.standard_normal()
                    pos = max(v, 0.0)
                    root = np.sqrt(pos * dt)
                    x += (carry - 0.5 * pos) * dt + root * z1
                    v += kappa * (theta - pos) * dt + xi * root * z2
                    w += pos * dt
                lo, hi, survive = bridge_update(x0, x, max(w, 1e-300), b, down, has_barrier, lo, hi, survive)
                total += np.exp(x)
                log_total += x
            out[0, p], out[1, p], out[2, p] = np.exp(x), total / n_fixings, np.exp(log_total / n_fixings)
            out[3, p], out[4, p], out[5, p] = np.exp(lo), np.exp(hi), survive
        return out

    _KERNELS.update(gbm=gbm, heston=heston)
    return _KERNELS


def warm_up():
    """Compile the kernels now (Render's build runs this), so numba's on-disk cache ships with the
    app and a cold start doesn't spend its first exotic price compiling."""
    from optlab.models.heston import HestonParams

    for barrier in (None, 90.0):
        stats_black_scholes_numba(100.0, 0.1, 0.05, 0.0, 0.2, 2, 2, seed=0, barrier=barrier)
        stats_heston_numba(
            100.0, 0.1, 0.05, 0.0, HestonParams(0.04, 2.0, 0.04, 0.5, -0.5), 2, 2, seed=0, barrier=barrier
        )


def _stats(out: np.ndarray) -> PathStats:
    return PathStats(*out)


def _barrier_args(S, barrier):
    if barrier is None:
        return 0.0, True, False
    return float(np.log(barrier)), bool(barrier < S), True


def stats_black_scholes_numba(S, T, r, q, sigma, n_paths, n_fixings, seed=0, barrier=None) -> PathStats:
    b, down, has = _barrier_args(S, barrier)
    out = _kernels()["gbm"](
        n_paths, n_fixings, np.log(S), r - q - 0.5 * sigma**2, sigma, T / n_fixings, b, down, has, seed
    )
    return _stats(out)


def stats_heston_numba(S, T, r, q, params, n_paths, n_fixings, substeps=4, seed=0, barrier=None) -> PathStats:
    b, down, has = _barrier_args(S, barrier)
    p = params
    dt = T / (n_fixings * substeps)
    out = _kernels()["heston"](
        n_paths,
        n_fixings,
        substeps,
        np.log(S),
        r - q,
        p.v0,
        p.kappa,
        p.theta,
        p.xi,
        p.rho,
        dt,
        b,
        down,
        has,
        seed,
    )
    return _stats(out)


# --- Closed forms under Black-Scholes -------------------------------------------------------


def geometric_asian_bs(S, K, T, r, q, sigma, n_fixings, kind="call") -> float:
    """Discretely monitored geometric-average option: the log of the average is normal."""
    n = n_fixings
    mean = np.log(S) + (r - q - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    var = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    sd = np.sqrt(var)
    d1 = (mean - np.log(K) + var) / sd
    d2 = d1 - sd
    forward = np.exp(mean + 0.5 * var)
    call = np.exp(-r * T) * (forward * ndtr(d1) - K * ndtr(d2))
    return float(call if kind == "call" else call - np.exp(-r * T) * (forward - K))


def barrier_bs(S, K, H, T, r, q, sigma, kind="call", knock="out") -> float:
    """Continuously monitored barrier option, no rebate (Haug 2007, section 4.17.1)."""
    b, vol = r - q, sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma**2) / sigma**2
    down = H < S
    phi = 1.0 if kind == "call" else -1.0
    eta = 1.0 if down else -1.0
    x1 = np.log(S / K) / vol + (1 + mu) * vol
    x2 = np.log(S / H) / vol + (1 + mu) * vol
    y1 = np.log(H * H / (S * K)) / vol + (1 + mu) * vol
    y2 = np.log(H / S) / vol + (1 + mu) * vol
    carry, disc = np.exp((b - r) * T), np.exp(-r * T)
    A = phi * S * carry * ndtr(phi * x1) - phi * K * disc * ndtr(phi * x1 - phi * vol)
    B = phi * S * carry * ndtr(phi * x2) - phi * K * disc * ndtr(phi * x2 - phi * vol)
    C = phi * S * carry * (H / S) ** (2 * (mu + 1)) * ndtr(eta * y1) - phi * K * disc * (H / S) ** (
        2 * mu
    ) * ndtr(eta * y1 - eta * vol)
    D = phi * S * carry * (H / S) ** (2 * (mu + 1)) * ndtr(eta * y2) - phi * K * disc * (H / S) ** (
        2 * mu
    ) * ndtr(eta * y2 - eta * vol)
    if kind == "call" and down:
        knocked_in = C if K > H else A - B + D
    elif kind == "call":
        knocked_in = A if K > H else B - C + D
    elif down:
        knocked_in = B - C + D if K > H else A
    else:
        knocked_in = A - B + D if K > H else C
    return float(knocked_in if knock == "in" else A - knocked_in)


def lookback_bs(S, T, r, q, sigma, kind="call") -> float:
    """Floating-strike lookback written today (running extreme = S), continuously monitored."""
    b = r - q
    if abs(b) < 1e-8:
        b = 1e-8  # the formula's b -> 0 limit is smooth; avoid dividing by zero
    vol = sigma * np.sqrt(T)
    a1 = (b + 0.5 * sigma**2) * T / vol
    a2 = a1 - vol
    scale = sigma**2 / (2 * b)
    carry, disc = np.exp((b - r) * T), np.exp(-r * T)
    if kind == "call":
        return float(
            S * carry * ndtr(a1)
            - S * disc * ndtr(a2)
            + S * disc * scale * (ndtr(-a1 + 2 * b * np.sqrt(T) / sigma) - np.exp(b * T) * ndtr(-a1))
        )
    return float(
        S * disc * ndtr(-a2)
        - S * carry * ndtr(-a1)
        + S * disc * scale * (-ndtr(a1 - 2 * b * np.sqrt(T) / sigma) + np.exp(b * T) * ndtr(a1))
    )


def closed_form_bs(exotic: Exotic, S, T, r, q, sigma) -> float | None:
    """The exact Black-Scholes price where there is one (not for arithmetic Asians)."""
    if exotic.product == "geometric_asian":
        return geometric_asian_bs(S, exotic.strike, T, r, q, sigma, exotic.fixings, exotic.kind)
    if exotic.product == "barrier":
        return barrier_bs(S, exotic.strike, exotic.barrier, T, r, q, sigma, exotic.kind, exotic.knock)
    if exotic.product == "lookback":
        return lookback_bs(S, T, r, q, sigma, exotic.kind)
    return None


# --- Variance reduction (arithmetic Asian under Black-Scholes) ------------------------------


def _bs_log_paths(S, T, r, q, sigma, z):
    """Log-price paths at the fixings from standard normals z (n_paths, n_fixings), PCA construction:
    the first coordinate carries the most variance of the Brownian path, which suits Sobol points."""
    n = z.shape[1]
    t = T * np.arange(1, n + 1) / n
    values, vectors = np.linalg.eigh(np.minimum.outer(t, t))
    order = np.argsort(values)[::-1]
    loadings = vectors[:, order] * np.sqrt(np.maximum(values[order], 0.0))
    W = z @ loadings.T
    return np.log(S) + (r - q - 0.5 * sigma**2) * t[None, :] + sigma * W


def _asian_payoffs(log_s, K, kind):
    sign = 1.0 if kind == "call" else -1.0
    arith = np.maximum(sign * (np.exp(log_s).mean(axis=1) - K), 0.0)
    geo = np.maximum(sign * (np.exp(log_s.mean(axis=1)) - K), 0.0)
    return arith, geo


def _with_control(y, x, x_mean):
    """Control-variate estimate: subtract beta (X - E[X]), beta fitted on the same sample. A control
    that never varies (every geometric payoff zero, as for a small batch far out of the money)
    carries no information, so it's left out rather than dividing by zero."""
    var_x = np.var(x, ddof=1)
    beta = np.cov(y, x)[0, 1] / var_x if var_x > 0 else 0.0
    return y - beta * (x - x_mean)


def variance_reduction(S, K, T, r, q, sigma, n_fixings=50, n_paths=50_000, kind="call", seed=0) -> list[dict]:
    """The same arithmetic Asian priced five ways, with each method's standard error, speed-up in
    variance per path over plain Monte Carlo, and time."""
    import time

    from scipy.stats import norm, qmc

    disc = np.exp(-r * T)
    geo_exact = geometric_asian_bs(S, K, T, r, q, sigma, n_fixings, kind)
    rng = np.random.default_rng(seed)
    rows = []

    def record(name, estimates, n, started):
        """estimates: independent draws of the price (per path, per antithetic pair, or per
        randomised Sobol set); their standard error is the method's."""
        estimates = np.asarray(estimates)
        price, se = estimates.mean(), estimates.std(ddof=1) / np.sqrt(len(estimates))
        rows.append({"method": name, "price": float(price), "std_error": float(se), "paths": n,
                     "seconds": time.perf_counter() - started})  # fmt: skip

    started = time.perf_counter()
    arith, _ = _asian_payoffs(
        _bs_log_paths(S, T, r, q, sigma, rng.standard_normal((n_paths, n_fixings))), K, kind
    )
    record("Plain Monte Carlo", disc * arith, n_paths, started)

    started = time.perf_counter()
    z = rng.standard_normal((n_paths // 2, n_fixings))
    a_plus, _ = _asian_payoffs(_bs_log_paths(S, T, r, q, sigma, z), K, kind)
    a_minus, _ = _asian_payoffs(_bs_log_paths(S, T, r, q, sigma, -z), K, kind)
    record("Antithetic paths", disc * 0.5 * (a_plus + a_minus), n_paths, started)  # pairs are independent

    started = time.perf_counter()
    arith, geo = _asian_payoffs(
        _bs_log_paths(S, T, r, q, sigma, rng.standard_normal((n_paths, n_fixings))), K, kind
    )
    record(
        "Control variate (geometric Asian)",
        _with_control(disc * arith, disc * geo, geo_exact),
        n_paths,
        started,
    )

    replicates = 16
    m = int(np.log2(max(n_paths // replicates, 2)))
    for name, control in (("Sobol + PCA paths", False), ("Sobol + PCA + control variate", True)):
        started = time.perf_counter()
        estimates = []
        for i in range(replicates):
            u = qmc.Sobol(n_fixings, scramble=True, seed=seed + 1000 * (i + 1)).random_base2(m)
            arith, geo = _asian_payoffs(_bs_log_paths(S, T, r, q, sigma, norm.ppf(u)), K, kind)
            values = _with_control(disc * arith, disc * geo, geo_exact) if control else disc * arith
            estimates.append(values.mean())
        record(name, estimates, replicates * 2**m, started)

    base = rows[0]["std_error"] ** 2 * rows[0]["paths"]
    for row in rows:
        row["variance_reduction"] = base / (row["std_error"] ** 2 * row["paths"])
    return rows
