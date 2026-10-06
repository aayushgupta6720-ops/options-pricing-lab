"""The rough Bergomi model (Bayer, Friz and Gatheral, 2016).

    V_t = xi0(t) exp(eta Y_t - eta^2 / 2 t^(2H)),   Y_t = sqrt(2H) Int_0^t (t - s)^(H - 1/2) dW_s
    dS_t / S_t = (r - q) dt + sqrt(V_t) dB_t,       B = rho W + sqrt(1 - rho^2) W_perp

Y is a Riemann-Liouville fractional Brownian motion: for H < 1/2 its paths are rougher than a
Brownian motion's, and the at-the-money skew it produces falls like T^(H - 1/2) as the maturity
shrinks, the power law seen in index option markets. xi0(t) = E[V_t] is the forward variance curve,
taken from the market (variance swaps), so only H, eta and rho are free.

There's no pricing formula, so it's simulated with the hybrid scheme of Bennedsen, Lunde and
Pakkanen (2017) with kappa = 1, as in McCrickerd and Pakkanen (2018): the integral over the most
recent step is drawn exactly (jointly with that step's Brownian increment), and the rest is a
Riemann sum evaluated at optimally chosen points, computed for every path at once with an FFT
convolution.

The spot is simulated relative to its forward, X_t = S_t / F(t), which is a martingale starting at 1.
That keeps the simulation independent of rates and dividends: a call is D(T) F(T) E[(X_T - K/F(T))^+].

The random numbers are drawn once (`draw`) and turned into paths for any parameters (`paths`), so a
calibration sees a smooth function of H, eta and rho rather than fresh noise at every step.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

STEPS_PER_DAY = 4  # 6-hour steps; calendar-day maturities land exactly on the grid


@dataclass(frozen=True)
class RoughBergomiParams:
    H: float  # Hurst exponent: below 0.5 is rough, 0.5 is a classical (Markovian) lognormal vol
    eta: float  # vol of vol
    rho: float  # correlation between spot and vol shocks

    def __post_init__(self):
        if not 0 < self.H <= 0.5 or self.eta < 0 or not -1 < self.rho < 1:
            raise ValueError(f"invalid rough Bergomi parameters: {self}")


@dataclass(frozen=True)
class ForwardVariance:
    """A piecewise-constant forward variance curve: `values[i]` applies up to `times[i]` (years),
    the last value beyond the last time."""

    times: np.ndarray
    values: np.ndarray

    def __call__(self, t) -> np.ndarray:
        i = np.searchsorted(self.times, np.asarray(t, dtype=float), side="left")
        return self.values[np.minimum(i, len(self.values) - 1)]

    @classmethod
    def flat(cls, variance: float) -> "ForwardVariance":
        return cls(np.array([np.inf]), np.array([float(variance)]))

    def total(self, T: float) -> float:
        """Int_0^T xi0(t) dt: the variance-swap total variance to T."""
        edges = np.concatenate([[0.0], np.minimum(self.times, T)])
        widths = np.diff(edges).clip(min=0.0)
        return float(widths @ self.values[: len(widths)] + max(T - self.times[-1], 0.0) * self.values[-1])


@dataclass(frozen=True)
class Normals:
    """Standard normals for `paths`: (n_paths, n_steps) each. With antithetic=True in `draw`, the
    second half of the paths are the first half's mirror images."""

    vol: np.ndarray  # drives W (and so the variance)
    vol_extra: np.ndarray  # second coordinate of the hybrid scheme's exact last-step integral
    spot: np.ndarray  # drives W_perp
    dt: float


def draw(n_paths: int, T: float, rng=None, antithetic=True, steps_per_day: int = STEPS_PER_DAY) -> Normals:
    rng = np.random.default_rng(rng)
    n_steps = int(round(T * 365 * steps_per_day))
    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((3, half, n_steps))
    if antithetic:
        z = np.concatenate([z, -z], axis=1)
    return Normals(z[0], z[1], z[2], 1.0 / (365 * steps_per_day))


def _kernel_weights(a: float, n_steps: int, dt: float) -> np.ndarray:
    """Riemann-sum weights g(b_k * dt) for k >= 2 (zero for k = 0, 1), b_k the optimal points."""
    k = np.arange(n_steps + 1, dtype=float)
    weights = np.zeros(n_steps + 1)
    if a == 0.0:  # H = 1/2: the kernel is 1
        weights[2:] = 1.0
        return weights
    kk = k[2:]
    b = ((kk ** (a + 1) - (kk - 1) ** (a + 1)) / (a + 1)) ** (1 / a)
    weights[2:] = (b * dt) ** a
    return weights


@dataclass(frozen=True)
class Paths:
    times: np.ndarray  # (n_steps + 1,)
    log_x: np.ndarray  # log(S_t / F(t)), (n_paths, n_steps + 1), starts at 0
    variance: np.ndarray  # V_t, (n_paths, n_steps + 1)


def volterra(normals: Normals, H: float) -> tuple[np.ndarray, np.ndarray]:
    """The Brownian increments dW and the Volterra process Y on the grid (Y_0 = 0)."""
    a, dt = H - 0.5, normals.dt
    n_paths, n_steps = normals.vol.shape
    # Joint law of (dW over a step, Int over that step of (t_i - s)^a dW_s), factored by hand: at
    # H = 1/2 the two are the same variable and the covariance matrix is singular.
    c00, c01, c11 = dt, dt ** (a + 1) / (a + 1), dt ** (2 * a + 1) / (2 * a + 1)
    l00 = np.sqrt(c00)
    l10 = c01 / l00
    l11 = np.sqrt(max(c11 - l10**2, 0.0))
    dW = l00 * normals.vol
    exact = l10 * normals.vol + l11 * normals.vol_extra
    riemann = fftconvolve(dW, _kernel_weights(a, n_steps, dt)[None, :], axes=1)[:, : n_steps + 1]
    Y = np.zeros((n_paths, n_steps + 1))
    Y[:, 1:] = exact
    Y += riemann
    return dW, np.sqrt(2 * a + 1) * Y


def paths(
    normals: Normals, params: RoughBergomiParams, xi: ForwardVariance, cache: dict | None = None
) -> Paths:
    """Spot and variance paths. Pass the same dict as `cache` across calls with the same normals
    (a calibration does) to reuse the Volterra process while H is unchanged and the variance
    factor while H and eta are, which are the expensive parts."""
    dt = normals.dt
    n_steps = normals.vol.shape[1]
    t = np.arange(n_steps + 1) * dt
    cache = {} if cache is None else cache
    if cache.get("H") != params.H:
        cache.clear()
        cache["H"] = params.H
        cache["dW"], cache["Y"] = volterra(normals, params.H)
    dW, Y = cache["dW"], cache["Y"]
    if cache.get("eta") != params.eta:
        cache["eta"] = params.eta
        cache["E"] = np.exp(params.eta * Y - 0.5 * params.eta**2 * t[None, :] ** (2 * params.H))
    V = xi(t)[None, :] * cache["E"]
    dB = params.rho * dW + np.sqrt(1 - params.rho**2) * np.sqrt(dt) * normals.spot
    increments = np.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * dt
    log_x = np.zeros_like(V)
    np.cumsum(increments, axis=1, out=log_x[:, 1:])
    return Paths(t, log_x, V)


def call_prices(log_x_T: np.ndarray, moneyness) -> np.ndarray:
    """E[(X_T - m)^+] for each m = K / F(T) (undiscounted, per unit forward)."""
    x = np.exp(log_x_T)
    x /= x.mean()  # martingale correction: E[X_T] = 1 exactly, so put-call parity holds exactly
    m = np.atleast_1d(np.asarray(moneyness, dtype=float))
    # Sort once and use cumulative sums: E[(X - m)^+] = (sum of x above m - m * count above m) / n.
    xs = np.sort(x)
    tail = np.concatenate([np.cumsum(xs[::-1])[::-1], [0.0]])
    idx = np.searchsorted(xs, m, side="right")
    return (tail[idx] - m * (len(xs) - idx)) / len(xs)


def price(F, K, T: float, r: float, sim: Paths, kind="call") -> np.ndarray:
    """European prices at maturity T (which must be on the simulation grid) for strikes K."""
    i = int(round(T / (sim.times[1] - sim.times[0])))
    if abs(sim.times[i] - T) > 1e-9:
        raise ValueError(f"maturity {T} is not on the simulation grid")
    K, kind = np.broadcast_arrays(np.asarray(K, dtype=float), np.asarray(kind))
    m = K.ravel() / F
    calls = call_prices(sim.log_x[:, i], m).reshape(K.shape)
    puts = calls - (1 - m.reshape(K.shape))
    return (np.exp(-r * T) * F * np.where(kind == "call", calls, puts))[()]
