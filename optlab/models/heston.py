"""The Heston (1993) stochastic-volatility model.

    dS/S = (r - q) dt + sqrt(v) dW1
    dv   = kappa (theta - v) dt + xi sqrt(v) dW2,     d<W1, W2> = rho dt

Prices come from the characteristic function of ln(S_T / F) in the "little Heston trap" form of
Albrecher et al. (2007), which stays on the right branch of the complex logarithm for long
maturities, through Lewis's (2001) single-integral formula:

    C = D [F - sqrt(F K) / pi * Int_0^inf Re(e^{i u x} phi(u - i/2)) / (u^2 + 1/4) du],  x = ln(F/K)

The integral is truncated where the integrand has decayed below 1e-12 (found by probing, since
Heston's characteristic function decays like a Gaussian for short maturities and only
exponentially for long ones) and done with 512-point Gauss-Legendre, once per maturity for all
strikes at once. Puts follow from put-call parity.

`HestonTermParams` lets the long-run variance theta step between given times (one level per
expiry, say), with kappa, xi and rho constant. The characteristic function stays in closed form:
D doesn't involve theta, and C = kappa * Int_0^T theta(s) D(T - s) ds, which for steps is a sum of
I(T - a) - I(T - b) over them, I(tau) being Int_0^tau D, i.e. constant-theta Heston's C / (kappa theta).

`simulate` is a full-truncation Euler Monte Carlo (Lord, Koekkoek and van Dijk, 2010), used to
check the pricer and to show it converging.
"""

from dataclasses import dataclass

import numpy as np

NODES, WEIGHTS = np.polynomial.legendre.leggauss(512)
_PROBE = np.geomspace(1.0, 1e5, 80)
TOLERANCE = 1e-12


@dataclass(frozen=True)
class HestonParams:
    v0: float  # current variance (sqrt(v0) is today's instantaneous vol)
    kappa: float  # speed of mean reversion
    theta: float  # long-run variance
    xi: float  # volatility of variance ("vol of vol")
    rho: float  # correlation between spot and variance shocks

    def __post_init__(self):
        if min(self.v0, self.theta) < 0 or self.kappa <= 0 or self.xi <= 0 or not -1 < self.rho < 1:
            raise ValueError(f"invalid Heston parameters: {self}")

    @property
    def feller_ratio(self) -> float:
        """2 kappa theta / xi^2. Above 1, the variance process never touches zero."""
        return 2 * self.kappa * self.theta / self.xi**2

    def as_array(self) -> np.ndarray:
        return np.array([self.v0, self.kappa, self.theta, self.xi, self.rho])

    def theta_at(self, t: float) -> float:
        """The long-run variance in force at time t."""
        return self.theta


@dataclass(frozen=True)
class HestonTermParams(HestonParams):
    """Heston whose long-run variance steps: thetas[i] holds on (times[i-1], times[i]], from time 0
    for the first, and the last one goes on after times[-1]. `theta` is that last one."""

    times: tuple[float, ...]
    thetas: tuple[float, ...]

    @classmethod
    def build(cls, v0, kappa, xi, rho, times, thetas) -> "HestonTermParams":
        times, thetas = tuple(float(t) for t in times), tuple(float(t) for t in thetas)
        return cls(v0, kappa, thetas[-1] if thetas else 0.0, xi, rho, times, thetas)

    def __post_init__(self):
        super().__post_init__()
        if not self.times or len(self.times) != len(self.thetas) or min(self.thetas) < 0:
            raise ValueError(f"need one non-negative theta per step time: {self}")
        if self.times[0] <= 0 or np.any(np.diff(self.times) <= 0):
            raise ValueError(f"step times must be positive and increasing: {self.times}")

    def theta_at(self, t: float) -> float:
        return self.thetas[min(int(np.searchsorted(self.times, t, side="left")), len(self.thetas) - 1)]


def char_fn(u, T: float, p: HestonParams) -> np.ndarray:
    """E[exp(i u ln(S_T / F))] for complex u, in the little-trap form."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    beta = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(beta**2 + p.xi**2 * (iu + u**2))
    g = (beta - d) / (beta + d)

    def integrated_D(tau):  # Int_0^tau D
        return ((beta - d) * tau - 2.0 * np.log((1.0 - g * np.exp(-d * tau)) / (1.0 - g))) / p.xi**2

    if isinstance(p, HestonTermParams):
        starts = (0.0, *p.times[:-1])
        ends = (*p.times[:-1], np.inf)
        C = sum(
            theta * (integrated_D(T - a) - integrated_D(T - min(b, T)))
            for a, b, theta in zip(starts, ends, p.thetas, strict=True)
            if a < T
        )
        C = p.kappa * C
    else:
        C = p.kappa * p.theta * integrated_D(T)
    e = np.exp(-d * T)
    D = (beta - d) / p.xi**2 * (1.0 - e) / (1.0 - g * e)
    return np.exp(C + D * p.v0)


def _lewis_integrand(u, T: float, p: HestonParams) -> np.ndarray:
    return char_fn(u - 0.5j, T, p) / (u**2 + 0.25)


def _cutoff(T: float, p: HestonParams) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        size = np.abs(_lewis_integrand(_PROBE, T, p))
    size = np.where(np.isfinite(size), size, 0.0)
    small = np.nonzero(size < TOLERANCE)[0]
    return float(_PROBE[small[0]] if len(small) else _PROBE[-1])


def lewis_integral(x, T: float, p: HestonParams) -> np.ndarray:
    """Int_0^inf Re(e^{i u x} phi(u - i/2)) / (u^2 + 1/4) du for each log-moneyness x = ln(F/K)."""
    upper = _cutoff(T, p)
    u = 0.5 * upper * (NODES + 1.0)
    w = 0.5 * upper * WEIGHTS
    f = _lewis_integrand(u, T, p)
    ux = np.outer(np.atleast_1d(x), u)
    return (np.cos(ux) * f.real - np.sin(ux) * f.imag) @ w


def price(F, K, T: float, r: float, p: HestonParams, kind="call"):
    """European option prices for one maturity, on the forward F (Black-76 style inputs).

    K and kind broadcast together; scalar in, numpy scalar out.
    """
    K, kind = np.broadcast_arrays(np.asarray(K, dtype=float), np.asarray(kind))
    if not np.isin(kind, ("call", "put")).all():
        raise ValueError(f"kind must be 'call' or 'put', got {np.unique(kind)}")
    discount = np.exp(-r * T)
    if T <= 0:
        call = np.maximum(F - K, 0.0)
    else:
        integral = lewis_integral(np.log(F / K.ravel()), T, p).reshape(K.shape)
        call = F - np.sqrt(F * K) / np.pi * integral
    put = call - (F - K)
    return (discount * np.where(kind == "call", call, put))[()]


def price_spot(S, K, T, r, q, p: HestonParams, kind="call"):
    """Same as `price`, from spot and a continuous dividend yield."""
    return price(S * np.exp((r - q) * T), K, T, r, p, kind)


def simulate(S, T, r, q, p: HestonParams, n_paths=50_000, n_steps=100, rng=None) -> np.ndarray:
    """Terminal spot prices from a full-truncation Euler scheme for the variance.

    The variance can go negative between steps; only its positive part feeds the drift and the
    diffusion, which keeps the scheme's bias small (Lord et al. 2010). The log-spot step is exact
    given the (frozen) variance.
    """
    rng = np.random.default_rng(rng)
    dt = T / n_steps
    log_s = np.full(n_paths, np.log(S))
    v = np.full(n_paths, p.v0)
    tilt = np.sqrt(1.0 - p.rho**2)
    for step in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = p.rho * z1 + tilt * rng.standard_normal(n_paths)
        positive = np.maximum(v, 0.0)
        root = np.sqrt(positive * dt)
        log_s += (r - q - 0.5 * positive) * dt + root * z1
        v += p.kappa * (p.theta_at((step + 0.5) * dt) - positive) * dt + p.xi * root * z2
    return np.exp(log_s)


def simulate_paths(S, T, r, q, p: HestonParams, n_paths, n_fixings, substeps=4, rng=None):
    """Log-spot at n_fixings equally spaced dates (plus today) and the variance integrated over each
    interval, from the same full-truncation scheme as `simulate`, with `substeps` steps per interval.

    Returns (log_s, integrated_variance) with shapes (n_paths, n_fixings + 1) and (n_paths, n_fixings).
    """
    rng = np.random.default_rng(rng)
    dt = T / (n_fixings * substeps)
    log_s = np.empty((n_paths, n_fixings + 1))
    log_s[:, 0] = np.log(S)
    integrated = np.zeros((n_paths, n_fixings))
    x, v = log_s[:, 0].copy(), np.full(n_paths, p.v0)
    tilt = np.sqrt(1.0 - p.rho**2)
    for i in range(n_fixings):
        for j in range(substeps):
            z1 = rng.standard_normal(n_paths)
            z2 = p.rho * z1 + tilt * rng.standard_normal(n_paths)
            positive = np.maximum(v, 0.0)
            root = np.sqrt(positive * dt)
            x += (r - q - 0.5 * positive) * dt + root * z1
            v += p.kappa * (p.theta_at((i * substeps + j + 0.5) * dt) - positive) * dt + p.xi * root * z2
            integrated[:, i] += positive * dt
        log_s[:, i + 1] = x
    return log_s, integrated
