"""Monte Carlo pricing under geometric Brownian motion.

The engine is split in two so exotic options can reuse it: `gbm_paths` simulates spot paths, and
`price_payoff` turns any payoff function of those paths into a price with its standard error.
A European option only depends on the final spot, so `price` samples that directly (one exact
step) instead of simulating a whole path.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optlab.contracts import OptionSpec

DEFAULT_PATHS = 100_000
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - Z_95 * self.std_error, self.price + Z_95 * self.std_error


def gbm_paths(S, T, r, sigma, q=0.0, n_paths=DEFAULT_PATHS, n_steps=1, rng=None) -> np.ndarray:
    """Spot paths, shape (n_paths, n_steps + 1), with column 0 equal to S.

    Each step uses the exact log-normal transition, so there is no discretisation error at the
    grid times; n_steps only matters for payoffs that look at the path in between.
    """
    rng = np.random.default_rng(rng)
    dt = T / n_steps
    shocks = rng.standard_normal((n_paths, n_steps))
    log_steps = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=1)], axis=1)
    return S * np.exp(log_paths)


def price_payoff(
    payoff: Callable[[np.ndarray], np.ndarray], paths: np.ndarray, r: float, T: float
) -> MCResult:
    """Discounted mean of payoff(paths); payoff returns one value per path (row)."""
    discounted = np.exp(-r * T) * np.asarray(payoff(paths), dtype=float)
    n = discounted.size
    return MCResult(float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n)), n)


def price(S, K, T, r, sigma, q=0.0, kind="call", n_paths=DEFAULT_PATHS, seed=None) -> MCResult:
    spec = OptionSpec(S, K, T, r, sigma, q, kind)
    paths = gbm_paths(S, T, r, sigma, q, n_paths=n_paths, n_steps=1, rng=seed)
    return price_payoff(lambda p: spec.payoff(p[:, -1]), paths, r, T)


def price_spec(spec: OptionSpec, n_paths: int = DEFAULT_PATHS, seed=None) -> MCResult:
    if spec.style != "european":
        raise ValueError("this Monte Carlo pricer handles European options only")
    return price(spec.S, spec.K, spec.T, spec.r, spec.sigma, spec.q, spec.kind, n_paths, seed)
