"""Binomial trees for European and American options.

The default is Cox-Ross-Rubinstein (u = e^(sigma sqrt(dt)), d = 1/u). CRR needs a risk-neutral
probability inside (0, 1), which fails when the drift per step outruns the volatility (low vol, high
rates, few steps, long maturities). Then the tree switches to a drift-centred lattice,
u, d = e^((r - q - sigma^2/2) dt +/- sigma sqrt(dt)), whose probability is always inside (0, 1).
Both converge to Black-Scholes; the error shrinks roughly like 1/steps.

Each backward step is one numpy operation over the nodes at that time, so a 2,000-step tree takes a
few milliseconds.
"""

from dataclasses import dataclass

import numpy as np

from optlab.contracts import OptionSpec

DEFAULT_STEPS = 500


@dataclass(frozen=True)
class Lattice:
    u: float
    d: float
    p: float
    discount: float
    dt: float
    kind: str  # "crr" | "drift-centred"


def lattice(T: float, r: float, sigma: float, q: float, steps: int) -> Lattice:
    dt = T / steps
    growth = np.exp((r - q) * dt)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    kind = "crr"
    if not d < growth < u:  # CRR's probability would fall outside (0, 1)
        centre = (r - q - 0.5 * sigma**2) * dt
        u, d = np.exp(centre + sigma * np.sqrt(dt)), np.exp(centre - sigma * np.sqrt(dt))
        kind = "drift-centred"
    p = (growth - d) / (u - d)
    return Lattice(float(u), float(d), float(p), float(np.exp(-r * dt)), dt, kind)


def _node_prices(S: float, lat: Lattice, i: int) -> np.ndarray:
    """Spot at the i + 1 nodes of step i, highest first: S * u**(i - j) * d**j."""
    j = np.arange(i + 1)
    return S * lat.u ** (i - j) * lat.d**j


def _roll_back(spec: OptionSpec, steps: int) -> tuple[float, np.ndarray, np.ndarray, Lattice]:
    """The option value now, plus the node values at steps 1 and 2 (for delta and gamma)."""
    lat = lattice(spec.T, spec.r, spec.sigma, spec.q, steps)
    values = spec.payoff(_node_prices(spec.S, lat, steps))
    kept = {}
    for i in range(steps - 1, -1, -1):
        values = lat.discount * (lat.p * values[:-1] + (1.0 - lat.p) * values[1:])
        if spec.style == "american":
            values = np.maximum(values, spec.payoff(_node_prices(spec.S, lat, i)))
        if i in (1, 2):
            kept[i] = values.copy()
    return float(values[0]), kept.get(1), kept.get(2), lat


def price(S, K, T, r, sigma, q=0.0, kind="call", style="european", steps=DEFAULT_STEPS) -> float:
    spec = OptionSpec(S, K, T, r, sigma, q, kind, style)
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if T == 0:
        return float(spec.payoff(S))
    if sigma == 0:
        return _deterministic(spec, steps)
    return _roll_back(spec, steps)[0]


def tree_delta_gamma(spec: OptionSpec, steps: int = DEFAULT_STEPS) -> tuple[float, float]:
    """Delta and gamma read off the tree's first two steps (Hull's method).

    Bumping spot by less than the node spacing and re-running the tree doesn't work: the tree price
    is piecewise linear in S at that scale, so a second difference comes out as 0 or explodes.
    """
    if steps < 2 or spec.T == 0 or spec.sigma == 0:
        raise ValueError("tree Greeks need at least 2 steps, time to expiry and volatility")
    _, step1, step2, lat = _roll_back(spec, steps)
    s1, s2 = _node_prices(spec.S, lat, 1), _node_prices(spec.S, lat, 2)
    delta = (step1[0] - step1[1]) / (s1[0] - s1[1])
    upper = (step2[0] - step2[1]) / (s2[0] - s2[1])
    lower = (step2[1] - step2[2]) / (s2[1] - s2[2])
    gamma = (upper - lower) / (0.5 * (s2[0] - s2[2]))
    return float(delta), float(gamma)


def _deterministic(spec: OptionSpec, steps: int) -> float:
    """With zero volatility the spot follows S * exp((r - q) t) for sure.

    A European option is worth its discounted payoff at expiry; an American one is worth the best
    discounted payoff over the exercise dates the tree would have.
    """
    t = np.linspace(0.0, spec.T, steps + 1)
    discounted = np.exp(-spec.r * t) * spec.payoff(spec.S * np.exp((spec.r - spec.q) * t))
    return float(discounted.max() if spec.style == "american" else discounted[-1])


def price_spec(spec: OptionSpec, steps: int = DEFAULT_STEPS) -> float:
    return price(spec.S, spec.K, spec.T, spec.r, spec.sigma, spec.q, spec.kind, spec.style, steps)
