"""Cox-Ross-Rubinstein binomial tree for European and American options.

Each backward step is one numpy operation over the nodes at that time, so a 2,000-step tree takes a
few milliseconds. The tree converges to Black-Scholes for European options with an error that
shrinks roughly like 1/steps, oscillating between odd and even step counts.
"""

import numpy as np

from optlab.contracts import OptionSpec

DEFAULT_STEPS = 500


def price(S, K, T, r, sigma, q=0.0, kind="call", style="european", steps=DEFAULT_STEPS) -> float:
    spec = OptionSpec(S, K, T, r, sigma, q, kind, style)
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if T == 0:
        return float(spec.payoff(S))
    if sigma == 0:
        return _deterministic(spec, steps)

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"risk-neutral probability {p:.4f} is outside (0, 1); use more steps "
            "(the drift per step is too large for this volatility)"
        )
    discount = np.exp(-r * dt)

    # Node j at step i has price S * u**(i - 2j), j = 0 (highest) .. i (lowest).
    values = spec.payoff(S * u ** np.arange(steps, -steps - 1, -2, dtype=float))
    for i in range(steps - 1, -1, -1):
        values = discount * (p * values[:-1] + (1.0 - p) * values[1:])
        if style == "american":
            values = np.maximum(values, spec.payoff(S * u ** np.arange(i, -i - 1, -2, dtype=float)))
    return float(values[0])


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
