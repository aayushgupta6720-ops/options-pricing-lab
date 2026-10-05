"""The SABR model's implied volatility (Hagan, Kumar, Lesniewski and Woodward, 2002).

    dF = alpha_t F^beta dW1,   d alpha_t = nu alpha_t dW2,   d<W1, W2> = rho dt

SABR describes one expiry's smile with three numbers: alpha (the vol level), rho (the skew) and
nu (the vol of vol, which curves the wings). Hagan's asymptotic formula gives the Black implied
vol directly, so fitting a smile needs no option pricing at all.

beta (the "backbone") is fixed rather than fitted: on a single day's smile it is nearly
indistinguishable from rho. The default beta = 1 makes F lognormal, so alpha is close to the
at-the-money vol.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SabrParams:
    alpha: float
    rho: float
    nu: float
    beta: float = 1.0

    def __post_init__(self):
        if self.alpha <= 0 or self.nu < 0 or not -1 < self.rho < 1 or not 0 <= self.beta <= 1:
            raise ValueError(f"invalid SABR parameters: {self}")


def implied_vol(F: float, K, T: float, p: SabrParams) -> np.ndarray:
    """Black (lognormal) implied vol at strikes K for one expiry; scalar in, numpy scalar out."""
    K = np.asarray(K, dtype=float)
    one_minus_beta = 1.0 - p.beta
    log_fk = np.log(F / K)
    fk_power = (F * K) ** (0.5 * one_minus_beta)  # (FK)^((1 - beta) / 2)

    denominator = fk_power * (1 + one_minus_beta**2 / 24 * log_fk**2 + one_minus_beta**4 / 1920 * log_fk**4)
    z = p.nu / p.alpha * fk_power * log_fk
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log((np.sqrt(1 - 2 * p.rho * z + z**2) + z - p.rho) / (1 - p.rho))
        ratio = np.where(np.abs(z) > 1e-7, z / x, 1.0 - 0.5 * p.rho * z)  # z / x(z) -> 1 at the money
    correction = 1 + T * (
        one_minus_beta**2 / 24 * p.alpha**2 / fk_power**2
        + p.rho * p.beta * p.nu * p.alpha / (4 * fk_power)
        + (2 - 3 * p.rho**2) / 24 * p.nu**2
    )
    return (p.alpha / denominator * ratio * correction)[()]
