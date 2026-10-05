"""Greeks by finite differences, for any pricer that maps an OptionSpec to a price.

Central differences throughout, with the same conventions as black_scholes: vega and rho per 1.00,
theta per year. For a Monte Carlo pricer, pass one with a fixed seed so both sides of each bump see
the same random numbers; otherwise the noise swamps the difference.
"""

from collections.abc import Callable

from optlab.contracts import OptionSpec

Pricer = Callable[[OptionSpec], float]


def finite_difference(
    pricer: Pricer, spec: OptionSpec, spot_bump=1e-3, vol_bump=1e-3, rate_bump=1e-4, time_bump=1 / 365
) -> dict:
    """spot_bump is relative to S; the other bumps are absolute."""
    dS = spec.S * spot_bump
    up, mid, down = pricer(spec.bump(S=spec.S + dS)), pricer(spec), pricer(spec.bump(S=spec.S - dS))

    vol_down = min(vol_bump, spec.sigma)  # one-sided at sigma = 0
    vega = (
        pricer(spec.bump(sigma=spec.sigma + vol_bump)) - pricer(spec.bump(sigma=spec.sigma - vol_down))
    ) / (vol_bump + vol_down)

    rho = (pricer(spec.bump(r=spec.r + rate_bump)) - pricer(spec.bump(r=spec.r - rate_bump))) / (
        2 * rate_bump
    )

    # Theta is the change as time passes, i.e. as T shrinks.
    shorter = min(time_bump, spec.T)
    theta = (pricer(spec.bump(T=spec.T - shorter)) - pricer(spec.bump(T=spec.T + time_bump))) / (
        shorter + time_bump
    )

    return {
        "delta": (up - down) / (2 * dS),
        "gamma": (up - 2 * mid + down) / dS**2,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
