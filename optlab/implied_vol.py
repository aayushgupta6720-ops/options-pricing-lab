"""Implied volatility: the sigma at which Black-Scholes reproduces a market price.

Solved for a whole chain at once with a bracketed Newton method: each element keeps a [lo, hi]
bracket that always contains the root, takes a Newton step when it lands inside the bracket and
bisects otherwise. That converges as fast as Newton near the root and can't diverge the way plain
Newton does for deep in- or out-of-the-money options, where vega is close to zero.

Prices outside the no-arbitrage range [BS(sigma_min), BS(sigma_max)] have no implied vol and
come back as NaN.
"""

import numpy as np

from optlab.models import black_scholes as bs

SIGMA_MIN, SIGMA_MAX = 1e-4, 5.0


def implied_vol(price, S, K, T, r, q=0.0, kind="call", tol=1e-10, max_iter=100):
    price, S, K, T, r, q, kind = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (price, S, K, T, r, q)), np.asarray(kind)
    )
    args = (S, K, T, r)

    lo = np.full(price.shape, SIGMA_MIN)
    hi = np.full(price.shape, SIGMA_MAX)
    valid = (T > 0) & np.isfinite(price)
    with np.errstate(invalid="ignore"):
        valid &= (price >= bs.price(*args, lo, q, kind)) & (price <= bs.price(*args, hi, q, kind))

    # Start at the Brenner-Subrahmanyam at-the-money approximation, kept inside the bracket.
    with np.errstate(divide="ignore", invalid="ignore"):
        guess = np.sqrt(2 * np.pi / T) * price / S
    sigma = np.clip(np.where(np.isfinite(guess), guess, 0.2), SIGMA_MIN, SIGMA_MAX)

    todo = valid.copy()
    for _ in range(max_iter):
        if not todo.any():
            break
        diff = np.where(todo, bs.price(*args, sigma, q, kind) - price, 0.0)
        todo &= np.abs(diff) > tol * np.maximum(1.0, price)
        hi = np.where(todo & (diff > 0), sigma, hi)
        lo = np.where(todo & (diff < 0), sigma, lo)
        vega = bs.vega(*args, sigma, q, kind)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            newton = sigma - diff / vega
        inside = (newton > lo) & (newton < hi) & np.isfinite(newton)
        sigma = np.where(todo, np.where(inside, newton, 0.5 * (lo + hi)), sigma)
        todo &= (hi - lo) > 1e-14

    return np.where(valid, sigma, np.nan)[()]
