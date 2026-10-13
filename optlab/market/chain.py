"""From one day's raw bhavcopy rows to a clean implied-volatility chain.

For each expiry: infer the forward (forwards.py), keep only out-of-the-money options (puts below
the forward, calls at or above it: they're the liquid side, and their price is all time value),
drop quotes that aren't real prices or that leave a static arbitrage, and invert Black-76 for the
implied vol. NSE options are European, so this inversion is exact rather than an approximation.
"""

import numpy as np
import pandas as pd
from scipy.special import ndtr

from optlab import config
from optlab.implied_vol import implied_vol
from optlab.market.forwards import forwards

CHAIN_COLUMNS = [
    "trade_date", "underlying", "expiry", "T", "strike", "option_type", "close", "volume", "n_trades",
    "open_interest", "lot_size", "spot", "forward", "forward_source", "r", "log_moneyness", "iv", "delta",
]  # fmt: skip


def build_chain(day: pd.DataFrame, underlying: str, r: float = config.RISK_FREE_RATE) -> pd.DataFrame:
    """Implied-vol chain for one underlying, from nse.parse() output for a single trade date."""
    rows = day[day["underlying"] == underlying]
    options = rows[rows["instrument"] == "option"]
    if options.empty:
        return pd.DataFrame(columns=CHAIN_COLUMNS)

    fwd = forwards(rows, r)
    chain = options.merge(fwd, on="expiry", how="inner")
    chain["spot"] = chain["underlying_price"]
    chain["r"] = r
    chain["log_moneyness"] = np.log(chain["strike"] / chain["forward"])

    otm = np.where(
        chain["option_type"] == "put", chain["strike"] < chain["forward"], chain["strike"] >= chain["forward"]
    )
    keep = (
        otm
        & (chain["T"] * 365 >= config.MIN_DAYS_TO_EXPIRY)
        & (chain["n_trades"] >= config.MIN_TRADES)
        & (chain["close"] >= config.MIN_PRICE)
        & (chain["log_moneyness"].abs() <= config.MAX_ABS_LOG_MONEYNESS)
    )
    chain = chain[keep].copy()
    chain = chain[arbitrage_free(chain)]
    if chain.empty:
        return pd.DataFrame(columns=CHAIN_COLUMNS)

    # Black-76 is Black-Scholes on the forward with q = r.
    F, K, T = chain["forward"].to_numpy(), chain["strike"].to_numpy(), chain["T"].to_numpy()
    chain["iv"] = implied_vol(chain["close"].to_numpy(), F, K, T, r, r, chain["option_type"].to_numpy())
    chain = chain[chain["iv"].notna()]
    if chain.empty:
        return pd.DataFrame(columns=CHAIN_COLUMNS)

    # Forward (undiscounted) delta, the convention for quoting 25-delta skew.
    d1 = (-chain["log_moneyness"] + 0.5 * chain["iv"] ** 2 * chain["T"]) / (chain["iv"] * np.sqrt(chain["T"]))
    chain["delta"] = np.where(chain["option_type"] == "call", ndtr(d1), ndtr(d1) - 1.0)

    return chain[CHAIN_COLUMNS].sort_values(["expiry", "strike"]).reset_index(drop=True)


def _worst_violation(K, c, quoted, discount, tolerance, share):
    """The largest static-arbitrage violation beyond tolerance among neighbouring strikes, as
    (excess in rupees, positions involved), or (0, ()) if there's none.

    `c` are call prices for every strike (puts by put-call parity on the forward), so the checks are
    the ones a call price curve must pass: it falls with the strike (a call spread costs nothing
    or more), no faster than the discount factor (a call spread costs less than its width), and it's
    convex (a butterfly costs nothing or more). The tolerance is on the quoted prices involved."""
    worst = (0.0, ())
    for i in range(len(K) - 1):
        allowed = max(tolerance, share * max(quoted[i], quoted[i + 1]))
        for excess in (c[i + 1] - c[i], c[i] - c[i + 1] - discount * (K[i + 1] - K[i])):
            if excess - allowed > worst[0]:
                worst = (excess - allowed, (i, i + 1))
    for i in range(len(K) - 2):
        w = (K[i + 2] - K[i + 1]) / (K[i + 2] - K[i])
        allowed = max(tolerance, share * max(quoted[i], quoted[i + 1], quoted[i + 2]))
        excess = c[i + 1] - (w * c[i] + (1 - w) * c[i + 2])
        if excess - allowed > worst[0]:
            worst = (excess - allowed, (i, i + 1, i + 2))
    return worst


def arbitrage_free(
    chain: pd.DataFrame,
    tolerance: float = config.ARBITRAGE_TOLERANCE,
    share: float = config.ARBITRAGE_TOLERANCE_SHARE,
) -> np.ndarray:
    """Which quotes to keep so no expiry's prices leave a vertical-spread or butterfly arbitrage
    beyond the tolerance (see config.ARBITRAGE_TOLERANCE). Needs strike, option_type, close,
    n_trades, forward, T and r. While a violation remains, the least-traded quote in the worst one
    goes: a stale close is the thin one, never the strike with thousands of trades."""
    keep = np.ones(len(chain), dtype=bool)
    if chain.empty:
        return keep
    positions = np.arange(len(chain))
    for _, q in chain.assign(_pos=positions).groupby("expiry", sort=False):
        q = q.sort_values("strike")
        F, T, r = float(q["forward"].iloc[0]), float(q["T"].iloc[0]), float(q["r"].iloc[0])
        discount = np.exp(-r * T)
        K = q["strike"].to_numpy(dtype=float)
        quoted = q["close"].to_numpy(dtype=float)
        c = np.where(q["option_type"].to_numpy() == "call", quoted, quoted + discount * (F - K))
        trades = q["n_trades"].to_numpy()
        live = np.ones(len(q), dtype=bool)
        while live.sum() >= 2:
            idx = np.flatnonzero(live)
            excess, involved = _worst_violation(K[idx], c[idx], quoted[idx], discount, tolerance, share)
            if not involved:
                break
            live[min((idx[j] for j in involved), key=lambda j: trades[j])] = False
        keep[q["_pos"].to_numpy()[~live]] = False
    return keep
