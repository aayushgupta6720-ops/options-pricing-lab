"""From one day's raw bhavcopy rows to a clean implied-volatility chain.

For each expiry: infer the forward (forwards.py), keep only out-of-the-money options (puts below
the forward, calls at or above it: they're the liquid side, and their price is all time value),
drop quotes that aren't real prices, and invert Black-76 for the implied vol. NSE options are
European, so this inversion is exact rather than an approximation.
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
