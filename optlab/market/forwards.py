"""The forward price for each expiry, inferred from the options themselves.

Put-call parity for European options says C - P = DF * (F - K), so every strike where both the
call and the put trade gives an estimate F = K + (C - P) / DF. We take the median over the strikes
nearest the money (where both legs are most liquid). On 1 Oct 2026 this agreed with the NIFTY,
BANKNIFTY and RELIANCE futures to within 1-11 basis points.

When too few strikes have both legs traded, the matching future's close is used instead; with
neither, the expiry has no forward and its options are dropped.
"""

import numpy as np
import pandas as pd

from optlab import config


def parity_forward(options: pd.DataFrame, T: float, r: float, min_trades: int = config.MIN_TRADES) -> float:
    """options: one expiry's rows with strike, option_type, close, n_trades."""
    liquid = options[(options["n_trades"] >= min_trades) & (options["close"] > 0)]
    calls = liquid[liquid["option_type"] == "call"].set_index("strike")["close"]
    puts = liquid[liquid["option_type"] == "put"].set_index("strike")["close"]
    pairs = pd.concat({"call": calls, "put": puts}, axis=1, join="inner")
    if len(pairs) < 2:
        return np.nan
    diff = pairs["call"] - pairs["put"]
    nearest = diff.abs().nsmallest(config.PARITY_STRIKES).index
    estimates = nearest.to_numpy(dtype=float) + diff.loc[nearest].to_numpy() * np.exp(r * T)
    return float(np.median(estimates))


def forwards(day: pd.DataFrame, r: float) -> pd.DataFrame:
    """One row per option expiry of a single underlying on one trade date.

    Columns: expiry, T (years), forward, forward_source ("parity" | "future"), future_close.
    """
    trade_date = day["trade_date"].iloc[0]
    futures = day[day["instrument"] == "future"].set_index("expiry")["close"]
    rows = []
    for expiry, options in day[day["instrument"] == "option"].groupby("expiry"):
        T = (expiry - trade_date).days / 365
        if T <= 0:
            continue
        future = futures.get(expiry, np.nan)
        forward, source = parity_forward(options, T, r), "parity"
        if np.isnan(forward) and future > 0:
            forward, source = float(future), "future"
        if not np.isnan(forward):
            rows.append((expiry, T, forward, source, future))
    return pd.DataFrame(rows, columns=["expiry", "T", "forward", "forward_source", "future_close"])
