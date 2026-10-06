"""Variance swaps and the forward variance curve, from out-of-the-money option prices.

A variance swap's fair strike is replicated by a strip of out-of-the-money options weighted by
1 / K^2 (the log contract):

    sigma_VS^2 T = 2 e^{rT} [ Int_0^F P(K) / K^2 dK + Int_F^inf C(K) / K^2 dK ]

This is the formula behind the VIX and India VIX. It's discretised the same way, with each strike
weighted by half the distance to its neighbours. Because the chain already splits puts and calls
at the forward, no correction term for an off-forward central strike is needed. The strip only
covers the quoted strikes, so far-wing truncation makes it slightly low.

The forward variance curve xi0(t) (the expected instantaneous variance at each future date) is the
slope of total variance sigma_VS^2 T in T. Total variance must not fall with maturity, so noisy
estimates are first made monotone (pool-adjacent-violators), then differenced between expiries.
"""

import numpy as np
import pandas as pd

from optlab.models.rough_bergomi import ForwardVariance

MIN_QUOTES = 8  # per expiry, with both puts and calls
MIN_FORWARD_VARIANCE = 1e-4  # a 1% vol floor between expiries


def variance_swap(quotes: pd.DataFrame) -> float:
    """Annualised fair variance for one expiry's out-of-the-money quotes (chain.py columns)."""
    q = quotes.sort_values("strike")
    if len(q) < MIN_QUOTES or not {"put", "call"} <= set(q["option_type"]):
        return np.nan
    K, price = q["strike"].to_numpy(), q["close"].to_numpy()
    T, r = float(q["T"].iloc[0]), float(q["r"].iloc[0])
    dK = np.gradient(K)  # half the distance to each neighbour; one side at the ends
    return float(2 / T * np.exp(r * T) * np.sum(dK / K**2 * price))


def monotone(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted least-squares non-decreasing fit (pool adjacent violators)."""
    blocks = [[v, w, 1] for v, w in zip(values, weights, strict=True)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v1, w1, n1 = blocks[i]
            v2, w2, n2 = blocks[i + 1]
            blocks[i] = [(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[v] * n for v, _, n in blocks])


def term_structure(chain: pd.DataFrame, max_days: float = np.inf) -> pd.DataFrame:
    """Per expiry: T, the variance-swap vol, and the monotone total variance used for xi0."""
    rows = []
    for expiry, q in chain.groupby("expiry"):
        T = float(q["T"].iloc[0])
        if T * 365 <= max_days:
            rows.append({"expiry": expiry, "T": T, "vs_var": variance_swap(q), "n_quotes": len(q)})
    table = pd.DataFrame(rows, columns=["expiry", "T", "vs_var", "n_quotes"]).dropna().sort_values("T")
    table["total"] = monotone((table["vs_var"] * table["T"]).to_numpy(), table["n_quotes"].to_numpy(float))
    table["vs_vol"] = np.sqrt(table["vs_var"])
    return table.reset_index(drop=True)


def forward_variance(chain: pd.DataFrame, max_days: float = np.inf) -> ForwardVariance | None:
    """Piecewise-constant xi0(t) between expiries; None if no expiry has a usable strip."""
    table = term_structure(chain, max_days)
    if table.empty:
        return None
    T, total = table["T"].to_numpy(), table["total"].to_numpy()
    widths = np.diff(np.concatenate([[0.0], T]))
    steps = np.diff(np.concatenate([[0.0], total]))
    values = np.maximum(steps / widths, MIN_FORWARD_VARIANCE)
    return ForwardVariance(T, values)


def vs_vol_at(table: pd.DataFrame, days: float) -> float:
    """Variance-swap vol at a fixed tenor, interpolating total variance linearly in time (as the
    VIX does between its two expiries); NaN outside the listed expiries."""
    T = days / 365
    if table.empty or not table["T"].iloc[0] <= T <= table["T"].iloc[-1]:
        return np.nan
    return float(np.sqrt(np.interp(T, table["T"], table["vs_var"] * table["T"]) / T))
