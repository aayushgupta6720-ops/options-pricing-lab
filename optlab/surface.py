"""Summary numbers and grids from an implied-vol chain (market/chain.py output).

Per expiry: at-the-money vol (interpolated at log-moneyness 0) and the 25-delta put and call vols.
Across expiries: values at fixed tenors, interpolating ATM *total variance* (sigma^2 * T) linearly
in time, which is the standard way to keep the term structure free of calendar arbitrage. Outside
the listed expiries the vol is held flat, but only up to EXTRAPOLATION_DAYS away; further out it's
NaN (a monthly-only underlying has no honest 7-day vol).
"""

import numpy as np
import pandas as pd

from optlab import config

EXTRAPOLATION_DAYS = 14


def _interp_inside(x_new, x, y):
    """Linear interpolation that returns NaN outside the data instead of clamping."""
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    if len(x) == 0 or not x[0] <= x_new <= x[-1]:
        return np.nan
    return float(np.interp(x_new, x, y))


METRIC_COLUMNS = ["expiry", "T", "forward", "atm_iv", "put_25d_iv", "call_25d_iv", "skew_25d", "n_quotes"]


def expiry_metrics(chain: pd.DataFrame) -> pd.DataFrame:
    """One row per expiry with the METRIC_COLUMNS."""
    rows = []
    for expiry, g in chain.groupby("expiry"):
        calls, puts = g[g["option_type"] == "call"], g[g["option_type"] == "put"]
        put_25 = _interp_inside(-0.25, puts["delta"], puts["iv"])
        call_25 = _interp_inside(0.25, calls["delta"], calls["iv"])
        rows.append(
            {
                "expiry": expiry,
                "T": g["T"].iloc[0],
                "forward": g["forward"].iloc[0],
                "atm_iv": _interp_inside(0.0, g["log_moneyness"], g["iv"]),
                "put_25d_iv": put_25,
                "call_25d_iv": call_25,
                "skew_25d": put_25 - call_25,
                "n_quotes": len(g),
            }
        )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def _at_tenor(metrics: pd.DataFrame, column: str, days: float, total_variance: bool) -> float:
    m = metrics.dropna(subset=[column]).sort_values("T")
    if m.empty:
        return np.nan
    T, first, last = days / 365, m.iloc[0], m.iloc[-1]
    limit = EXTRAPOLATION_DAYS / 365
    if T <= first["T"]:
        return float(first[column]) if first["T"] - T <= limit else np.nan
    if T >= last["T"]:
        return float(last[column]) if T - last["T"] <= limit else np.nan
    if not total_variance:
        return _interp_inside(T, m["T"], m[column])
    variance = _interp_inside(T, m["T"], m[column] ** 2 * m["T"])
    return float(np.sqrt(variance / T))


def atm_at_tenor(metrics: pd.DataFrame, days: float) -> float:
    return _at_tenor(metrics, "atm_iv", days, total_variance=True)


def skew_at_tenor(metrics: pd.DataFrame, days: float) -> float:
    return _at_tenor(metrics, "skew_25d", days, total_variance=False)


def daily_summary(chain: pd.DataFrame) -> dict:
    """The per-day numbers stored in summary.parquet (without the cross-day ones like realized vol)."""
    metrics = expiry_metrics(chain)
    row = {f"atm_iv_{d}d": atm_at_tenor(metrics, d) for d in config.TENORS}
    row["skew_25d_30d"] = skew_at_tenor(metrics, 30)
    row["n_quotes"] = len(chain)
    row["n_expiries"] = int(metrics["atm_iv"].notna().sum()) if len(metrics) else 0
    return row


def smile_grid(chain: pd.DataFrame, grid: np.ndarray, standardised: bool = False) -> pd.DataFrame:
    """IV on a moneyness grid, one row per expiry; NaN where an expiry has no quotes that far out.

    With standardised=True the grid is in standard deviations, ln(K/F) / (ATM vol * sqrt(T)), so a
    one-week and a six-month smile cover comparable ground instead of the short one's steep wings
    dwarfing everything else.
    """
    atm = expiry_metrics(chain).set_index("expiry")["atm_iv"] if standardised else None
    out = {}
    for expiry, g in chain.groupby("expiry"):
        x = g["log_moneyness"]
        if standardised:
            if not atm.get(expiry, np.nan) > 0:
                continue
            x = x / (atm[expiry] * np.sqrt(g["T"]))
        if len(g) >= 3:
            out[expiry] = [_interp_inside(v, x, g["iv"]) for v in grid]
    return pd.DataFrame.from_dict(out, orient="index", columns=grid)
