"""Summary numbers and grids from an implied-vol chain (market/chain.py output).

Per expiry: at-the-money vol (interpolated at log-moneyness 0) and the 25-delta put and call vols.
ATM is only reported when the quotes either side of the forward are within MAX_ATM_GAP_SD standard
deviations of each other; a thin expiry whose nearest put and call sit far apart would otherwise
interpolate straight across the skew.
Across expiries: values at fixed tenors, interpolating ATM *total variance* (sigma^2 * T) linearly
in time, which is the standard way to keep the term structure free of calendar arbitrage. Outside
the listed expiries the vol is held flat, but only up to EXTRAPOLATION_DAYS away; further out it's
NaN (a monthly-only underlying has no honest 7-day vol).
"""

import numpy as np
import pandas as pd

from optlab import config, variance

EXTRAPOLATION_DAYS = 14
MAX_ATM_GAP_SD = 1.0


def _interp_inside(x_new, x, y):
    """Linear interpolation that returns NaN outside the data instead of clamping."""
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    if len(x) == 0 or not x[0] <= x_new <= x[-1]:
        return np.nan
    return float(np.interp(x_new, x, y))


def atm_vol(g: pd.DataFrame) -> float:
    """IV at log-moneyness 0 for one expiry, or NaN if the bracketing quotes are too far apart."""
    g = g.sort_values("log_moneyness")
    k, iv = g["log_moneyness"].to_numpy(), g["iv"].to_numpy()
    if len(k) == 0 or not k[0] <= 0 <= k[-1]:
        return np.nan
    hi = int(np.searchsorted(k, 0.0))
    if k[hi] == 0:
        return float(iv[hi])
    lo = hi - 1
    gap = (k[hi] - k[lo]) / (0.5 * (iv[lo] + iv[hi]) * np.sqrt(g["T"].iloc[0]))
    return float(np.interp(0.0, k[lo : hi + 1], iv[lo : hi + 1])) if gap <= MAX_ATM_GAP_SD else np.nan


def vol_at_delta(side: pd.DataFrame, target: float) -> float:
    """IV where one side's forward delta equals `target` (e.g. -0.25 for puts, 0.25 for calls).

    Delta falls as the strike rises for both puts and calls. Noisy quotes can break that order,
    so it's enforced (running minimum along the strikes) before finding the strike at the target
    delta; the vol is then read off the smile at that strike. That makes the answer independent of
    how out-of-order points happen to sort.
    """
    side = side.sort_values("log_moneyness")
    k, iv = side["log_moneyness"].to_numpy(), side["iv"].to_numpy()
    delta = np.minimum.accumulate(side["delta"].to_numpy())
    if len(k) < 2 or not delta[-1] <= target <= delta[0]:
        return np.nan
    k_target = np.interp(target, delta[::-1], k[::-1])
    return float(np.interp(k_target, k, iv))


METRIC_COLUMNS = ["expiry", "T", "forward", "atm_iv", "put_25d_iv", "call_25d_iv", "skew_25d", "n_quotes"]


def expiry_metrics(chain: pd.DataFrame) -> pd.DataFrame:
    """One row per expiry with the METRIC_COLUMNS."""
    if chain.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    rows = []
    for expiry, g in chain.groupby("expiry"):
        calls, puts = g[g["option_type"] == "call"], g[g["option_type"] == "put"]
        put_25 = vol_at_delta(puts, -0.25)
        call_25 = vol_at_delta(calls, 0.25)
        rows.append(
            {
                "expiry": expiry,
                "T": g["T"].iloc[0],
                "forward": g["forward"].iloc[0],
                "atm_iv": atm_vol(g),
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
    # The 30-day variance-swap rate from the whole strip (as India VIX is), interpolated like the
    # ATM vols so the two compare day for day. It's the fair rate to set against realized variance.
    strip = variance.term_structure(chain) if len(chain) else pd.DataFrame(columns=["T", "vs_vol"])
    row["vs_vol_30d"] = _at_tenor(strip, "vs_vol", 30, total_variance=True)
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
