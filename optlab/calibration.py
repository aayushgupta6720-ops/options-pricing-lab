"""Fitting Heston and SABR to a day's implied-vol chain (market/chain.py output).

SABR is fitted per expiry, directly to the implied vols (its formula gives vols, not prices), with
beta fixed at 1. Each expiry gets several starting points and keeps the best.

Heston is fitted to the whole surface at once: one set of five parameters for every expiry from
HESTON_MIN_DAYS to HESTON_MAX_DAYS. The residual for each quote is (model price - market price)
/ market vega, which is the vol error to first order but needs no implied-vol inversion inside the
optimiser. Each expiry carries equal total weight (1 / sqrt(its quote count) per quote), so a
weekly with 100 strikes doesn't drown out a quarterly with 10. Expiries under a week are left out:
their very steep smiles are a jump-like effect that a diffusion like Heston can't produce without
distorting everything else. The fit starts from a few parameter sets (plus the previous day's fit
when there is one) and keeps the best. Fit quality is reported as the RMSE of the model's implied
vols against the market's, in vol (0.01 = 1 point).

Both fits use quotes between the 5-delta put and the 5-delta call. Further out, options trade at a
few ticks as lottery tickets (a one-week NIFTY put 11 standard deviations out can close at Rs 0.85,
an implied vol of 52%); no diffusion model produces those, and dividing by their near-zero vega
would let a handful of them dominate a fit.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from optlab.implied_vol import implied_vol
from optlab.models import black_scholes, heston, sabr
from optlab.models.heston import HestonParams
from optlab.models.sabr import SabrParams

HESTON_MIN_DAYS = 7
HESTON_MAX_DAYS = 365
MIN_QUOTES_PER_EXPIRY = 5
MIN_ABS_DELTA = 0.05  # forward delta; drops the far wings (see above)

HESTON_BOUNDS = (
    np.array([1e-4, 0.05, 1e-4, 0.02, -0.99]),  # v0, kappa, theta, xi, rho
    np.array([1.5, 30.0, 1.5, 6.0, 0.99]),
)
HESTON_STARTS = ((2.0, 0.6, -0.6), (6.0, 1.5, -0.5), (1.0, 0.3, -0.8))  # kappa, xi, rho

SABR_BOUNDS = (np.array([1e-3, -0.999, 1e-4]), np.array([5.0, 0.999, 50.0]))  # alpha, rho, nu
SABR_STARTS = ((-0.6, 0.5), (-0.6, 3.0), (-0.1, 1.0), (-0.3, 8.0))  # rho, nu


@dataclass(frozen=True)
class SabrFit:
    expiry: date
    T: float
    forward: float
    params: SabrParams
    rmse: float
    n_quotes: int


@dataclass(frozen=True)
class HestonFit:
    params: HestonParams
    rmse: float
    n_quotes: int
    n_expiries: int
    cost: float


def calibration_quotes(chain: pd.DataFrame, min_days=0, max_days=np.inf) -> pd.DataFrame:
    """The quotes a fit uses: inside the maturity window and the 5-delta wings."""
    days = chain["T"] * 365
    keep = (days >= min_days) & (days <= max_days)
    if "delta" in chain:
        keep &= chain["delta"].abs() >= MIN_ABS_DELTA
    return chain[keep]


def _usable_expiries(chain: pd.DataFrame, min_days=0, max_days=np.inf):
    window = calibration_quotes(chain, min_days, max_days)
    for expiry, quotes in window.groupby("expiry"):
        if len(quotes) >= MIN_QUOTES_PER_EXPIRY:
            yield expiry, quotes.sort_values("strike")


# --- SABR -----------------------------------------------------------------------------------


def fit_sabr(quotes: pd.DataFrame, beta: float = 1.0) -> SabrFit | None:
    """Fit one expiry's smile; None if it has too few quotes."""
    if len(quotes) < MIN_QUOTES_PER_EXPIRY:
        return None
    F, T = float(quotes["forward"].iloc[0]), float(quotes["T"].iloc[0])
    K, market = quotes["strike"].to_numpy(), quotes["iv"].to_numpy()
    atm_guess = float(market[np.abs(quotes["log_moneyness"].to_numpy()).argmin()])

    def residuals(x):
        return sabr.implied_vol(F, K, T, SabrParams(x[0], x[1], x[2], beta)) - market

    best = None
    for rho0, nu0 in SABR_STARTS:
        start = np.clip([atm_guess, rho0, nu0], SABR_BOUNDS[0] * 1.01, SABR_BOUNDS[1] * 0.99)
        result = least_squares(residuals, start, bounds=SABR_BOUNDS, method="trf", xtol=1e-10, ftol=1e-12)
        if np.isfinite(result.cost) and (best is None or result.cost < best.cost):
            best = result
    if best is None:
        return None
    params = SabrParams(*best.x, beta)
    rmse = float(np.sqrt(np.mean(residuals(best.x) ** 2)))
    return SabrFit(quotes["expiry"].iloc[0], T, F, params, rmse, len(quotes))


def fit_sabr_chain(chain: pd.DataFrame, beta: float = 1.0) -> list[SabrFit]:
    fits = (fit_sabr(quotes, beta) for _, quotes in _usable_expiries(chain))
    return [f for f in fits if f is not None]


def sabr_vols(chain: pd.DataFrame, fits: list[SabrFit] | pd.DataFrame) -> np.ndarray:
    """SABR's vol for every quote in the chain (NaN for expiries without a fit).

    `fits` is either SabrFit objects or rows read back from storage (sabr_rows' columns).
    """
    if isinstance(fits, pd.DataFrame):
        by_expiry = {
            row.expiry: (row.forward, SabrParams(row.alpha, row.rho, row.nu, row.beta))
            for row in fits.itertuples()
        }
    else:
        by_expiry = {f.expiry: (f.forward, f.params) for f in fits}
    out = np.full(len(chain), np.nan)
    for i, (expiry, K, T) in enumerate(zip(chain["expiry"], chain["strike"], chain["T"], strict=True)):
        if expiry in by_expiry:
            F, p = by_expiry[expiry]
            out[i] = sabr.implied_vol(F, K, T, p)
    return out


# --- Heston ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Expiry:
    F: float
    T: float
    r: float
    K: np.ndarray
    kind: np.ndarray
    price: np.ndarray
    scale: np.ndarray  # 1 / (vega * sqrt(n)): turns a price error into a weighted vol error


def _heston_inputs(chain: pd.DataFrame) -> list[_Expiry]:
    inputs = []
    for _, q in _usable_expiries(chain, HESTON_MIN_DAYS, HESTON_MAX_DAYS):
        F, T, r = float(q["forward"].iloc[0]), float(q["T"].iloc[0]), float(q["r"].iloc[0])
        K, iv = q["strike"].to_numpy(), q["iv"].to_numpy()
        vega = np.maximum(black_scholes.vega(F, K, T, r, iv, r), 1e-6 * F)
        inputs.append(
            _Expiry(
                F, T, r, K, q["option_type"].to_numpy(), q["close"].to_numpy(), 1 / (vega * np.sqrt(len(q)))
            )
        )
    return inputs


def _heston_residuals(x, expiries: list[_Expiry]) -> np.ndarray:
    p = HestonParams(*x)
    return np.concatenate(
        [(heston.price(e.F, e.K, e.T, e.r, p, e.kind) - e.price) * e.scale for e in expiries]
    )


def _starting_points(expiries: list[_Expiry], chain: pd.DataFrame, previous: HestonParams | None):
    by_T = sorted(expiries, key=lambda e: e.T)
    quotes = chain.set_index("expiry")
    atm = []
    for e in (by_T[0], by_T[-1]):
        near = quotes.loc[quotes["T"] == e.T]
        atm.append(float(near["iv"].iloc[np.abs(near["log_moneyness"].to_numpy()).argmin()]))
    v0, theta = atm[0] ** 2, atm[1] ** 2
    starts = [np.array([v0, kappa, theta, xi, rho]) for kappa, xi, rho in HESTON_STARTS]
    if previous is not None:
        starts.insert(0, previous.as_array())
    lo, hi = HESTON_BOUNDS
    return [np.clip(s, lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo)) for s in starts]


def fit_heston(chain: pd.DataFrame, previous: HestonParams | None = None) -> HestonFit | None:
    """Fit one day's surface; None if fewer than two expiries have enough quotes."""
    expiries = _heston_inputs(chain)
    if len(expiries) < 2:
        return None
    best = None
    for start in _starting_points(expiries, chain, previous):
        try:
            result = least_squares(
                _heston_residuals,
                start,
                args=(expiries,),
                bounds=HESTON_BOUNDS,
                method="trf",
                diff_step=1e-6,  # well above the pricer's ~1e-11 relative noise
                xtol=1e-8,
                ftol=1e-10,
                max_nfev=300,
            )
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(result.cost) and (best is None or result.cost < best.cost):
            best = result
    if best is None:
        return None
    params = HestonParams(*best.x)
    fitted = pd.concat([q for _, q in _usable_expiries(chain, HESTON_MIN_DAYS, HESTON_MAX_DAYS)])
    errors = heston_vols(fitted, params) - fitted["iv"].to_numpy()
    rmse = float(np.sqrt(np.nanmean(errors**2)))
    return HestonFit(params, rmse, len(fitted), len(expiries), float(best.cost))


def heston_vols(chain: pd.DataFrame, params: HestonParams) -> np.ndarray:
    """The Heston model's Black-76 implied vol for every quote in the chain."""
    out = np.full(len(chain), np.nan)
    positions = np.arange(len(chain))
    for _, q in chain.assign(_pos=positions).groupby("expiry"):
        F, T, r = float(q["forward"].iloc[0]), float(q["T"].iloc[0]), float(q["r"].iloc[0])
        kinds = q["option_type"].to_numpy()
        model = heston.price(F, q["strike"].to_numpy(), T, r, params, kinds)
        out[q["_pos"].to_numpy()] = implied_vol(model, F, q["strike"].to_numpy(), T, r, r, kinds)
    return out
