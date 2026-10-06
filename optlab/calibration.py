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
from optlab.models import black_scholes, heston, rough_bergomi, sabr
from optlab.models.heston import HestonParams
from optlab.models.sabr import SabrParams
from optlab.surface import atm_vol
from optlab.variance import monotone

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


def _price_inputs(chain: pd.DataFrame, min_days: float, max_days: float) -> list[_Expiry]:
    inputs = []
    for _, q in _usable_expiries(chain, min_days, max_days):
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


def fit_heston(
    chain: pd.DataFrame,
    previous: HestonParams | None = None,
    min_days: float = HESTON_MIN_DAYS,
    max_days: float = HESTON_MAX_DAYS,
) -> HestonFit | None:
    """Fit one day's surface; None if fewer than two expiries have enough quotes."""
    expiries = _price_inputs(chain, min_days, max_days)
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
    fitted = pd.concat([q for _, q in _usable_expiries(chain, min_days, max_days)])
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


# --- Rough Bergomi --------------------------------------------------------------------------

ROUGH_MIN_DAYS = 2
ROUGH_MAX_DAYS = 91
ROUGH_PATHS = 20_000
ROUGH_SEED = 2026  # the same random numbers for every fit, so fits are comparable day to day
ROUGH_BOUNDS = (np.array([0.01, 0.1, -0.99]), np.array([0.5, 5.0, 0.5]))  # H, eta, rho
ROUGH_STARTS = ((0.1, 1.5, -0.7), (0.3, 1.0, -0.5))
LOG_XI_BOUNDS = (np.log(1e-4), np.log(1.0))


@dataclass(frozen=True)
class RoughFit:
    params: rough_bergomi.RoughBergomiParams
    xi: rough_bergomi.ForwardVariance
    rmse: float
    shape_rmse: float  # RMSE after removing each expiry's average miss: the smile's shape alone
    n_quotes: int
    n_expiries: int
    cost: float
    quotes: pd.DataFrame  # the fitted quotes, with the model's implied vol in "model_iv"
    expiries: pd.DataFrame  # per expiry: T, forward, r, atm_vol, skew_market (SABR), skew_rough


def shape_rmse(misses: pd.Series, expiries: pd.Series) -> float:
    """RMSE of model-minus-market vols after subtracting each expiry's mean miss."""
    centred = misses - misses.groupby(expiries.to_numpy()).transform("mean").to_numpy()
    return float(np.sqrt(np.nanmean(centred**2)))


def _atm_forward_variance(chain: pd.DataFrame, times: np.ndarray, min_days, max_days) -> np.ndarray:
    """Starting forward variances between the fitted expiries, from their ATM vols."""
    total = []
    for _, q in _usable_expiries(chain, min_days, max_days):
        vol = atm_vol(q)
        total.append((vol if np.isfinite(vol) else float(q["iv"].median())) ** 2 * float(q["T"].iloc[0]))
    total = monotone(np.array(total), np.ones(len(total)))
    steps = np.diff(np.concatenate([[0.0], total])) / np.diff(np.concatenate([[0.0], times]))
    return np.clip(steps, np.exp(LOG_XI_BOUNDS[0]) * 1.01, np.exp(LOG_XI_BOUNDS[1]) * 0.99)


def fit_rough_bergomi(
    chain: pd.DataFrame,
    previous: rough_bergomi.RoughBergomiParams | None = None,
    n_paths: int = ROUGH_PATHS,
    min_days: float = ROUGH_MIN_DAYS,
    max_days: float = ROUGH_MAX_DAYS,
) -> RoughFit | None:
    """Fit rough Bergomi to one day's expiries from min_days to max_days.

    Free parameters: H, eta and rho, plus the forward variance curve xi0(t), one level per interval
    between expiries (started from the ATM term structure). Variance swaps replicated from the quoted
    strikes would pin xi0 down without fitting, but they were too noisy expiry by expiry: the
    strip for a one-week expiry picks up the lottery-ticket wings, and a thinly quoted expiry's strip
    stops short. Prices come from Monte Carlo with the same random numbers at every step of the
    optimiser and on every day; residuals are the same vega-weighted price errors as Heston's.
    """
    expiries = sorted(_price_inputs(chain, min_days, max_days), key=lambda e: e.T)
    if len(expiries) < 2:
        return None
    times = np.array([e.T for e in expiries])
    log_xi0 = np.log(_atm_forward_variance(chain, times, min_days, max_days))
    normals = rough_bergomi.draw(n_paths, times[-1], rng=ROUGH_SEED)
    cache: dict = {}

    def unpack(x):
        return rough_bergomi.RoughBergomiParams(*x[:3]), rough_bergomi.ForwardVariance(times, np.exp(x[3:]))

    def residuals(x):
        sim = rough_bergomi.paths(normals, *unpack(x), cache=cache)
        return np.concatenate(
            [(rough_bergomi.price(e.F, e.K, e.T, e.r, sim, e.kind) - e.price) * e.scale for e in expiries]
        )

    lower = np.concatenate([ROUGH_BOUNDS[0], np.full(len(times), LOG_XI_BOUNDS[0])])
    upper = np.concatenate([ROUGH_BOUNDS[1], np.full(len(times), LOG_XI_BOUNDS[1])])
    starts = [np.concatenate([s, log_xi0]) for s in ROUGH_STARTS]
    if previous is not None:
        starts = [np.concatenate([[previous.H, previous.eta, previous.rho], log_xi0]), starts[0]]
    best = None
    for start in starts:
        start = np.clip(start, lower + 1e-3 * (upper - lower), upper - 1e-3 * (upper - lower))
        result = least_squares(
            residuals,
            start,
            bounds=(lower, upper),
            method="trf",
            diff_step=1e-3,
            xtol=1e-6,
            ftol=1e-8,
            max_nfev=50,
        )
        if np.isfinite(result.cost) and (best is None or result.cost < best.cost):
            best = result
    if best is None:
        return None
    params, xi = unpack(best.x)
    fitted = pd.concat([q for _, q in _usable_expiries(chain, min_days, max_days)])
    sim = rough_bergomi.paths(normals, params, xi)
    fitted = fitted.assign(model_iv=rough_vols(fitted, sim))
    misses = fitted["model_iv"] - fitted["iv"]
    rmse = float(np.sqrt(np.nanmean(misses**2)))
    return RoughFit(
        params,
        xi,
        rmse,
        shape_rmse(misses, fitted["expiry"]),
        len(fitted),
        len(expiries),
        float(best.cost),
        fitted,
        _expiry_skews(fitted, sim),
    )


def _expiry_skews(fitted: pd.DataFrame, sim) -> pd.DataFrame:
    """ATM level and skew per expiry: the market's from a SABR fit to its quotes, rough Bergomi's
    from the simulated paths (the same paths for both strikes, so the difference isn't noise)."""
    rows = []
    for expiry, q in fitted.groupby("expiry"):
        F, T, r = float(q["forward"].iloc[0]), float(q["T"].iloc[0]), float(q["r"].iloc[0])
        level = atm_vol(q)
        level = level if np.isfinite(level) else float(q["iv"].median())
        smile = fit_sabr(q)

        def rough_at(k, F=F, T=T, r=r):
            K, kinds = F * np.exp(k), np.where(k < 0, "put", "call")
            return implied_vol(rough_bergomi.price(F, K, T, r, sim, kinds), F, K, T, r, r, kinds)

        def sabr_at(k, F=F, T=T, smile=smile):
            return sabr.implied_vol(F, F * np.exp(k), T, smile.params)

        rows.append(
            {
                "expiry": expiry,
                "T": T,
                "forward": F,
                "r": r,
                "atm_vol": level,
                "skew_market": atm_skew(sabr_at, T, level) if smile else np.nan,
                "skew_rough": atm_skew(rough_at, T, level),
            }
        )
    return pd.DataFrame(rows)


def heston_skew(params: HestonParams, F: float, T: float, r: float, level: float) -> float:
    """Heston's ATM skew by the same central difference as the others."""

    def vols(k):
        K, kinds = F * np.exp(k), np.where(k < 0, "put", "call")
        return implied_vol(heston.price(F, K, T, r, params, kinds), F, K, T, r, r, kinds)

    return atm_skew(vols, T, level)


def rough_vols(chain: pd.DataFrame, sim) -> np.ndarray:
    """Black-76 implied vols of simulated rough Bergomi prices for every quote in the chain."""
    out = np.full(len(chain), np.nan)
    for _, q in chain.assign(_pos=np.arange(len(chain))).groupby("expiry"):
        F, T, r = float(q["forward"].iloc[0]), float(q["T"].iloc[0]), float(q["r"].iloc[0])
        K, kinds = q["strike"].to_numpy(), q["option_type"].to_numpy()
        model = rough_bergomi.price(F, K, T, r, sim, kinds)
        out[q["_pos"].to_numpy()] = implied_vol(model, F, K, T, r, r, kinds)
    return out


def atm_skew(vol_at, T: float, atm_vol: float) -> float:
    """dsigma / d ln K at the money by a central difference over +/- a quarter standard deviation.

    vol_at(log_moneyness_array) -> vols. Negative when puts are dearer than calls.
    """
    h = 0.25 * atm_vol * np.sqrt(T)
    lo, hi = vol_at(np.array([-h, h]))
    return float((hi - lo) / (2 * h))
