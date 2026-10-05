"""Black-Scholes-Merton prices and Greeks for European options with a continuous dividend yield.

Every function broadcasts over numpy arrays, so a whole option chain prices in one call. `kind` is
"call", "put", or an array of them. Scalars in give numpy scalars out.

Conventions: vega and rho are per 1.00 change (divide by 100 for "per 1%"); theta is per year
(divide by 365 for "per calendar day").
"""

import numpy as np
from scipy.special import ndtr

from optlab.contracts import OptionSpec


def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _is_call(kind):
    kind = np.asarray(kind)
    if not np.isin(kind, ("call", "put")).all():
        raise ValueError(f"kind must be 'call' or 'put', got {np.unique(kind)}")
    return kind == "call"


def _terms(S, K, T, r, sigma, q):
    """d1, d2 and the pieces the formulas share.

    When sigma * sqrt(T) is zero the option is a forward contract that is either in or out of the
    money for sure, so d1 = d2 = +/-inf. Plugging that into the formulas gives the right limits
    for price, delta and rho; gamma, vega and the volatility part of theta are zeroed by `live`.
    """
    S, K, T, r, sigma, q = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q)))
    vol_sqrt_t = sigma * np.sqrt(T)
    live = vol_sqrt_t > 0
    log_moneyness = np.log(S / K) + (r - q) * T  # ln(F / K)
    safe = np.where(live, vol_sqrt_t, 1.0)
    d1 = np.where(
        live, (log_moneyness + 0.5 * vol_sqrt_t**2) / safe, np.where(log_moneyness > 0, np.inf, -np.inf)
    )
    d2 = d1 - vol_sqrt_t
    return S, K, T, r, sigma, q, d1, d2, live, safe


def price(S, K, T, r, sigma, q=0.0, kind="call"):
    S, K, T, r, sigma, q, d1, d2, *_ = _terms(S, K, T, r, sigma, q)
    is_call = _is_call(kind)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    call = S * df_q * ndtr(d1) - K * df_r * ndtr(d2)
    put = K * df_r * ndtr(-d2) - S * df_q * ndtr(-d1)
    return np.where(is_call, call, put)[()]


def delta(S, K, T, r, sigma, q=0.0, kind="call"):
    S, K, T, r, sigma, q, d1, *_ = _terms(S, K, T, r, sigma, q)
    df_q = np.exp(-q * T)
    return np.where(_is_call(kind), df_q * ndtr(d1), -df_q * ndtr(-d1))[()]


def gamma(S, K, T, r, sigma, q=0.0, kind="call"):
    S, K, T, r, sigma, q, d1, _, live, safe = _terms(S, K, T, r, sigma, q)
    _is_call(kind)
    return np.where(live, np.exp(-q * T) * _norm_pdf(d1) / (S * safe), 0.0)[()]


def vega(S, K, T, r, sigma, q=0.0, kind="call"):
    S, K, T, r, sigma, q, d1, _, live, _ = _terms(S, K, T, r, sigma, q)
    _is_call(kind)
    return np.where(live, S * np.exp(-q * T) * _norm_pdf(d1) * np.sqrt(T), 0.0)[()]


def theta(S, K, T, r, sigma, q=0.0, kind="call"):
    """Time decay per year: the change in value as calendar time passes (so usually negative)."""
    S, K, T, r, sigma, q, d1, d2, live, _ = _terms(S, K, T, r, sigma, q)
    is_call = _is_call(kind)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = np.where(live, -S * df_q * _norm_pdf(d1) * sigma / (2.0 * np.sqrt(T)), 0.0)
    call = decay - r * K * df_r * ndtr(d2) + q * S * df_q * ndtr(d1)
    put = decay + r * K * df_r * ndtr(-d2) - q * S * df_q * ndtr(-d1)
    return np.where(is_call, call, put)[()]


def rho(S, K, T, r, sigma, q=0.0, kind="call"):
    S, K, T, r, sigma, q, _, d2, *_ = _terms(S, K, T, r, sigma, q)
    is_call = _is_call(kind)
    df_r = np.exp(-r * T)
    return np.where(is_call, K * T * df_r * ndtr(d2), -K * T * df_r * ndtr(-d2))[()]


GREEKS = {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def greeks(S, K, T, r, sigma, q=0.0, kind="call") -> dict:
    return {name: fn(S, K, T, r, sigma, q, kind) for name, fn in GREEKS.items()}


def _args(spec: OptionSpec):
    return spec.S, spec.K, spec.T, spec.r, spec.sigma, spec.q, spec.kind


def price_spec(spec: OptionSpec) -> float:
    if spec.style != "european":
        raise ValueError("Black-Scholes prices European options only")
    return float(price(*_args(spec)))


def greeks_spec(spec: OptionSpec) -> dict:
    return {name: float(value) for name, value in greeks(*_args(spec)).items()}
