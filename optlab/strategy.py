"""Multi-leg option positions: payoff, value before expiry, breakevens and Greeks.

A leg is long (+) or short (-) some lots of a European call or put. Values are per position, in
rupees: price per unit x lot size x lots. Before expiry each leg is valued with Black-76 on the
forward at its own implied vol, and a spot move shifts the forward proportionally (constant
carry) while each strike keeps its vol ("sticky strike").
"""

from dataclasses import dataclass

import numpy as np

from optlab.models import black_scholes


@dataclass(frozen=True)
class Leg:
    side: int  # +1 long, -1 short
    kind: str  # "call" | "put"
    strike: float
    lots: int = 1


# name -> legs as (side, kind, offset in wing widths from the at-the-money strike)
PRESETS: dict[str, list[tuple[int, str, int]]] = {
    "Long call": [(1, "call", 0)],
    "Long put": [(1, "put", 0)],
    "Long straddle": [(1, "call", 0), (1, "put", 0)],
    "Short straddle": [(-1, "call", 0), (-1, "put", 0)],
    "Long strangle": [(1, "call", 1), (1, "put", -1)],
    "Bull call spread": [(1, "call", 0), (-1, "call", 1)],
    "Bear put spread": [(1, "put", 0), (-1, "put", -1)],
    "Iron condor": [(1, "put", -2), (-1, "put", -1), (-1, "call", 1), (1, "call", 2)],
    "Long call butterfly": [(1, "call", -1), (-1, "call", 0), (-1, "call", 0), (1, "call", 1)],
}


def preset_legs(name: str, atm_strike: float, width: float) -> list[Leg]:
    return [Leg(side, kind, atm_strike + offset * width) for side, kind, offset in PRESETS[name]]


def payoff(legs: list[Leg], spots) -> np.ndarray:
    """Exercise value of the position per unit (before multiplying by lot size)."""
    spots = np.asarray(spots, dtype=float)
    total = np.zeros_like(spots)
    for leg in legs:
        intrinsic = (
            np.maximum(spots - leg.strike, 0) if leg.kind == "call" else np.maximum(leg.strike - spots, 0)
        )
        total += leg.side * leg.lots * intrinsic
    return total


def value(legs: list[Leg], spots, forward_ratio: float, T: float, r: float, vols) -> np.ndarray:
    """Model value per unit with T years left; forward = spot * forward_ratio."""
    spots = np.asarray(spots, dtype=float)
    if T <= 0:
        return payoff(legs, spots)
    total = np.zeros_like(spots)
    for leg, vol in zip(legs, vols, strict=True):
        price = black_scholes.price(spots * forward_ratio, leg.strike, T, r, vol, r, leg.kind)
        total += leg.side * leg.lots * price
    return total


def leg_prices(legs: list[Leg], spot: float, forward_ratio: float, T: float, r: float, vols) -> list[float]:
    """Price of one unit of each leg's option (unsigned: the cost of buying it)."""
    return [
        float(black_scholes.price(spot * forward_ratio, leg.strike, T, r, vol, r, leg.kind))
        for leg, vol in zip(legs, vols, strict=True)
    ]


def premium(legs: list[Leg], prices) -> float:
    """Net premium per unit: positive means the position costs money (a debit)."""
    return float(sum(leg.side * leg.lots * p for leg, p in zip(legs, prices, strict=True)))


def breakevens(spots, pnl) -> list[float]:
    """Spots where the P&L crosses zero, linearly interpolated between grid points."""
    spots, pnl = np.asarray(spots, dtype=float), np.asarray(pnl, dtype=float)
    sign = np.sign(pnl)
    out = [
        float(spots[i] - pnl[i] * (spots[i + 1] - spots[i]) / (pnl[i + 1] - pnl[i]))
        for i in np.nonzero(sign[:-1] * sign[1:] < 0)[0]
    ]
    # A crossing that lands exactly on a grid point.
    out += [float(spots[i]) for i in np.nonzero(sign[1:-1] == 0)[0] + 1 if sign[i - 1] * sign[i + 1] < 0]
    return sorted(out)


def extremes(legs: list[Leg], net_premium: float) -> tuple[float, float]:
    """Max profit and max loss at expiry per unit, +inf / -inf when unlimited.

    The payoff is piecewise linear with kinks at the strikes, so its extremes are at a strike, at
    spot 0, or off to infinity; only a net long (short) call position makes the upside unlimited.
    """
    points = np.array([0.0, *sorted({leg.strike for leg in legs})])
    pnl = payoff(legs, points) - net_premium
    calls = sum(leg.side * leg.lots for leg in legs if leg.kind == "call")
    best = np.inf if calls > 0 else pnl.max()
    worst = -np.inf if calls < 0 else pnl.min()
    return float(best), float(worst)


def greeks(legs: list[Leg], spot: float, forward_ratio: float, T: float, r: float, vols) -> dict:
    """Position Greeks per unit; same conventions as black_scholes (vega per 1.00, theta per year)."""
    total = dict.fromkeys(("delta", "gamma", "vega", "theta"), 0.0)
    for leg, vol in zip(legs, vols, strict=True):
        g = black_scholes.greeks(spot * forward_ratio, leg.strike, T, r, vol, r, leg.kind)
        weight = leg.side * leg.lots
        # Greeks above are with respect to the forward; dF/dS = forward_ratio.
        total["delta"] += weight * float(g["delta"]) * forward_ratio
        total["gamma"] += weight * float(g["gamma"]) * forward_ratio**2
        total["vega"] += weight * float(g["vega"])
        total["theta"] += weight * float(g["theta"])
    return total
