"""Local volatility (Dupire) from a smooth implied-vol surface, and Monte Carlo under it.

The local vol sigma(t, y), with y = ln(K / F_t), that reproduces every European price on a surface
is Dupire's. In total implied variance w(y, T) = sigma_imp(y, T)^2 T (Gatheral, The Volatility
Surface, 2006, eq. 1.10):

    sigma_LV^2 = (dw/dT) / (1 - (y/w) dw/dy + 1/4 (-1/4 - 1/w + y^2/w^2) (dw/dy)^2 + 1/2 d2w/dy2)

Here it's built from the day's Heston fit, whose implied-vol surface is smooth in both strike and
maturity, as finite differences need. Local vol then prices every European option the way that
Heston fit does, so any gap between the two on a path-dependent option comes from the dynamics
alone: Heston's smile moves with its random variance, local vol's is a fixed function of spot and
time. That gap is model risk the vanilla market can't settle.

Where the formula breaks down (a non-positive denominator or calendar slope, or deep wings whose
prices are too small to invert), the local vol is filled from the nearest good value in strike and
kept within [VOL_FLOOR, VOL_CAP]. Paths rarely reach those regions: they're many standard
deviations out.
"""

from dataclasses import dataclass

import numpy as np

from optlab.implied_vol import implied_vol
from optlab.models import heston
from optlab.models.heston import HestonParams

VOL_FLOOR, VOL_CAP = 0.01, 2.0
Y_GRID = np.linspace(-0.6, 0.6, 121)  # log-moneyness ln(K / F_t)
MIN_T = 1 / 365


@dataclass(frozen=True)
class LocalVolSurface:
    times: np.ndarray  # (n_t,), increasing
    log_moneyness: np.ndarray  # (n_y,), increasing
    vols: np.ndarray  # (n_t, n_y)

    def __call__(self, t, y) -> np.ndarray:
        """Bilinear in (t, y), flat outside the grid."""
        t = np.clip(np.asarray(t, float), self.times[0], self.times[-1])
        y = np.clip(np.asarray(y, float), self.log_moneyness[0], self.log_moneyness[-1])
        i = np.clip(np.searchsorted(self.times, t) - 1, 0, len(self.times) - 2)
        j = np.clip(np.searchsorted(self.log_moneyness, y) - 1, 0, len(self.log_moneyness) - 2)
        t0, t1 = self.times[i], self.times[i + 1]
        y0, y1 = self.log_moneyness[j], self.log_moneyness[j + 1]
        a, b = (t - t0) / (t1 - t0), (y - y0) / (y1 - y0)
        v = self.vols
        return (1 - a) * ((1 - b) * v[i, j] + b * v[i, j + 1]) + a * (
            (1 - b) * v[i + 1, j] + b * v[i + 1, j + 1]
        )


def from_total_variance(w: np.ndarray, times: np.ndarray, y: np.ndarray = Y_GRID) -> LocalVolSurface:
    """Dupire's local vol from total implied variance on a grid, w[i, j] = w(y[j], times[i])."""
    w_t = np.gradient(w, times, axis=0)
    w_y = np.gradient(w, y, axis=1)
    w_yy = np.gradient(w_y, y, axis=1)
    yy = y[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1 - yy / w * w_y + 0.25 * (-0.25 - 1 / w + yy**2 / w**2) * w_y**2 + 0.5 * w_yy
        local_var = np.where((denom > 1e-3) & (w_t > 0) & np.isfinite(w), w_t / denom, np.nan)
    vols = _fill_along_strike(np.sqrt(local_var))
    return LocalVolSurface(times, y, np.clip(vols, VOL_FLOOR, VOL_CAP))


def _fill_along_strike(vols: np.ndarray) -> np.ndarray:
    """Each NaN takes the nearest finite value in its row (same maturity); an all-NaN row takes the
    row before it."""
    out = vols.copy()
    idx = np.arange(out.shape[1])
    for i, row in enumerate(out):
        good = np.isfinite(row)
        if good.any():
            nearest = idx[good][np.abs(idx[:, None] - idx[good][None, :]).argmin(axis=1)]
            out[i] = row[nearest]
        elif i:
            out[i] = out[i - 1]
    return out


WING_SDS = 5.0  # beyond this many standard deviations, implied vol is held flat


def heston_total_variance(params: HestonParams, times: np.ndarray, y: np.ndarray = Y_GRID) -> np.ndarray:
    """Total implied variance of a Heston model on a (time, log-moneyness) grid. Prices are taken
    on a forward of 1 (they scale with it), out of the money on each side, and inverted. Beyond
    WING_SDS standard deviations the prices are too small to invert reliably, and their noise
    would wreck the strike derivatives, so implied vol is held flat there."""
    K = np.exp(y)
    kinds = np.where(y >= 0, "call", "put")
    w = np.empty((len(times), len(y)))
    for i, T in enumerate(times):
        prices = heston.price(1.0, K, T, 0.0, params, kinds)
        vols = implied_vol(prices, 1.0, K, T, 0.0, 0.0, kinds)
        atm = vols[np.argmin(np.abs(y))]
        vols[np.abs(y) > WING_SDS * (atm if np.isfinite(atm) else 0.2) * np.sqrt(T)] = np.nan
        w[i] = _fill_along_strike(vols[None, :])[0] ** 2 * T
    return w


def from_heston(params: HestonParams, T_max: float, n_times: int = 40) -> LocalVolSurface:
    """Local vol reproducing a Heston fit's European prices, out to T_max."""
    times = np.linspace(MIN_T, max(T_max, 2 * MIN_T) * 1.05, n_times)
    return from_total_variance(heston_total_variance(params, times), times)


# --- Monte Carlo (numba) ---------------------------------------------------------------------

_KERNEL: dict = {}


def _kernel():
    if _KERNEL:
        return _KERNEL["paths"]
    import numba

    from optlab.exotics import _kernels

    bridge_update = _kernels()["bridge_update"]

    @numba.njit(cache=True)
    def local_vol_at(times, ys, vols, t, y):
        nt, ny = times.size, ys.size
        t = min(max(t, times[0]), times[nt - 1])
        y = min(max(y, ys[0]), ys[ny - 1])
        i = min(max(np.searchsorted(times, t) - 1, 0), nt - 2)
        j = min(max(np.searchsorted(ys, y) - 1, 0), ny - 2)
        a = (t - times[i]) / (times[i + 1] - times[i])
        b = (y - ys[j]) / (ys[j + 1] - ys[j])
        return (1 - a) * ((1 - b) * vols[i, j] + b * vols[i, j + 1]) + a * (
            (1 - b) * vols[i + 1, j] + b * vols[i + 1, j + 1]
        )

    @numba.njit(cache=True)
    def paths(n_paths, n_fixings, substeps, x_start, carry, dt, times, ys, vols, b, down, has_barrier, seed):
        np.random.seed(seed)
        out = np.empty((6, n_paths))
        for p in range(n_paths):
            x, t, total, log_total, lo, hi, survive = x_start, 0.0, 0.0, 0.0, x_start, x_start, 1.0
            for _ in range(n_fixings):
                x0, w = x, 0.0
                for _ in range(substeps):
                    sigma = local_vol_at(times, ys, vols, t, x - x_start - carry * t)
                    x += (carry - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(
                        dt
                    ) * np.random.standard_normal()
                    w += sigma * sigma * dt
                    t += dt
                lo, hi, survive = bridge_update(x0, x, max(w, 1e-300), b, down, has_barrier, lo, hi, survive)
                total += np.exp(x)
                log_total += x
            out[0, p], out[1, p], out[2, p] = np.exp(x), total / n_fixings, np.exp(log_total / n_fixings)
            out[3, p], out[4, p], out[5, p] = np.exp(lo), np.exp(hi), survive
        return out

    _KERNEL["paths"] = paths
    return paths


def stats_local_vol_numba(
    S, T, r, q, surface: LocalVolSurface, n_paths, n_fixings, substeps=4, seed=0, barrier=None
):
    """Per-path statistics (see optlab.exotics.PathStats) under local vol."""
    from optlab.exotics import PathStats, _barrier_args

    b, down, has = _barrier_args(S, barrier)
    out = _kernel()(
        n_paths, n_fixings, substeps, float(np.log(S)), r - q, T / (n_fixings * substeps),
        surface.times, surface.log_moneyness, np.ascontiguousarray(surface.vols), b, down, has, seed,
    )  # fmt: skip
    return PathStats(*out)
