from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from optlab import calibration
from optlab.implied_vol import implied_vol
from optlab.market import nse
from optlab.market.chain import build_chain
from optlab.models import black_scholes, heston, sabr
from optlab.models.heston import HestonParams
from optlab.models.sabr import SabrParams

FIXTURES = Path(__file__).parent / "fixtures"
DAY = date(2026, 10, 1)
F, R = 22500.0, 0.055


def synthetic_chain(vol_of: dict) -> pd.DataFrame:
    """A chain-shaped frame: for each expiry (in days), OTM quotes over +/- 2.5 standard deviations,
    with implied vols from vol_of[days](strikes) and prices to match."""
    rows = []
    for days, vols_for in vol_of.items():
        T = days / 365
        k = np.linspace(-2.5, 2.5, 21) * 0.15 * np.sqrt(T)
        K = np.round(F * np.exp(k), 2)
        iv = vols_for(K, T)
        kind = np.where(K < F, "put", "call")
        price = black_scholes.price(F, K, T, R, iv, R, kind)
        rows.append(
            pd.DataFrame(
                {
                    "expiry": DAY + timedelta(days=days),
                    "T": T,
                    "strike": K,
                    "option_type": kind,
                    "close": price,
                    "iv": iv,
                    "forward": F,
                    "r": R,
                    "log_moneyness": np.log(K / F),
                    "delta": black_scholes.delta(F, K, T, R, iv, R, kind) * np.exp(R * T),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def heston_chain(p: HestonParams, expiries=(14, 30, 60, 120, 240)) -> pd.DataFrame:
    def vols(K, T):
        kind = np.where(K < F, "put", "call")
        return implied_vol(heston.price(F, K, T, R, p, kind), F, K, T, R, R, kind)

    return synthetic_chain(dict.fromkeys(expiries, vols))


def test_sabr_recovers_the_parameters_it_was_generated_with():
    truth = SabrParams(alpha=0.13, rho=-0.55, nu=2.5)
    chain = synthetic_chain({30: lambda K, T: sabr.implied_vol(F, K, T, truth)})
    fit = calibration.fit_sabr(chain)
    assert fit.rmse < 1e-6
    assert (fit.params.alpha, fit.params.rho, fit.params.nu) == pytest.approx((0.13, -0.55, 2.5), rel=1e-4)


def test_heston_recovers_a_known_surface():
    truth = HestonParams(v0=0.018, kappa=4.0, theta=0.03, xi=0.9, rho=-0.65)
    fit = calibration.fit_heston(heston_chain(truth))
    assert fit.rmse < 2e-4  # 0.02 vol points
    assert fit.n_expiries == 5
    p = fit.params
    assert p.v0 == pytest.approx(truth.v0, rel=0.05)
    assert p.rho == pytest.approx(truth.rho, abs=0.05)
    assert p.xi == pytest.approx(truth.xi, rel=0.15)


def test_heston_warm_start_from_the_answer_stays_there():
    truth = HestonParams(v0=0.02, kappa=2.5, theta=0.035, xi=0.7, rho=-0.6)
    fit = calibration.fit_heston(heston_chain(truth), previous=truth)
    assert fit.rmse < 1e-5
    np.testing.assert_allclose(fit.params.as_array(), truth.as_array(), rtol=1e-3, atol=1e-4)


def test_heston_needs_two_expiries_in_its_window():
    truth = HestonParams(0.02, 2.0, 0.03, 0.6, -0.6)
    assert calibration.fit_heston(heston_chain(truth, expiries=(30,))) is None
    assert calibration.fit_heston(heston_chain(truth, expiries=(3, 5, 400))) is None  # all outside


@pytest.fixture(scope="module")
def nifty():
    return build_chain(nse.parse((FIXTURES / "fo_20261001_subset.csv.zip").read_bytes()), "NIFTY")


def test_fits_a_real_nifty_day(nifty):
    fits = calibration.fit_sabr_chain(nifty)
    assert len(fits) == 3
    assert all(f.rmse < 0.01 and f.params.rho < 0 for f in fits)  # under a vol point, downside skew

    fit = calibration.fit_heston(nifty)
    assert fit.n_expiries == 2  # the 5-day weekly is outside the Heston window
    assert fit.rmse < 0.015
    assert fit.params.rho < 0


def test_model_vols_line_up_with_the_chain(nifty):
    fits = calibration.fit_sabr_chain(nifty)
    vols = calibration.sabr_vols(nifty, fits)
    assert vols.shape == (len(nifty),) and np.isfinite(vols).all()
    as_rows = pd.DataFrame(
        [
            {
                "expiry": f.expiry,
                "forward": f.forward,
                "alpha": f.params.alpha,
                "rho": f.params.rho,
                "nu": f.params.nu,
                "beta": f.params.beta,
            }  # fmt: skip
            for f in fits[:1]
        ]
    )
    from_rows = calibration.sabr_vols(nifty, as_rows)
    first = (nifty["expiry"] == fits[0].expiry).to_numpy()
    np.testing.assert_allclose(from_rows[first], vols[first])
    assert np.isnan(from_rows[~first]).all()

    model = calibration.heston_vols(nifty, calibration.fit_heston(nifty).params)
    assert model.shape == (len(nifty),)
    assert np.isfinite(model).mean() > 0.95


def test_fits_leave_out_the_lottery_ticket_wings():
    truth = SabrParams(alpha=0.13, rho=-0.55, nu=2.5)
    chain = synthetic_chain({30: lambda K, T: sabr.implied_vol(F, K, T, truth)})
    wing = chain["delta"].abs() < calibration.MIN_ABS_DELTA
    assert wing.any()
    chain.loc[wing, "iv"] += 0.3  # far-wing quotes at absurd vols
    fit = calibration.fit_sabr_chain(chain)[0]
    assert fit.n_quotes == (~wing).sum()
    assert fit.params.nu == pytest.approx(2.5, rel=1e-3)


def test_rough_bergomi_recovers_a_rough_market():
    from optlab.models import rough_bergomi as rb

    truth = rb.RoughBergomiParams(H=0.12, eta=1.8, rho=-0.6)
    market = rb.paths(rb.draw(60_000, 60 / 365, rng=99), truth, rb.ForwardVariance.flat(0.15**2))

    def vols(K, T):
        kind = np.where(K < F, "put", "call")
        return implied_vol(rb.price(F, K, T, R, market, kind), F, K, T, R, R, kind)

    chain = synthetic_chain(dict.fromkeys((3, 7, 14, 30, 60), vols))
    fit = calibration.fit_rough_bergomi(chain, n_paths=16_000)  # different random numbers from the market's
    assert fit.n_expiries == 5
    assert fit.rmse < 0.003
    assert fit.params.H == pytest.approx(0.12, abs=0.06)
    assert fit.params.rho == pytest.approx(-0.6, abs=0.15)
    np.testing.assert_allclose(np.sqrt(fit.xi.values), 0.15, rtol=0.1)
    assert (fit.expiries["skew_rough"] < 0).all() and (fit.expiries["skew_market"] < 0).all()


def test_rough_bergomi_fits_a_real_nifty_day(nifty):
    fit = calibration.fit_rough_bergomi(nifty, n_paths=8_000)
    assert fit.n_expiries == 3  # includes the 5-day weekly, unlike Heston's default window
    assert fit.rmse < 0.01 and fit.shape_rmse <= fit.rmse
    assert fit.params.rho < 0
    assert len(fit.quotes) == fit.n_quotes and fit.quotes["model_iv"].notna().mean() > 0.95


def test_shape_error_ignores_level_offsets():
    misses = pd.Series([0.01, 0.01, 0.01, -0.02, -0.02])
    expiries = pd.Series(["a", "a", "a", "b", "b"])
    assert calibration.shape_rmse(misses, expiries) == pytest.approx(0.0, abs=1e-15)
