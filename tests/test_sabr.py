import numpy as np
import pytest

from optlab.models import sabr
from optlab.models.sabr import SabrParams

F, T = 22500.0, 30 / 365


def test_no_vol_of_vol_with_lognormal_backbone_is_flat():
    K = F * np.exp(np.linspace(-0.3, 0.3, 13))
    np.testing.assert_allclose(sabr.implied_vol(F, K, T, SabrParams(0.15, -0.5, 0.0)), 0.15)


def test_at_the_money_matches_the_expansion():
    p = SabrParams(alpha=0.14, rho=-0.6, nu=2.0)
    expected = 0.14 * (1 + T * (-0.6 * 2.0 * 0.14 / 4 + (2 - 3 * 0.36) / 24 * 4.0))
    assert sabr.implied_vol(F, F, T, p) == pytest.approx(expected, rel=1e-12)
    # and the near-the-money branch joins up with the exact one
    assert sabr.implied_vol(F, F * (1 + 1e-9), T, p) == pytest.approx(expected, rel=1e-8)
    assert sabr.implied_vol(F, F * 1.001, T, p) == pytest.approx(expected, rel=1e-2)


def test_zero_correlation_gives_a_symmetric_smile():
    k = np.linspace(0.01, 0.3, 10)
    p = SabrParams(0.15, 0.0, 1.5)
    np.testing.assert_allclose(
        sabr.implied_vol(F, F * np.exp(k), T, p), sabr.implied_vol(F, F * np.exp(-k), T, p), rtol=1e-12
    )


def test_negative_correlation_skews_towards_low_strikes():
    p = SabrParams(0.15, -0.7, 1.5)
    low, atm, high = sabr.implied_vol(F, F * np.exp([-0.1, 0.0, 0.1]), T, p)
    assert low > atm and low > high  # nu also lifts the call wing, so high vs atm can go either way


def test_other_backbones_are_finite_and_close_to_alpha_scaled():
    # beta = 0.5: alpha is in units of F^(1 - beta), so ATM vol ~ alpha / sqrt(F).
    p = SabrParams(alpha=0.15 * np.sqrt(F), rho=-0.3, nu=1.0, beta=0.5)
    vols = sabr.implied_vol(F, F * np.exp(np.linspace(-0.3, 0.3, 13)), T, p)
    assert np.isfinite(vols).all()
    assert vols[6] == pytest.approx(0.15, rel=0.02)


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        SabrParams(0.15, 1.0, 1.0)
    with pytest.raises(ValueError):
        SabrParams(-0.1, 0.0, 1.0)
