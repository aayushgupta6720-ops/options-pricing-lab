from datetime import date
from pathlib import Path

import pytest

from optlab.market import nse, store
from scripts import calibrate
from scripts.ingest import process_day

FIXTURES = Path(__file__).parent / "fixtures"
DAYS = (date(2026, 9, 30), date(2026, 10, 1))


@pytest.fixture
def dataset(tmp_path):
    raw = nse.parse((FIXTURES / "fo_20261001_subset.csv.zip").read_bytes())
    for day in DAYS:
        raw["trade_date"] = day
        store.save(tmp_path, *process_day(day, raw, 0.1446))
    return tmp_path


def test_fits_every_day_with_quotes(dataset):
    assert calibrate.main(["--data-dir", str(dataset), "--models", "heston"]) == 0
    fits = store.read_table(dataset, store.HESTON_FITS)
    # BANKNIFTY has only a future in the fixture, so no quotes and nothing to fit.
    assert set(zip(fits["trade_date"], fits["underlying"], strict=True)) == {
        (d, u) for d in DAYS for u in ("NIFTY", "RELIANCE")
    }
    assert fits["fitted"].all() and (fits["rmse"] < 0.015).all() and (fits["rho"] < 0.5).all()
    smiles = store.read_table(dataset, store.SABR_FITS)
    assert len(smiles[(smiles["underlying"] == "NIFTY") & (smiles["trade_date"] == DAYS[0])]) == 3


def test_skips_fitted_days_and_refits_with_force(dataset, capsys):
    calibrate.main(["--data-dir", str(dataset)])
    before = store.read_table(dataset, store.SABR_FITS)
    assert calibrate.main(["--data-dir", str(dataset)]) == 0
    out = capsys.readouterr().out
    assert "heston: nothing to fit" in out and "rough: nothing to fit" in out
    assert calibrate.main(["--data-dir", str(dataset), "--force"]) == 0
    after = store.read_table(dataset, store.SABR_FITS)
    assert len(after) == len(before)  # replaced, not duplicated


def test_parallel_workers_give_the_same_days(dataset):
    # Heston only: worker processes don't see the test session's smaller rough-Bergomi path count.
    assert calibrate.main(["--data-dir", str(dataset), "--workers", "2", "--models", "heston"]) == 0
    assert len(store.read_table(dataset, store.HESTON_FITS)) == 4


def test_a_day_that_fails_is_reported_and_the_rest_saved(dataset, monkeypatch, capsys):
    real = calibrate.fit_heston

    def flaky(chain, previous=None, **window):
        if chain["trade_date"].iloc[0] == DAYS[0]:
            raise RuntimeError("boom")
        return real(chain, previous, **window)

    monkeypatch.setattr(calibrate, "fit_heston", flaky)
    assert calibrate.main(["--data-dir", str(dataset)]) == 1
    assert set(store.read_table(dataset, store.HESTON_FITS)["trade_date"]) == {DAYS[1]}
    out = capsys.readouterr().out
    assert "FAILED: NIFTY 2026-09-30: RuntimeError: boom" in out
    assert "FAILED: NIFTY 2026-10-01" not in out  # the other day went through, in both models
    assert set(store.read_table(dataset, store.ROUGH_FITS)["trade_date"]) == {DAYS[1]}


def test_rough_bergomi_tables(dataset):
    assert calibrate.main(["--data-dir", str(dataset), "--models", "rough"]) == 0
    fits = store.read_table(dataset, store.ROUGH_FITS)
    assert set(fits["underlying"]) == {"NIFTY"}  # config.ROUGH_UNDERLYINGS
    assert set(fits["trade_date"]) == set(DAYS) and fits["fitted"].all()
    assert (fits["H"].between(0.01, 0.5) & (fits["rho"] < 0.5)).all()
    assert (fits["heston_rmse"] > 0).all() and (fits["shape_rmse"] <= fits["rmse"] + 1e-12).all()
    assert len(fits["xi_times"].iloc[0]) == len(fits["xi_values"].iloc[0]) == fits["n_expiries"].iloc[0]

    expiries = store.read_table(dataset, store.ROUGH_EXPIRIES)
    assert len(expiries) == fits["n_expiries"].sum()
    assert (expiries[["skew_market", "skew_rough", "skew_heston"]] < 0).all().all()

    quotes = store.read_table(dataset, store.ROUGH_QUOTES)
    assert len(quotes) == fits["n_quotes"].sum()
    assert quotes[["rough_iv", "heston_iv"]].notna().mean().min() > 0.95


def test_unknown_model_is_an_error(dataset):
    with pytest.raises(SystemExit):
        calibrate.main(["--data-dir", str(dataset), "--models", "sabr2"])
