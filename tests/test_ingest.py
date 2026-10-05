from datetime import date
from pathlib import Path

import pytest

from optlab.market import nse, store
from scripts import ingest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_nse(monkeypatch):
    """Serve the fixture for 1 Oct 2026 and 'no file' for every other date, without the network."""
    requested = []

    def load(day, underlyings=None, cache_dir=None, session=None):
        requested.append(day)
        if day != date(2026, 10, 1):
            raise nse.NotPublished(str(day))
        return nse.parse((FIXTURES / "fo_20261001_subset.csv.zip").read_bytes(), underlyings)

    monkeypatch.setattr(nse, "load", load)
    monkeypatch.setattr(nse, "india_vix", lambda day, session=None: 0.1446)
    monkeypatch.setattr(ingest, "PAUSE_SECONDS", 0)
    return requested


def test_ingests_a_day_and_skips_dates_without_files(tmp_path, fake_nse, capsys):
    assert ingest.main(["--data-dir", str(tmp_path), "--since", "2026-09-30"]) == 0
    summary = store.read_summary(tmp_path)
    # The fixture has NIFTY and RELIANCE options but only a BANKNIFTY future (no spot row for it).
    assert set(summary["underlying"]) == {"NIFTY", "RELIANCE", "BANKNIFTY"}
    nifty = summary[summary["underlying"] == "NIFTY"].iloc[0]
    assert nifty["india_vix"] == 0.1446
    assert 0.11 < nifty["atm_iv_30d"] < 0.15
    assert date(2026, 10, 3) not in fake_nse and date(2026, 10, 4) not in fake_nse  # weekend: no request
    assert "no file" in capsys.readouterr().out


def test_skips_days_already_ingested(tmp_path, fake_nse):
    ingest.main(["--data-dir", str(tmp_path), "--date", "2026-10-01"])
    ingest.main(["--data-dir", str(tmp_path), "--date", "2026-10-01"])
    assert fake_nse == [date(2026, 10, 1)]
    ingest.main(["--data-dir", str(tmp_path), "--date", "2026-10-01", "--force"])
    assert len(fake_nse) == 2
    assert len(store.read_summary(tmp_path)) == 3
