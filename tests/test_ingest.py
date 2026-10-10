from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest
import requests

from optlab import config
from optlab.market import nse, store
from scripts import ingest

FIXTURES = Path(__file__).parent / "fixtures"
RAW = (FIXTURES / "fo_20261001_subset.csv.zip").read_bytes()


@pytest.fixture
def fake_nse(monkeypatch):
    """Serve the fixture (re-dated) for the dates in `published`, 'no file' for every other date.

    Returns the list of dates requested; set `fake_nse.broken` / `fake_nse.no_vix` to make some
    dates fail.
    """

    class Fake(list):
        published = {date(2026, 10, 1)}
        broken: set = set()
        no_vix: set = set()

    requested = Fake()

    def load(day, underlyings=None, cache_dir=None, session=None):
        requested.append(day)
        if day in requested.broken:
            raise requests.HTTPError("403 Client Error: Forbidden")
        if day not in requested.published:
            raise nse.NotPublished(str(day))
        raw = nse.parse(RAW, underlyings)
        raw["trade_date"] = day
        return raw

    def india_vix(day, session=None, cache_dir=None):
        if day in requested.no_vix:
            raise requests.HTTPError("500 Server Error")
        return 0.1446

    monkeypatch.setattr(nse, "load", load)
    monkeypatch.setattr(nse, "india_vix", india_vix)
    monkeypatch.setattr(ingest, "PAUSE_SECONDS", 0)
    return requested


def test_ingests_a_day_and_skips_dates_without_files(tmp_path, fake_nse, capsys):
    assert ingest.main(["--data-dir", str(tmp_path), "--since", "2026-09-30"]) == 0
    summary = store.read_summary(tmp_path)
    assert set(summary["underlying"]) == set(config.UNDERLYINGS)
    nifty = summary[summary["underlying"] == "NIFTY"].iloc[0]
    assert nifty["india_vix"] == 0.1446
    assert 0.11 < nifty["atm_iv_30d"] < 0.15
    assert "no file" in capsys.readouterr().out


def test_weekend_sessions_are_ingested(tmp_path, fake_nse):
    fake_nse.published = {date(2025, 2, 1)}  # Budget day, a Saturday
    assert ingest.main(["--data-dir", str(tmp_path), "--since", "2025-01-31", "--days", "0"]) == 0
    assert date(2025, 2, 1) in set(store.read_summary(tmp_path)["trade_date"])
    assert ingest.main(["--data-dir", str(tmp_path), "--date", "2025-02-01", "--force"]) == 0


def test_one_bad_day_does_not_cost_the_others(tmp_path, fake_nse, capsys):
    fake_nse.published = {date(2026, 10, 1), date(2026, 10, 5)}
    fake_nse.broken = {date(2026, 10, 2)}
    assert ingest.main(["--data-dir", str(tmp_path), "--since", "2026-10-01"]) == 1
    days = set(store.read_summary(tmp_path)["trade_date"])
    assert days == {date(2026, 10, 1), date(2026, 10, 5)}
    assert "2026-10-02: FAILED" in capsys.readouterr().out


def test_skips_days_already_ingested(tmp_path, fake_nse):
    ingest.main(["--data-dir", str(tmp_path), "--date", "2026-10-01"])
    ingest.main(["--data-dir", str(tmp_path), "--date", "2026-10-01"])
    assert fake_nse == [date(2026, 10, 1)]
    ingest.main(["--data-dir", str(tmp_path), "--date", "2026-10-01", "--force"])
    assert len(fake_nse) == 2
    assert len(store.read_summary(tmp_path)) == len(config.UNDERLYINGS)


def test_a_newly_added_underlying_gets_backfilled(tmp_path, fake_nse, monkeypatch):
    monkeypatch.setattr(config, "UNDERLYINGS", ("NIFTY",))
    ingest.main(["--data-dir", str(tmp_path), "--date", "2026-10-01"])
    monkeypatch.setattr(config, "UNDERLYINGS", ("NIFTY", "RELIANCE"))
    ingest.main(["--data-dir", str(tmp_path), "--date", "2026-10-01"])
    assert len(fake_nse) == 2  # the day was redone for the new underlying
    assert set(store.read_summary(tmp_path)["underlying"]) == {"NIFTY", "RELIANCE"}


def test_a_day_whose_vix_failed_is_retried(tmp_path, fake_nse):
    fake_nse.no_vix = {date(2026, 10, 1)}
    assert ingest.main(["--data-dir", str(tmp_path), "--date", "2026-10-01"]) == 0
    assert store.read_summary(tmp_path)["india_vix"].isna().all()
    fake_nse.no_vix = set()
    ingest.main(["--data-dir", str(tmp_path), "--date", "2026-10-01"])
    assert (store.read_summary(tmp_path)["india_vix"] == 0.1446).all()


def test_month_boundaries_flush_and_both_months_are_saved(tmp_path, fake_nse):
    fake_nse.published = {date(2026, 9, 30), date(2026, 10, 1)}
    ingest.main(["--data-dir", str(tmp_path), "--since", "2026-09-30"])
    assert store.chain_path(tmp_path, "NIFTY", "2026-09").exists()
    assert store.chain_path(tmp_path, "NIFTY", "2026-10").exists()
    assert not store.read_chain(tmp_path, "NIFTY", date(2026, 9, 30)).empty


def test_an_unlisted_underlying_gets_a_row_without_a_spot(tmp_path, fake_nse, monkeypatch):
    monkeypatch.setattr(config, "UNDERLYINGS", ("NIFTY", "FINNIFTY"))
    ingest.main(["--data-dir", str(tmp_path), "--date", "2026-10-01"])
    summary = store.read_summary(tmp_path).set_index("underlying")
    assert np.isnan(summary.loc["FINNIFTY", "spot"])
    assert summary.loc["FINNIFTY", "n_quotes"] == 0


def test_rebuild_summary_recomputes_metrics_and_keeps_chains(tmp_path, fake_nse):
    from scripts import rebuild_summary

    ingest.main(["--data-dir", str(tmp_path), "--date", "2026-10-01"])
    summary = store.read_summary(tmp_path)
    summary["atm_iv_30d"] = 0.99  # pretend an older formula produced this
    summary = summary.drop(columns="vs_vol_30d")  # and that it predates this column
    store.write_summary(tmp_path, summary)
    quotes = len(store.read_chain(tmp_path, "NIFTY", date(2026, 10, 1)))

    assert rebuild_summary.main(["--data-dir", str(tmp_path)]) == 0
    rebuilt = store.read_summary(tmp_path).set_index("underlying")
    assert 0.11 < rebuilt.loc["NIFTY", "atm_iv_30d"] < 0.15
    assert 0.13 < rebuilt.loc["NIFTY", "vs_vol_30d"] < 0.16
    assert rebuilt.loc["NIFTY", "india_vix"] == 0.1446
    assert len(store.read_chain(tmp_path, "NIFTY", date(2026, 10, 1))) == quotes


def at(monkeypatch, when: datetime):
    """Run ingest as if it were `when` (IST)."""
    monkeypatch.setattr(ingest, "now", lambda: when.replace(tzinfo=ingest.IST))


def test_closed_days_are_remembered_and_not_asked_again(tmp_path, fake_nse, monkeypatch):
    fake_nse.published = {date(2026, 10, 1), date(2026, 10, 5)}  # 2 Oct a holiday, then a weekend
    at(monkeypatch, datetime(2026, 10, 5, 20, 17))
    assert ingest.main(["--data-dir", str(tmp_path), "--since", "2026-10-01"]) == 0
    assert store.closed_days(tmp_path) == {date(2026, 10, 2), date(2026, 10, 3), date(2026, 10, 4)}

    # NSE stops answering (as it did late at night on 9 Oct 2026); nothing is left to ask it.
    fake_nse.broken = {date(2026, 10, d) for d in range(1, 6)}
    fake_nse.clear()
    assert ingest.main(["--data-dir", str(tmp_path), "--since", "2026-10-01"]) == 0
    assert fake_nse == []
    # An explicit date still asks.
    fake_nse.broken = set()
    ingest.main(["--data-dir", str(tmp_path), "--date", "2026-10-03"])
    assert fake_nse == [date(2026, 10, 3)]


def test_a_date_with_no_later_file_is_not_closed(tmp_path, fake_nse, monkeypatch):
    at(monkeypatch, datetime(2026, 10, 2, 21, 0))  # 2 Oct not published yet: maybe late, maybe a holiday
    ingest.main(["--data-dir", str(tmp_path), "--since", "2026-10-01"])
    assert store.closed_days(tmp_path) == set()
    fake_nse.clear()
    ingest.main(["--data-dir", str(tmp_path), "--since", "2026-10-01"])
    assert fake_nse == [date(2026, 10, 2)]  # asked again


def test_today_is_only_tried_from_the_evening(tmp_path, fake_nse, monkeypatch):
    fake_nse.published = {date(2026, 10, 1)}
    at(monkeypatch, datetime(2026, 10, 1, 9, 30))
    ingest.main(["--data-dir", str(tmp_path), "--since", "2026-10-01"])
    assert fake_nse == []
    at(monkeypatch, datetime(2026, 10, 1, 18, 5))
    ingest.main(["--data-dir", str(tmp_path), "--since", "2026-10-01"])
    assert fake_nse == [date(2026, 10, 1)]
    assert date(2026, 10, 1) in set(store.read_summary(tmp_path)["trade_date"])
