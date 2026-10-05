"""The app's pages, driven through Streamlit's AppTest against a small dataset built from the fixture.

Every test points the remote URL at a closed port, so anything that tried to fetch from GitHub
instead of the local dataset would fail.
"""

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from optlab.market import nse, store
from scripts import calibrate
from scripts.ingest import process_day
from ui import data

FIXTURES = Path(__file__).parent / "fixtures"
APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["views/surface.py", "views/history.py", "views/strategy.py", "views/models.py", "views/pricer.py",
         "views/greeks.py", "views/convergence.py", "views/heston.py"]  # fmt: skip
DAYS = (date(2026, 9, 30), date(2026, 10, 1))


def build_dataset(root: Path, fitted: bool = True) -> Path:
    raw = nse.parse((FIXTURES / "fo_20261001_subset.csv.zip").read_bytes())
    for day in DAYS:
        raw["trade_date"] = day
        chains, rows = process_day(day, raw, 0.1446)
        store.save(root, chains, rows)
    if fitted:
        calibrate.main(["--data-dir", str(root)])
    return root


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("market-data"))


@pytest.fixture(autouse=True)
def local_data(dataset, monkeypatch):
    monkeypatch.setattr(data, "LOCAL_DIR", dataset)
    monkeypatch.setattr(data, "REMOTE_URL", "http://127.0.0.1:9")
    # AppTest swaps sys.modules["__main__"] for the app script and leaves it there. A process pool
    # started later in the same session (scripts/calibrate.py's tests) would then re-run app.py in
    # every worker it spawns.
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    st.cache_data.clear()


def open_page(page: str) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60)
    app.switch_page(page)
    app.run()
    assert not app.exception, app.exception[0].value if app.exception else ""
    return app


def metrics(app: AppTest) -> dict:
    return {m.label: m.value for m in app.metric}


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    assert open_page(page).title


def test_pricer_loads_the_market_option_and_matches_its_close():
    app = open_page("views/pricer.py")
    assert app.session_state["market"]["label"].startswith("NIFTY")
    close = app.session_state["market"]["market_price"]
    model = float(metrics(app)["Black-Scholes at the market's implied vol"].lstrip("₹").replace(",", ""))
    assert model == pytest.approx(close, rel=0.01)


def test_market_comparison_disappears_once_the_inputs_change():
    app = open_page("views/pricer.py")
    app.number_input(key="in_sigma").set_value(30.0).run()
    assert "NSE close" not in metrics(app)


def test_hull_button_loads_the_textbook_example():
    app = open_page("views/pricer.py")
    app.button[1].click().run()
    assert app.session_state["in_S"] == 42.0
    prices = app.dataframe[0].value.set_index("Model")["Price"]
    # Hull's 4.76 is for T = 0.5 years; the app takes whole days, so 183 days is T = 0.5014.
    assert prices["Black-Scholes"] == pytest.approx(4.76, abs=0.01)


def test_sidebar_inputs_survive_a_visit_to_a_market_page():
    app = open_page("views/pricer.py")
    app.button[1].click().run()
    app.number_input(key="in_sigma").set_value(35.0).run()
    app.segmented_control(key="in_style").set_value("american").run()
    for page in ("views/surface.py", "views/pricer.py"):
        app.switch_page(page).run()
        assert not app.exception
    assert (app.session_state["in_S"], app.session_state["in_sigma"], app.session_state["in_style"]) == (
        42.0,
        35.0,
        "american",
    )


def test_american_gamma_comes_from_the_tree():
    app = open_page("views/pricer.py")
    app.button[1].click().run()
    app.segmented_control(key="in_style").set_value("american").run()
    gamma = float(metrics(app)["Gamma"])
    assert gamma == pytest.approx(0.0499, abs=0.001)  # no-dividend call: same as the European one


@pytest.mark.parametrize("page", ["views/pricer.py", "views/convergence.py"])
@pytest.mark.parametrize("style", ["european", "american"])
def test_low_volatility_does_not_crash_the_trees(page, style):
    app = open_page(page)
    app.button[1].click().run()
    app.segmented_control(key="in_style").set_value(style).run()
    for sigma in (2.0, 0.5, 0.0):
        app.number_input(key="in_sigma").set_value(sigma).run()
        assert not app.exception, (sigma, app.exception[0].value if app.exception else "")


def test_strategy_wing_width_does_not_carry_over_between_underlyings():
    app = open_page("views/strategy.py")
    nifty_width = app.number_input(key="strategy_width_NIFTY").value
    app.selectbox(key="strategy_underlying").set_value("RELIANCE").run()
    assert not app.exception
    reliance_width = app.number_input(key="strategy_width_RELIANCE").value
    assert reliance_width < nifty_width / 5
    assert "nan" not in metrics(app)["Net premium"]


def test_strategy_uses_the_previous_day_when_the_latest_chain_is_missing(tmp_path, monkeypatch):
    root = build_dataset(tmp_path / "data")
    for path in (root / "chains").rglob("2026-10.parquet"):
        path.unlink()  # the summary has 1 Oct, the chain files don't (e.g. a stale cache)
    monkeypatch.setattr(data, "LOCAL_DIR", root)
    app = open_page("views/strategy.py")
    assert "30 Sep 2026" in app.caption[-1].value


def test_market_pages_say_when_an_underlying_has_no_data(monkeypatch):
    monkeypatch.setattr(data, "UNDERLYINGS", (*data.UNDERLYINGS, "FINNIFTY"))
    for page in ("views/surface.py", "views/history.py", "views/strategy.py"):
        app = open_page(page)
        app.selectbox[0].set_value("FINNIFTY").run()
        assert not app.exception, page
        assert any("FINNIFTY" in i.value for i in app.info), page


def test_offline_the_lab_still_works_and_market_pages_explain(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "LOCAL_DIR", tmp_path / "missing")
    app = open_page("views/pricer.py")
    assert app.session_state["in_S"] == 42.0  # fell back to the textbook example
    assert any("can't be reached" in w.value for w in app.sidebar.warning)
    app = open_page("views/surface.py")
    assert any("can't be reached" in e.value for e in app.error)


def test_local_dataset_never_falls_back_to_github(tmp_path, monkeypatch):
    root = build_dataset(tmp_path / "data")
    shutil.rmtree(root / "chains" / "RELIANCE")
    monkeypatch.setattr(data, "LOCAL_DIR", root)
    assert data.chain("RELIANCE", DAYS[-1]).empty  # missing locally, and not fetched remotely
    assert np.isfinite(data.summary()["spot"]).any()


def test_model_page_shows_the_fit_and_both_models():
    app = open_page("views/models.py")
    labels = metrics(app)
    assert float(labels["Fit error"].split()[0]) < 1.5
    assert float(labels["Spot-vol correlation ρ"].replace("−", "-")) < 0
    smile = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert {"Market", "Heston", "SABR"} <= {trace["name"] for trace in smile["data"]}


def test_model_page_before_any_fits(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "LOCAL_DIR", build_dataset(tmp_path / "data", fitted=False))
    app = open_page("views/models.py")
    assert any("No model fits" in i.value for i in app.info)


def test_heston_page_presets_and_monte_carlo():
    app = open_page("views/heston.py")
    fitted = app.session_state["hs_vol0"]
    app.button(key="hs_load_textbook").click().run()
    assert not app.exception
    assert app.session_state["hs_vol0"] == pytest.approx(100 * np.sqrt(0.0175), abs=0.05)
    assert app.session_state["hs_vol0"] != fitted
    assert any("containing" in c.value for c in app.caption)
