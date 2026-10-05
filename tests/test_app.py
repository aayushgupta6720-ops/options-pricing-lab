"""Every page renders without an exception, against a small dataset built from the fixture."""

from datetime import date
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from optlab.market import nse, store
from scripts.ingest import process_day
from ui import data

FIXTURES = Path(__file__).parent / "fixtures"
APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["views/surface.py", "views/history.py", "views/strategy.py", "views/pricer.py", "views/greeks.py",
         "views/convergence.py"]  # fmt: skip


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("market-data")
    raw = nse.parse((FIXTURES / "fo_20261001_subset.csv.zip").read_bytes())
    chains, rows = process_day(date(2026, 10, 1), raw, 0.1446)
    store.save(root, chains, rows)
    return root


@pytest.fixture(autouse=True)
def local_data(dataset, monkeypatch):
    monkeypatch.setattr(data, "LOCAL_DIR", dataset)
    st.cache_data.clear()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    app = AppTest.from_file(APP, default_timeout=60)
    app.switch_page(page)
    app.run()
    assert not app.exception, app.exception[0].value if app.exception else ""
    assert app.title


def test_pricer_loads_the_market_option_and_matches_its_close():
    app = AppTest.from_file(APP, default_timeout=60)
    app.switch_page("views/pricer.py")
    app.run()
    assert app.session_state["market"]["label"].startswith("NIFTY")
    close = app.session_state["market"]["market_price"]
    metrics = {m.label: m.value for m in app.metric}
    model = float(metrics["Black-Scholes at the market's implied vol"].lstrip("₹").replace(",", ""))
    assert model == pytest.approx(close, rel=0.01)


def test_hull_button_loads_the_textbook_example():
    app = AppTest.from_file(APP, default_timeout=60)
    app.switch_page("views/pricer.py")
    app.run()
    app.button[1].click().run()
    assert app.session_state["in_S"] == 42.0
    prices = app.dataframe[0].value.set_index("Model")["Price"]
    # Hull's 4.76 is for T = 0.5 years; the app takes whole days, so 183 days is T = 0.5014.
    assert prices["Black-Scholes"] == pytest.approx(4.76, abs=0.01)
