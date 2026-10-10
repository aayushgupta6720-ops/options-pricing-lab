"""The app's pages, driven through Streamlit's AppTest against a small dataset built from the fixture.

Every test points the remote URL at a closed port, so anything that tried to fetch from GitHub
instead of the local dataset would fail.
"""

import json
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from optlab.market import live, nse, store
from scripts import calibrate
from scripts.ingest import process_day
from ui import data

FIXTURES = Path(__file__).parent / "fixtures"
APP = str(Path(__file__).resolve().parents[1] / "app.py")
VOLATILITY, MODELS, PRICER = "views/volatility.py", "views/models.py", "views/pricer.py"
STRATEGY, EXOTICS = "views/strategy.py", "views/exotics.py"
SECTIONS = [(VOLATILITY, "Surface"), (VOLATILITY, "History"), (MODELS, "Heston & SABR"), (MODELS, "Rough volatility"),
            (STRATEGY, None), (PRICER, "Prices"), (PRICER, "Greeks"), (PRICER, "Convergence"),
            (PRICER, "Heston model"), (EXOTICS, None)]  # fmt: skip
DAYS = (date(2026, 9, 30), date(2026, 10, 1))


def build_dataset(root: Path, fitted: bool = True, models: str = "heston,rough") -> Path:
    raw = nse.parse((FIXTURES / "fo_20261001_subset.csv.zip").read_bytes())
    for day in DAYS:
        raw["trade_date"] = day
        chains, rows = process_day(day, raw, 0.1446)
        store.save(root, chains, rows)
    if fitted:
        calibrate.main(["--data-dir", str(root), "--models", models])
    return root


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("market-data"))


@pytest.fixture(autouse=True)
def local_data(dataset, monkeypatch):
    monkeypatch.setattr(data, "LOCAL_DIR", dataset)
    monkeypatch.setattr(data, "REMOTE_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(data, "LIVE_PRICES", False)  # tests that want a live price use serve_live()
    # AppTest swaps sys.modules["__main__"] for the app script and leaves it there. A process pool
    # started later in the same session (scripts/calibrate.py's tests) would then re-run app.py in
    # every worker it spawns.
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    st.cache_data.clear()
    st.cache_resource.clear()


def open_page(page: str, tab: str | None = None) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60)
    app.switch_page(page)
    if tab:
        app.query_params["tab"] = tab  # what a link to /page?tab=... does
    app.run()
    assert not app.exception, app.exception[0].value if app.exception else ""
    return app


def serve_live(monkeypatch, move: float = 0.02, when: datetime = datetime(2026, 10, 2, 11, 5)):
    """Live prices `move` above each underlying's last close in the dataset (1 Oct), timed `when` IST."""
    monkeypatch.setattr(data, "LIVE_PRICES", True)

    def latest(underlying):
        close = float(data.latest_chain(underlying)[1]["spot"].iloc[0])
        return live.Quote(underlying, close * (1 + move), when.replace(tzinfo=live.IST), "NSE")

    monkeypatch.setattr(data.live, "latest", latest)
    st.cache_data.clear()


def captions(app: AppTest, text: str) -> list[str]:
    return [c.value for c in app.caption if text in c.value]


def metrics(app: AppTest) -> dict:
    return {m.label: m.value for m in app.metric}


@pytest.mark.parametrize("page, tab", SECTIONS, ids=lambda x: x and x.split("/")[-1])
def test_page_renders(page, tab):
    app = open_page(page, tab)
    assert app.title
    if tab:
        assert [t.label for t in app.tabs if t.label == tab]


def test_only_the_open_tab_runs():
    app = open_page(PRICER)
    assert "Delta" in metrics(app) and "Black-Scholes price" not in metrics(app)
    app = open_page(PRICER, "Convergence")
    assert "Black-Scholes price" in metrics(app) and "Delta" not in metrics(app)


def test_choices_inside_a_tab_survive_a_visit_to_another_tab():
    app = open_page(VOLATILITY, "Surface")
    app.selectbox(key="surface_day").set_value(DAYS[0]).run()
    app.selectbox(key="vol_underlying").set_value("BANKNIFTY").run()
    for tab in ("History", "Surface"):
        app.session_state["tab"] = tab
        app.run()
        assert not app.exception
    assert app.selectbox(key="surface_day").value == DAYS[0]
    assert app.selectbox(key="vol_underlying").value == "BANKNIFTY"


def test_sidebar_shows_the_essentials_and_folds_the_rest():
    app = open_page(PRICER)
    labels = [n.label for n in app.sidebar.number_input]
    assert labels == [
        "Spot",
        "Strike",
        "Days to expiry",
        "Volatility (%)",
        "Risk-free rate (%)",
        "Dividend yield (%)",
    ]
    assert [e.label for e in app.sidebar.expander] == ["Rate, dividend, exercise"]


def test_pricer_loads_the_market_option_and_matches_its_close():
    app = open_page(PRICER)
    assert app.session_state["market"]["label"].startswith("NIFTY")
    close = app.session_state["market"]["market_price"]
    model = float(metrics(app)["Black-Scholes at the market's implied vol"].lstrip("₹").replace(",", ""))
    assert model == pytest.approx(close, rel=0.01)


def test_market_comparison_disappears_once_the_inputs_change():
    app = open_page(PRICER)
    app.number_input(key="in_sigma").set_value(30.0).run()
    assert "NSE close" not in metrics(app)


def test_hull_button_loads_the_textbook_example():
    app = open_page(PRICER)
    app.button[1].click().run()
    assert app.session_state["in_S"] == 42.0
    prices = app.dataframe[0].value.set_index("Model")["Price"]
    # Hull's 4.76 is for T = 0.5 years; the app takes whole days, so 183 days is T = 0.5014.
    assert prices["Black-Scholes"] == pytest.approx(4.76, abs=0.01)


def test_sidebar_inputs_survive_a_visit_to_a_market_page():
    app = open_page(PRICER)
    app.button[1].click().run()
    app.number_input(key="in_sigma").set_value(35.0).run()
    app.segmented_control(key="in_style").set_value("american").run()
    for page in (VOLATILITY, PRICER):
        app.switch_page(page).run()
        assert not app.exception
    assert (app.session_state["in_S"], app.session_state["in_sigma"], app.session_state["in_style"]) == (
        42.0,
        35.0,
        "american",
    )


def test_american_gamma_comes_from_the_tree():
    app = open_page(PRICER)
    app.button[1].click().run()
    app.segmented_control(key="in_style").set_value("american").run()
    gamma = float(metrics(app)["Gamma"])
    assert gamma == pytest.approx(0.0499, abs=0.001)  # no-dividend call: same as the European one


@pytest.mark.parametrize("tab", ["Prices", "Convergence"])
@pytest.mark.parametrize("style", ["european", "american"])
def test_low_volatility_does_not_crash_the_trees(tab, style):
    app = open_page(PRICER, tab)
    app.button[1].click().run()
    app.segmented_control(key="in_style").set_value(style).run()
    for sigma in (2.0, 0.5, 0.0):
        app.number_input(key="in_sigma").set_value(sigma).run()
        assert not app.exception, (sigma, app.exception[0].value if app.exception else "")


def test_strategy_wing_width_does_not_carry_over_between_underlyings():
    app = open_page(STRATEGY)
    nifty_width = app.number_input(key="strategy_width_NIFTY").value
    app.selectbox(key="strategy_underlying").set_value("RELIANCE").run()
    assert not app.exception
    reliance_width = app.number_input(key="strategy_width_RELIANCE").value
    assert reliance_width < nifty_width / 5
    assert "nan" not in metrics(app)["Net premium received"]


def test_strategy_uses_the_previous_day_when_the_latest_chain_is_missing(tmp_path, monkeypatch):
    root = build_dataset(tmp_path / "data", models="heston")
    for path in (root / "chains").rglob("2026-10.parquet"):
        path.unlink()  # the summary has 1 Oct, the chain files don't (e.g. a stale cache)
    monkeypatch.setattr(data, "LOCAL_DIR", root)
    app = open_page(STRATEGY)
    assert captions(app, "30 Sep 2026")


def test_market_pages_say_when_an_underlying_has_no_data(monkeypatch):
    monkeypatch.setattr(data, "UNDERLYINGS", (*data.UNDERLYINGS, "FINNIFTY"))
    for page, tab in ((VOLATILITY, "Surface"), (VOLATILITY, "History"), (STRATEGY, None)):
        app = open_page(page, tab)
        app.selectbox[0].set_value("FINNIFTY").run()
        assert not app.exception, (page, tab)
        assert any("FINNIFTY" in i.value for i in app.info), (page, tab)


def test_offline_the_lab_still_works_and_market_pages_explain(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "LOCAL_DIR", tmp_path / "missing")
    app = open_page(PRICER)
    assert app.session_state["in_S"] == 42.0  # fell back to the textbook example
    assert any("can't be reached" in w.value for w in app.sidebar.warning)
    app = open_page(VOLATILITY)
    assert any("can't be reached" in e.value for e in app.error)


def test_local_dataset_never_falls_back_to_github(tmp_path, monkeypatch):
    root = build_dataset(tmp_path / "data", models="heston")
    shutil.rmtree(root / "chains" / "RELIANCE")
    monkeypatch.setattr(data, "LOCAL_DIR", root)
    assert data.chain("RELIANCE", DAYS[-1]).empty  # missing locally, and not fetched remotely
    assert np.isfinite(data.summary()["spot"]).any()


def test_model_page_shows_the_fit_and_both_models():
    app = open_page(MODELS)
    labels = metrics(app)
    assert float(labels["Fit error (vol pts)"]) < 1.5
    assert float(labels["Spot-vol correlation"].replace("−", "-")) < 0
    smile = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert {"Market", "Heston", "SABR"} <= {trace["name"] for trace in smile["data"]}


def test_history_draws_the_variance_swap_rate_and_copes_with_older_data(tmp_path, monkeypatch):
    app = open_page(VOLATILITY, "History")
    lines = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert {"30-day at-the-money", "30-day variance swap", "20-day realized"} <= {
        t["name"] for t in lines["data"]
    }

    # A summary written before vs_vol_30d existed (the deployed data, until it's rebuilt).
    older = tmp_path / "older"
    shutil.copytree(data.LOCAL_DIR, older)
    store.write_summary(older, store.read_summary(older).drop(columns="vs_vol_30d"))
    monkeypatch.setattr(data, "LOCAL_DIR", older)
    st.cache_data.clear()
    app = open_page(VOLATILITY, "History")
    lines = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert "30-day variance swap" not in {t["name"] for t in lines["data"]}


def test_model_page_before_any_fits(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "LOCAL_DIR", build_dataset(tmp_path / "data", fitted=False))
    app = open_page(MODELS)
    assert any("No model fits" in i.value for i in app.info)


def test_heston_page_presets_and_monte_carlo():
    app = open_page(PRICER, "Heston model")
    fitted = app.session_state["hs_vol0"]
    app.button(key="hs_load_textbook").click().run()
    assert not app.exception
    assert app.session_state["hs_vol0"] == pytest.approx(100 * np.sqrt(0.0175), abs=0.05)
    assert app.session_state["hs_vol0"] != fitted
    assert any("containing" in c.value for c in app.caption)


def test_rough_page_shows_the_fit_and_the_skew_term_structure():
    app = open_page(MODELS, "Rough volatility")
    assert 0.01 <= float(metrics(app)["Roughness H"]) <= 0.5
    skew = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert [t["name"] for t in skew["data"]] == ["Market (SABR)", "Rough Bergomi", "Heston"]
    assert skew["layout"]["xaxis"]["type"] == "log" and skew["layout"]["yaxis"]["type"] == "log"


def test_exotics_page_prices_under_all_four_models():
    app = open_page(EXOTICS)
    app.selectbox(key="ex_product").set_value("Asian (geometric average)").run()
    assert not app.exception
    table = app.dataframe[0].value.set_index("Model")
    assert list(table.index) == ["Black-Scholes", "Heston", "Local vol", "Rough Bergomi"]
    bs = table.loc["Black-Scholes"]
    assert abs(bs["Price"] - bs["Black-Scholes formula"]) < bs["± 95%"] * 2  # 4 standard errors
    reductions = app.dataframe[1].value.set_index("Method")["Variance reduction"]
    assert reductions["Sobol + PCA + control variate"] > reductions["Plain Monte Carlo"]
    assert app.dataframe[1].value["Paths"].max() == 32_000  # the table is the largest run only
    errors = json.loads(app.get("plotly_chart")[-1].proto.spec)  # error against paths
    assert [t["name"] for t in errors["data"]] == list(reductions.index)
    for trace in errors["data"]:
        assert len(trace["x"]) == 3 and trace["y"] == sorted(trace["y"], reverse=True), trace["name"]


def test_exotics_page_barrier_controls():
    app = open_page(EXOTICS)
    app.selectbox(key="ex_product").set_value("Barrier").run()
    assert not app.exception
    assert app.segmented_control(key="ex_knock").value == "out"
    app.segmented_control(key="ex_knock").set_value("in").run()
    assert not app.exception and len(app.dataframe[0].value) == 4


def test_pricer_loads_the_live_at_the_money_option(monkeypatch):
    serve_live(monkeypatch, move=0.02)
    app = open_page(PRICER)
    close = float(data.latest_chain("NIFTY")[1]["spot"].iloc[0])
    ss = app.session_state
    assert ss["in_S"] == pytest.approx(close * 1.02)
    assert ss["in_K"] > close * 1.01  # the strike follows the live price
    label = captions(app.sidebar, "live from NSE")[0]  # "Loaded NIFTY 23,300 call, 27 Oct 2026 at ..."
    expiry = datetime.strptime(label.split(", ")[1][:11], "%d %b %Y").date()
    assert ss["in_days"] == (expiry - date(2026, 10, 2)).days
    assert "NSE close" not in metrics(app)  # no closing price to compare a live option with
    assert not app.exception


def test_a_live_price_moves_the_vol_by_how_vol_has_moved_with_the_index(monkeypatch):
    serve_live(monkeypatch, move=0.02)
    flat = open_page(PRICER).session_state["in_sigma"]  # this dataset is too short for a slope: 0
    monkeypatch.setattr(data.surface, "spot_vol_slope", lambda rows, days: -1.0)
    app = open_page(PRICER)
    assert app.session_state["in_sigma"] == pytest.approx(flat - 100 * np.log(1.02), abs=0.01)
    assert captions(app.sidebar, "moved -1.98 vol pts for the move since")


@pytest.mark.parametrize(
    "move, when",
    [(0.02, datetime(2026, 10, 1, 15, 30)), (0.5, datetime(2026, 10, 2, 11, 5))],
    ids=["same day", "implausible"],
)
def test_live_prices_that_add_nothing_are_ignored(monkeypatch, move, when):
    serve_live(monkeypatch, move, when)
    app = open_page(PRICER)
    assert "NSE close" in metrics(app) and not captions(app.sidebar, "live from")


def test_strategy_centres_on_the_live_price_and_keeps_it_until_refreshed(monkeypatch):
    serve_live(monkeypatch, move=0.03)
    app = open_page(STRATEGY)
    close = float(data.latest_chain("NIFTY")[1]["spot"].iloc[0])
    assert captions(app, f"NIFTY {close * 1.03:,.2f}, live from NSE")
    strikes = app.dataframe[0].value["Strike"]
    assert strikes.mean() > close * 1.015  # the iron condor is centred on the live price
    assert "nan" not in metrics(app)["Net premium received"]

    serve_live(monkeypatch, move=0.04)
    app.slider[0].set_value(1).run()  # any interaction: the price stays the one first taken
    assert captions(app, f"NIFTY {close * 1.03:,.2f}, live from NSE")
    app.button(key="strategy_refresh").click().run()
    assert captions(app, f"NIFTY {close * 1.04:,.2f}, live from NSE") and not app.exception

    app.selectbox(key="strategy_underlying").set_value("RELIANCE").run()  # stocks stay at the close
    assert captions(app, "the close on 01 Oct 2026") and not captions(app, "live from")
    assert not [b for b in app.button if b.key == "strategy_refresh"] and not app.exception
