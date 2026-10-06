"""Options Pricing & Volatility Lab.

Run: streamlit run app.py
The option inputs for the Pricing lab pages live here, in the sidebar, so they carry across those
pages; each page reads the resulting OptionSpec from st.session_state["spec"].

Streamlit drops the state of widgets that aren't drawn in a run: the sidebar inputs while the
viewer is on a market page, and the controls of every tab but the open one (ui/layout.py).
Re-assigning their keys at the top of every run keeps them alive.
"""

import streamlit as st

from optlab.contracts import OptionSpec
from ui import data

st.set_page_config(page_title="Options Lab", page_icon=":material/ssid_chart:", layout="wide")

MARKET = [
    st.Page("views/volatility.py", title="Implied volatility", icon=":material/landscape:", default=True),
    st.Page("views/models.py", title="Models vs market", icon=":material/query_stats:"),
    st.Page("views/strategy.py", title="Strategy builder", icon=":material/stacked_line_chart:"),
]
LAB = [
    st.Page("views/pricer.py", title="Option pricer", icon=":material/calculate:"),
    st.Page("views/exotics.py", title="Exotic options", icon=":material/route:"),
]
page = st.navigation({"NSE market": MARKET, "Pricing lab": LAB})

HULL = {"S": 42.0, "K": 40.0, "days": 183, "sigma": 0.20, "r": 0.10, "q": 0.0, "kind": "call"}
KEEP_ALIVE = ("in_S", "in_K", "in_days", "in_sigma", "in_r", "in_q", "in_kind", "in_style")  # sidebar
KEEP_ALIVE += ("hs_vol0", "hs_volbar", "hs_kappa", "hs_xi", "hs_rho", "hs_paths")  # Heston model tab
KEEP_ALIVE += ("surface_day", "history_range", "models_range", "rough_day", "rough_range")  # other tabs

for key in KEEP_ALIVE:
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]


def load_inputs(values: dict):
    inputs = {
        "in_S": float(values["S"]),
        "in_K": float(values["K"]),
        "in_days": int(values["days"]),
        "in_sigma": round(100 * values["sigma"], 2),
        "in_r": round(100 * values["r"], 2),
        "in_q": round(100 * values["q"], 2),
        "in_kind": values["kind"],
    }
    st.session_state.update(inputs)
    st.session_state.setdefault("in_style", "european")
    # Remember exactly what was loaded, so the Pricer only compares with the market close while
    # the inputs are still that option's.
    st.session_state["market"] = {**values, "inputs": inputs} if "market_price" in values else None
    st.session_state["loaded"] = values.get("note")


def latest_market_option() -> dict | None:
    try:
        return data.market_option()
    except data.DataUnavailable:
        st.sidebar.warning("Market data can't be reached right now, so the textbook example is loaded.")
        return None


if page.title in {p.title for p in LAB}:
    if "in_S" not in st.session_state:
        load_inputs(latest_market_option() or HULL)

    with st.sidebar:
        st.markdown("### Option")
        c1, c2 = st.columns(2)
        if c1.button(
            "NIFTY ATM",
            help="The at-the-money NIFTY option, first expiry 20+ days out, at the live index level (from NSE, "
            "else Yahoo Finance) or, when there's nothing newer, the last close.",
        ):
            market = latest_market_option()
            if market:
                load_inputs(market)
            else:
                st.warning("No market data available.")
        if c2.button(
            "Textbook", help="Hull's textbook example: spot 42, strike 40, rate 10%, vol 20%, 6 months."
        ):
            load_inputs(HULL)

        st.number_input("Spot", min_value=0.01, key="in_S", format="%.2f")
        st.number_input("Strike", min_value=0.01, key="in_K", format="%.2f")
        st.number_input("Days to expiry", min_value=0, max_value=3650, key="in_days")
        st.number_input("Volatility (%)", min_value=0.0, max_value=500.0, step=0.5, key="in_sigma")
        st.segmented_control("Type", ["call", "put"], key="in_kind", format_func=str.title, required=True)
        with st.expander("Rate, dividend, exercise"):
            st.number_input("Risk-free rate (%)", min_value=-5.0, max_value=50.0, step=0.25, key="in_r")
            st.number_input(
                "Dividend yield (%)",
                min_value=-20.0,
                max_value=50.0,
                step=0.25,
                key="in_q",
                help="For a market option it's backed out of the forward price, so it can come out slightly "
                "negative.",
            )
            st.segmented_control(
                "Exercise",
                ["european", "american"],
                key="in_style",
                format_func=str.title,
                required=True,
                help="NSE options are European. American exercise is priced by the binomial tree only.",
            )
        if st.session_state.get("in_style") == "american":
            st.caption("American exercise (set under Rate, dividend, exercise).")
        if st.session_state.get("loaded"):
            st.caption(st.session_state["loaded"])

    ss = st.session_state
    st.session_state["spec"] = OptionSpec(
        S=ss.in_S,
        K=ss.in_K,
        T=ss.in_days / 365,
        r=ss.in_r / 100,
        sigma=ss.in_sigma / 100,
        q=ss.in_q / 100,
        kind=ss.in_kind,
        style=ss.get("in_style") or "european",
    )

page.run()
