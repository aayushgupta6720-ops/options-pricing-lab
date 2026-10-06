import numpy as np
import pandas as pd
import streamlit as st

from optlab.surface import expiry_metrics, smile_grid
from ui import charts, data, theme

underlying = st.session_state["vol_underlying"]  # chosen above the tabs (views/volatility.py)
days = data.load_or_stop(data.days_for, underlying)
if not days:
    st.info(f"No data for {underlying} yet.")
    st.stop()
c1, _ = st.columns([1, 3])
day = c1.selectbox("Trading day", days, format_func=lambda d: f"{d:%a %d %b %Y}", key="surface_day")

rows = data.rows_for(underlying).set_index("trade_date")
today = rows.loc[day]
previous = rows[rows.index < day].iloc[-1] if (rows.index < day).any() else None


def delta(column, scale=100, unit=" vol pts"):
    if previous is None or pd.isna(today[column]) or pd.isna(previous[column]):
        return None
    return f"{scale * (today[column] - previous[column]):+.2f}{unit}"


def pct(value):
    return "—" if pd.isna(value) else f"{value:.2%}"


k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Spot", f"{today['spot']:,.2f}", delta("spot", scale=1, unit=""), delta_color="off")
k2.metric(
    "30-day implied vol",
    pct(today["atm_iv_30d"]),
    delta("atm_iv_30d"),
    delta_color="off",
    help="At-the-money implied vol, interpolated to 30 days.",
)
k3.metric(
    "30-day skew",
    pct(today["skew_25d_30d"]),
    delta("skew_25d_30d"),
    delta_color="off",
    help="25-delta put vol minus 25-delta call vol, 30 days out. Positive: downside protection costs more.",
)
k4.metric(
    "20-day realized vol",
    pct(today["rv_20d"]),
    delta("rv_20d"),
    delta_color="off",
    help="How much the price actually moved: the annualised standard deviation of the last 20 daily returns.",
)
k5.metric(
    "India VIX",
    pct(today["india_vix"]),
    delta("india_vix"),
    delta_color="off",
    help="NSE's 30-day volatility index for NIFTY.",
)

chain = data.load_or_stop(data.chain, underlying, day)
if chain.empty:
    st.warning("No usable quotes for this day.")
    st.stop()
metrics = expiry_metrics(chain)
pal = theme.current()

st.subheader("Smile", help="Implied vol against strike, one curve per expiry.")
expiries = list(metrics.loc[metrics["n_quotes"] >= 3, "expiry"])
c1, c2 = st.columns([3, 1])
chosen = c1.multiselect(
    "Expiries",
    expiries,
    default=expiries[:4],
    max_selections=4,
    format_func=lambda e: f"{e:%d %b %Y}",
    help="Up to four at once, so each keeps a distinct colour.",
)
axis = c2.segmented_control(
    "X axis",
    ["strike", "log_moneyness"],
    default="strike",
    required=True,
    format_func={"strike": "Strike", "log_moneyness": "Moneyness"}.get,
)
if chosen:
    st.plotly_chart(charts.smile(chain, chosen, pal, x=axis), width="stretch")
    st.caption("Circles are puts (below the forward), diamonds are calls (above it).")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Term structure", help="At-the-money implied vol against time to expiry.")
    st.plotly_chart(charts.term_structure(metrics, pal), width="stretch")
with c2:
    st.subheader("Surface")
    near = metrics[(metrics["T"] <= 1.0) & (metrics["n_quotes"] >= 8)]
    sd_grid = np.round(np.linspace(-3, 3, 49), 3)
    grid = smile_grid(chain[chain["expiry"].isin(near["expiry"])], sd_grid, standardised=True)
    if len(grid) >= 2:
        days_axis = near.set_index("expiry").loc[grid.index, "T"] * 365
        st.plotly_chart(charts.surface(grid, days_axis, pal), width="stretch")
        st.caption(
            "Strikes are measured in standard deviations from the forward, so short and long expiries line up. "
            "Gaps are strikes with no clean quote."
        )
    else:
        st.info("Need at least two expiries with quotes to draw a surface.")

with st.expander("How this is computed"):
    st.markdown(
        """
- Prices are the official daily closes from NSE's F&O bhavcopy; India VIX comes from NSE's index closes.
- Only out-of-the-money options with at least 20 trades that day are used: puts below the forward, calls above.
- Each expiry's forward comes from put-call parity, and implied vols invert Black's formula on the forward
  (Black-76), so dividends and financing costs never need modelling.
- 30-day figures interpolate total variance between the expiries either side of 30 days. At-the-money vol is
  left blank when the nearest quotes either side of the forward are more than a standard deviation apart.
- The surface's moneyness is ln(K/F) / (at-the-money vol × √T), for expiries up to a year out with 8+ quotes.
"""
    )

with st.expander("Data: per-expiry metrics and every quote"):
    shown = metrics.assign(days=(metrics["T"] * 365).round().astype(int)).drop(columns="T")
    st.dataframe(
        shown,
        hide_index=True,
        column_config={
            c: st.column_config.NumberColumn(format="percent")
            for c in shown.columns
            if "iv" in c or "skew" in c
        },
    )
    st.dataframe(
        chain.drop(columns=["trade_date", "underlying"]),
        hide_index=True,
        column_config={"iv": st.column_config.NumberColumn(format="percent")},
    )
    st.download_button(
        "Download quotes (CSV)", chain.to_csv(index=False), f"{underlying}_{day:%Y%m%d}_iv.csv", "text/csv"
    )
