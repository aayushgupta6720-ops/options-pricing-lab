from datetime import timedelta

import pandas as pd
import streamlit as st

from optlab import variance
from optlab.market.store import REALIZED_WINDOW
from ui import charts, data, theme

RANGES = {"3 months": 91, "6 months": 182, "1 year": 365, "All": None}
underlying = st.session_state["vol_underlying"]  # chosen above the tabs (views/volatility.py)
span = st.segmented_control("Range", list(RANGES), default="1 year", required=True, key="history_range")
st.caption("One point per trading day, back to July 2024.")

rows = data.load_or_stop(data.rows_for, underlying)
if rows.empty:
    st.info(f"No data for {underlying} yet.")
    st.stop()
# Volatility that actually followed: realized vol over the next 20 trading days.
rows["rv_next_20d"] = rows["rv_20d"].shift(-REALIZED_WINDOW)
if RANGES[span]:
    rows = rows[rows["trade_date"] >= rows["trade_date"].max() - timedelta(days=RANGES[span])]
pal = theme.current()
x = pd.to_datetime(rows["trade_date"])

st.subheader("Implied vs realized")
has_swap = "vs_vol_30d" in rows and rows["vs_vol_30d"].notna().any()
series = {"30-day at-the-money": rows["atm_iv_30d"]}
if has_swap:
    series["30-day variance swap"] = rows["vs_vol_30d"]
series["20-day realized"] = rows["rv_20d"]
if underlying == "NIFTY":
    series["India VIX"] = rows["india_vix"]
st.plotly_chart(
    charts.lines(x, series, pal, y_title="Annualised volatility", percent_y=True), width="stretch"
)

if has_swap:
    swap = variance.premium(rows["vs_vol_30d"], rows["rv_next_20d"], block=REALIZED_WINDOW)
    atm = variance.premium(rows["atm_iv_30d"], rows["rv_next_20d"], block=REALIZED_WINDOW)
    if swap["days"] >= 2 * REALIZED_WINDOW:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Variance premium",
            f"{100 * swap['premium']:+.2f} vol pts",
            help="What selling 30-day variance at the variance-swap rate earned against the variance that "
            "followed, as the gap between their root-mean-square vols.",
        )
        c2.metric(
            "95% interval (vol pts)",
            f"{100 * swap['low']:+.1f} to {100 * swap['high']:+.1f}",
            help="A moving-block bootstrap with 20-day blocks: consecutive days share most of their realized "
            "window, so the sample holds far fewer independent periods than days.",
        )
        c3.metric(
            "At-the-money instead",
            f"{100 * atm['premium']:+.2f} vol pts",
            help="The same measure with 30-day at-the-money vol as the rate. It sits below the variance-swap "
            "rate because the strip includes the expensive put wing.",
        )
        st.caption(
            f"Over {swap['days']:,} days. Sellers of volatility are usually paid for the risk, but a range that "
            "includes zero means this sample can't tell the premium from luck. The share of days implied vol "
            "beat what followed isn't shown: realized vol is right-skewed, so even a constant forecast beats "
            "it on about two days in three. The last 20 days aren't compared yet."
        )

c1, c2 = st.columns(2)
with c1:
    st.subheader("Term structure", help="At-the-money implied vol at fixed maturities.")
    st.plotly_chart(
        charts.lines(
            x,
            {"7-day": rows["atm_iv_7d"], "30-day": rows["atm_iv_30d"], "90-day": rows["atm_iv_90d"]},
            pal,
            y_title="At-the-money implied vol",
            percent_y=True,
        ),
        width="stretch",
    )
with c2:
    st.subheader("Skew (30-day)", help="25-delta put vol minus 25-delta call vol, 30 days out.")
    st.plotly_chart(
        charts.lines(
            x, {"Put minus call vol": rows["skew_25d_30d"]}, pal, y_title="Vol points", percent_y=True
        ),
        width="stretch",
    )

st.subheader("Spot")
st.plotly_chart(
    charts.lines(x, {underlying: rows["spot"]}, pal, y_title="Close", hover_format=",.2f"), width="stretch"
)

with st.expander("Data"):
    st.dataframe(rows.sort_values("trade_date", ascending=False), hide_index=True)
