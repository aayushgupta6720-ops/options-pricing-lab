from datetime import timedelta

import pandas as pd
import streamlit as st

from optlab.market.store import REALIZED_WINDOW
from ui import charts, data, theme

st.title("Volatility history")
st.caption("One point per trading day from the daily snapshots, back to July 2024.")

summary = data.summary()
if summary.empty:
    st.error("No market data yet.")
    st.stop()

RANGES = {"3 months": 91, "6 months": 182, "1 year": 365, "All": None}
c1, c2, _ = st.columns([1, 2, 1])
underlying = c1.selectbox("Underlying", data.UNDERLYINGS, key="history_underlying")
span = c2.segmented_control("Range", list(RANGES), default="1 year", required=True, key="history_range")

rows = summary[summary["underlying"] == underlying].sort_values("trade_date").reset_index(drop=True)
# Volatility that actually followed: realized vol over the next 20 trading days.
rows["rv_next_20d"] = rows["rv_20d"].shift(-REALIZED_WINDOW)
if RANGES[span]:
    rows = rows[rows["trade_date"] >= rows["trade_date"].max() - timedelta(days=RANGES[span])]
pal = theme.current()
x = pd.to_datetime(rows["trade_date"])

st.subheader("Implied vs realized")
series = {"30-day ATM implied": rows["atm_iv_30d"], "20-day realized": rows["rv_20d"]}
if underlying == "NIFTY":
    series["India VIX"] = rows["india_vix"]
st.plotly_chart(
    charts.lines(x, series, pal, y_title="Annualised volatility", percent_y=True), width="stretch"
)

paired = rows.dropna(subset=["atm_iv_30d", "rv_next_20d"])
if len(paired) >= 20:
    premium = paired["atm_iv_30d"] - paired["rv_next_20d"]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Implied above what followed",
        f"{(premium > 0).mean():.0%} of days",
        help="How often 30-day implied vol exceeded the realized vol over the next 20 trading days.",
    )
    c2.metric(
        "Average gap", f"{100 * premium.mean():+.2f} pts", help="Implied minus subsequently realized vol."
    )
    c3.metric("Days compared", f"{len(paired):,}")
    st.caption(
        "Option sellers are usually paid a premium for bearing volatility risk, so implied vol tends to sit above "
        "the volatility that's later realized. The last 20 days aren't compared yet: their future isn't known."
    )

c1, c2 = st.columns(2)
with c1:
    st.subheader("Term structure")
    st.plotly_chart(
        charts.lines(
            x,
            {"7-day": rows["atm_iv_7d"], "30-day": rows["atm_iv_30d"], "90-day": rows["atm_iv_90d"]},
            pal,
            y_title="ATM implied vol",
            percent_y=True,
        ),
        width="stretch",
    )
with c2:
    st.subheader("25-delta skew (30-day)")
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
