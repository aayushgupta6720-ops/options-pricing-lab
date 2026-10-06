from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optlab.calibration import MIN_ABS_DELTA, ROUGH_MAX_DAYS, ROUGH_MIN_DAYS, ROUGH_PATHS
from ui import charts, data, theme

st.caption(
    "In a rough volatility model (rough Bergomi) the at-the-money skew keeps steepening as expiries get shorter; in "
    f"Heston it levels off. Both are fitted to NIFTY's expiries {ROUGH_MIN_DAYS} to {ROUGH_MAX_DAYS} days out to see "
    "which matches the market. Roughness H runs from 0.5 (a classical model) down towards 0 (very rough)."
)

fits = data.load_or_stop(data.rough_fits, "NIFTY")
if fits.empty:
    st.info("No rough Bergomi fits yet.")
    st.stop()
expiries = data.load_or_stop(data.rough_expiries, "NIFTY")
days = sorted(fits["trade_date"], reverse=True)
c1, _ = st.columns([1, 3])
day = c1.selectbox("Trading day", days, format_func=lambda d: f"{d:%a %d %b %Y}", key="rough_day")
fit = fits.set_index("trade_date").loc[day]
today = expiries[expiries["trade_date"] == day].sort_values("T")
pal = theme.current()

k = st.columns(3)
k[0].metric(
    "Roughness H", f"{fit['H']:.2f}", help="Hurst exponent: 0.5 is a classical diffusion, lower is rougher."
)
k[1].metric("Vol of vol", f"{fit['eta']:.2f}", help="η: how strongly volatility itself moves.")
k[2].metric(
    "Spot-vol correlation",
    f"{fit['rho']:+.2f}".replace("-", "−"),
    help="ρ. Negative: vol rises when the market falls.",
)
k = st.columns(3)
k[0].metric(
    "Rough Bergomi error",
    f"{100 * fit['rmse']:.2f} vol pts",
    help="Root-mean-square gap between the model's implied vols and the market's, over all fitted quotes.",
)
k[1].metric(
    "Heston error", f"{100 * fit['heston_rmse']:.2f} vol pts", help="Heston fitted to the same quotes."
)
k[2].metric(
    "Shape error: rough / Heston",
    f"{100 * fit['shape_rmse']:.2f} / {100 * fit['heston_shape_rmse']:.2f} vol pts",
    help="RMSE after removing each expiry's average miss: how well each model gets the smile's shape, setting "
    "aside its level (rough Bergomi fits one forward-variance level per expiry; Heston can't).",
)

# --- Skew against maturity, log-log ---------------------------------------------------------
st.subheader("At-the-money skew against maturity")
fig = go.Figure()
x = today["T"] * 365
series = [("Market (SABR)", "skew_market", "markers"), ("Rough Bergomi", "skew_rough", "lines+markers"),
          ("Heston", "skew_heston", "lines+markers")]  # fmt: skip
for name, column, mode in series:
    i = charts.MODEL_SLOTS[name.split(" (")[0]]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=today[column].abs(),
            name=name,
            mode=mode,
            line=dict(color=pal.series[i], width=charts.LINE),
            marker=dict(size=charts.MARKER + (2 if i == 0 else 0), color=pal.series[i]),
            hovertemplate=f"%{{x:.0f}} days<br>|skew| %{{y:.3f}}<extra>{name}</extra>",
        )
    )
market = today.dropna(subset=["skew_market"])
charts.log_ticks(
    fig, "x", [d for d in (2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90) if x.min() * 0.8 <= d <= x.max() * 1.25]
)
fig.update_xaxes(title_text="Days to expiry")
fig.update_yaxes(type="log", title_text="At-the-money skew (absolute)")
st.plotly_chart(theme.style(fig, pal, 380), width="stretch")
slope_note = ""
if len(market) >= 3:
    slope = np.polyfit(np.log(market["T"]), np.log(market["skew_market"].abs()), 1)[0]
    slope_note = (
        f" This day the market's points have slope {slope:+.2f}, which reads as H ≈ {0.5 + slope:.2f}; "
        "one day's estimate is noisy."
    ).replace("-", "−")
st.caption("On log-log axes a power law is a straight line, with slope H − ½." + slope_note)

# --- One short expiry -----------------------------------------------------------------------
st.subheader("A short expiry up close")
quotes = data.load_or_stop(data.rough_quotes, "NIFTY", day)
if not quotes.empty:
    choices = list(today["expiry"])
    expiry = st.selectbox(
        "Expiry",
        choices,
        format_func=lambda e: f"{e:%d %b %Y} ({(e - day).days}d)",
        key=f"rough_expiry_{day}",
    )
    q = quotes[quotes["expiry"] == expiry].sort_values("strike")
    curves = {"Rough Bergomi": (q["strike"], q["rough_iv"]), "Heston": (q["strike"], q["heston_iv"])}
    smile = charts.model_smile(q, curves, pal)
    st.plotly_chart(smile, width="stretch")

# --- Across the whole history ---------------------------------------------------------------
st.subheader("Which model gets the shape right, by maturity")
buckets = pd.cut(
    expiries["T"] * 365, [0, 7, 14, 30, 91], labels=["under a week", "1–2 weeks", "2–4 weeks", "1–3 months"]
)
by_bucket = expiries.groupby(buckets, observed=True)[["rough_shape", "heston_shape"]].median()
bars = go.Figure()
for name, column in (("Rough Bergomi", "rough_shape"), ("Heston", "heston_shape")):
    bars.add_trace(
        go.Bar(
            x=list(by_bucket.index.astype(str)),
            y=by_bucket[column],
            name=name,
            marker_color=pal.series[charts.MODEL_SLOTS[name]],
            hovertemplate=f"%{{x}}<br>median shape error %{{y:.2%}}<extra>{name}</extra>",
        )
    )
bars.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
bars.update_yaxes(title_text="Median shape error (vol)", tickformat=".2%")
bars.update_xaxes(title_text="Expiry")
st.plotly_chart(theme.style(bars, pal, 340), width="stretch")
st.caption(
    f"Median over {fits['trade_date'].nunique()} trading days of each expiry's shape error: its misses after "
    "removing their average, in vol points. Lower is better."
)

st.subheader("Fitted roughness over time")
RANGES = {"6 months": 182, "1 year": 365, "All": None}
span = st.segmented_control("Range", list(RANGES), default="All", required=True, key="rough_range")
history = (
    fits
    if RANGES[span] is None
    else fits[fits["trade_date"] >= fits["trade_date"].max() - timedelta(days=RANGES[span])]
)


def implied_by_slope(rows: pd.DataFrame) -> float:
    rows = rows.dropna(subset=["skew_market"])
    if len(rows) < 3:
        return np.nan
    return 0.5 + np.polyfit(np.log(rows["T"]), np.log(rows["skew_market"].abs()), 1)[0]


slope_h = expiries.groupby("trade_date").apply(implied_by_slope, include_groups=False).sort_index()
# A single day's slope is noisy (one near-symmetric smile can flip it); show a 20-day rolling median.
slope_h = slope_h.rolling(20, min_periods=5).median()
history = history.assign(slope_h=history["trade_date"].map(slope_h))
hx = pd.to_datetime(history["trade_date"])
c1, c2 = st.columns(2)
with c1:
    fig = charts.lines(
        hx,
        {
            "From the market's skew slope (20-day median)": history["slope_h"],
            "Fitted by rough Bergomi": history["H"],
        },
        pal,
        y_title="H",
        hover_format=".2f",
        height=300,
        slots=[charts.MODEL_SLOTS["Market"], charts.MODEL_SLOTS["Rough Bergomi"]],
    )
    fig.add_vline(x=pd.Timestamp(day), line=dict(color=pal.muted, width=1))
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = charts.lines(
        hx, {"Rough Bergomi": history["rmse"], "Heston": history["heston_rmse"]}, pal, y_title="Fit error (vol)",
        percent_y=True, height=300, slots=[charts.MODEL_SLOTS["Rough Bergomi"], charts.MODEL_SLOTS["Heston"]],
    )  # fmt: skip
    fig.add_vline(x=pd.Timestamp(day), line=dict(color=pal.muted, width=1))
    st.plotly_chart(fig, width="stretch")

st.caption(
    "One day's fitted H is noisy, because H, vol of vol and the variance levels can trade off against each other. "
    "The level it hovers around is the steadier reading."
)

with st.expander("How this is computed"):
    st.markdown(
        f"""
- Rough Bergomi's variance is a forward-variance curve times the exponential of a rough (fractional)
  Brownian motion: roughness H, vol of vol η, spot-vol correlation ρ. It predicts an at-the-money skew
  that grows like T^(H − ½) as maturity T shrinks; Heston's flattens out instead.
- It has no pricing formula, so it's priced by Monte Carlo: the hybrid scheme, {ROUGH_PATHS:,} paths, four
  time steps a day, and the same random numbers every day so day-to-day changes aren't noise.
- The fit finds H, η, ρ and one forward-variance level per expiry, for quotes between the {MIN_ABS_DELTA:.0%}-delta
  put and call. Heston is refitted to exactly the same quotes, so the errors compare like with like.
- Skew is the slope of implied vol against log-strike at the money: a central difference over ±¼ standard
  deviation, for the market (via its SABR fit) and for both models.
- The fits run in the daily data job; this page only reads them.
"""
    )

with st.expander("Data: this day's expiries and the fit history"):
    st.dataframe(today.drop(columns=["trade_date", "underlying"]), hide_index=True)
    st.dataframe(
        fits.drop(columns=["xi_times", "xi_values", "underlying", "fitted"]).sort_values(
            "trade_date", ascending=False
        ),
        hide_index=True,
    )
