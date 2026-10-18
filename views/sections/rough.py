from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optlab.calibration import (
    FINE_DAYS,
    FINE_STEPS,
    MIN_ABS_DELTA,
    ROUGH_MAX_DAYS,
    ROUGH_MIN_DAYS,
    ROUGH_PATHS,
    ROUGH_SEEDS,
)
from optlab.market.calendar import SESSIONS_PER_YEAR
from ui import charts, data, theme

st.caption(
    "In a rough volatility model (rough Bergomi) the at-the-money skew keeps steepening as expiries get shorter; in "
    f"Heston it levels off. Both are fitted to NIFTY's expiries {ROUGH_MIN_DAYS} to {ROUGH_MAX_DAYS} days out to see "
    "which matches the market, with time counted in trading days, so a weekend isn't two days of variance. "
    "Roughness H runs from 0.5 (a classical model) down towards 0 (very rough)."
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

seed_h = list(fit["H_seeds"]) if "H_seeds" in fit and isinstance(fit["H_seeds"], (list, np.ndarray)) else []
k = st.columns(3)
k[0].metric(
    "Roughness H",
    f"{fit['H']:.2f}",
    help="Hurst exponent: 0.5 is a classical diffusion, lower is rougher. The median of the day's fits with "
    f"{len(ROUGH_SEEDS)} sets of random numbers"
    + (f"; they range from {min(seed_h):.2f} to {max(seed_h):.2f}." if len(seed_h) > 1 else "."),
)
k[1].metric("Vol of vol", f"{fit['eta']:.2f}", help="η: how strongly volatility itself moves.")
k[2].metric(
    "Spot-vol correlation",
    f"{fit['rho']:+.2f}".replace("-", "−"),
    help="ρ. Negative: vol rises when the market falls.",
)


def pts(value) -> str:
    return "–" if pd.isna(value) else f"{100 * value:.2f}"


k = st.columns(3)
k[0].metric(
    "Rough Bergomi error",
    f"{pts(fit['rmse'])} vol pts",
    help="Root-mean-square gap between the model's implied vols and the market's, over all fitted quotes.",
)
k[1].metric(
    "Heston error: one θ / θ(t)",
    f"{pts(fit['heston_rmse'])} / {pts(fit.get('heston_tt_rmse'))} vol pts",
    help="Heston fitted to the same quotes, first with one long-run variance θ, then with one per expiry, "
    "θ(t). With θ(t) Heston has as many level parameters as rough Bergomi's forward-variance curve, plus one.",
)
k[2].metric(
    "Shape error: rough / Heston θ(t)",
    f"{pts(fit['shape_rmse'])} / {pts(fit.get('heston_tt_shape_rmse'))} vol pts",
    help="RMSE after removing each expiry's average miss: how well each model gets the smile's shape, setting "
    "aside its level. Against Heston θ(t), both models can match each expiry's level, so this is the fair test.",
)

# --- Skew against maturity, log-log ---------------------------------------------------------
st.subheader("At-the-money skew against maturity")
fig = go.Figure()
x = today["T"] * SESSIONS_PER_YEAR
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
            hovertemplate=f"%{{x:.0f}} trading days<br>|skew| %{{y:.3f}}<extra>{name}</extra>",
        )
    )
market = today.dropna(subset=["skew_market"])
charts.log_ticks(
    fig, "x", [d for d in (1, 2, 3, 5, 7, 10, 15, 21, 30, 42, 63) if x.min() * 0.8 <= d <= x.max() * 1.25]
)
fig.update_xaxes(title_text="Trading days to expiry")
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
    if "heston_tt_iv" in q and q["heston_tt_iv"].notna().any():
        curves["Heston θ(t)"] = (q["strike"], q["heston_tt_iv"])
    smile = charts.model_smile(q, curves, pal)
    st.plotly_chart(smile, width="stretch")

# --- Across the whole history ---------------------------------------------------------------
st.subheader("Which model gets the shape right, by maturity")
buckets = pd.cut(
    expiries["T"] * SESSIONS_PER_YEAR,
    [0, 5, 10, 21, 63],
    labels=["under a week", "1–2 weeks", "2–4 weeks", "1–3 months"],
)
shape_columns = {"Rough Bergomi": "rough_shape", "Heston": "heston_shape", "Heston θ(t)": "heston_tt_shape"}
shape_columns = {name: c for name, c in shape_columns.items() if c in expiries and expiries[c].notna().any()}
by_bucket = expiries.groupby(buckets, observed=True)[list(shape_columns.values())].median()
bars = go.Figure()
for name, column in shape_columns.items():
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
    "removing their average, in vol points. Lower is better. Heston θ(t) can match every expiry's level, as rough "
    "Bergomi can, yet improves Heston's shape only a little: what's left is the smile's shape itself."
)

st.subheader("Fitted roughness over time")
RANGES = {"6 months": 182, "1 year": 365, "All": None}
span = st.segmented_control("Range", list(RANGES), default="All", required=True, key="rough_range")
history = (
    fits
    if RANGES[span] is None
    else fits[fits["trade_date"] >= fits["trade_date"].max() - timedelta(days=RANGES[span])]
)


slope_h = data.skew_slope_h("NIFTY")
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
    if "H_seeds" in history:
        seeds = history["H_seeds"].map(
            lambda v: list(v) if isinstance(v, (list, np.ndarray)) and len(v) else None
        )
        has = seeds.notna()
        if has.any():
            colour = pal.series[charts.MODEL_SLOTS["Rough Bergomi"]]
            band = dict(mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
            fig.add_trace(go.Scatter(x=hx[has], y=seeds[has].map(max), **band))
            fig.add_trace(
                go.Scatter(
                    x=hx[has], y=seeds[has].map(min), fill="tonexty", fillcolor=colour, opacity=0.2, **band
                )
            )
    fig.add_vline(x=pd.Timestamp(day), line=dict(color=pal.muted, width=1))
    st.plotly_chart(fig, width="stretch")
with c2:
    errors = {"Rough Bergomi": history["rmse"], "Heston": history["heston_rmse"]}
    if "heston_tt_rmse" in history and history["heston_tt_rmse"].notna().any():
        errors["Heston θ(t)"] = history["heston_tt_rmse"]
    fig = charts.lines(
        hx, errors, pal, y_title="Fit error (vol)", percent_y=True, height=300,
        slots=[charts.MODEL_SLOTS[name] for name in errors],
    )  # fmt: skip
    fig.add_vline(x=pd.Timestamp(day), line=dict(color=pal.muted, width=1))
    st.plotly_chart(fig, width="stretch")

st.caption(
    f"The shaded band is the range of H over the day's {len(ROUGH_SEEDS)} fits with different random numbers. It's "
    "mostly narrow (a median 0.02, half the typical day-to-day move), so the swings are the market's rather than "
    "Monte Carlo noise; zoom in to see it. H, vol of vol and the variance levels still trade off against each other, "
    "so the level H hovers around is the steadier reading."
)

with st.expander("How this is computed"):
    st.markdown(
        f"""
- Rough Bergomi's variance is a forward-variance curve times the exponential of a rough (fractional)
  Brownian motion: roughness H, vol of vol η, spot-vol correlation ρ. It predicts an at-the-money skew
  that grows like T^(H − ½) as maturity T shrinks; Heston's flattens out instead.
- It has no pricing formula, so it's priced by Monte Carlo: the hybrid scheme, {ROUGH_PATHS:,} paths, four
  time steps a trading day ({FINE_STEPS} for expiries within {FINE_DAYS} trading days, where four
  understated a 2-day skew by about a tenth), and the same random numbers every day so day-to-day changes
  aren't noise. Each day is fitted {len(ROUGH_SEEDS)} times with different random numbers; the day's fit
  is the median-H one.
- Time is counted in trading days ({SESSIONS_PER_YEAR} a year), each expiry's implied vol rescaled to keep
  its total variance. In calendar days a Friday's week-out expiry spans a weekend that carries almost no
  variance: NIFTY's 7-day vol averaged 0.950 of its 30-day vol on Fridays and 1.025 on Mondays.
- The fit finds H, η, ρ and one forward-variance level per expiry, for quotes between the {MIN_ABS_DELTA:.0%}-delta
  put and call. Heston is refitted to exactly the same quotes, so the errors compare like with like: once with
  one long-run variance θ, and once with θ stepping at each expiry, θ(t), which gives it a level per expiry too.
  θ(t) keeps Heston's pricing formula: the long-run variance only enters the characteristic function's C
  term, which becomes a sum over the steps.
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
