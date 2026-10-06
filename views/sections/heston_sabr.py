from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

from optlab.calibration import (
    HESTON_MAX_DAYS,
    HESTON_MIN_DAYS,
    MIN_ABS_DELTA,
    calibration_quotes,
    heston_vols,
    sabr_vols,
)
from optlab.implied_vol import implied_vol
from optlab.models import heston, sabr
from optlab.models.sabr import SabrParams
from optlab.surface import expiry_metrics
from ui import charts, data, theme

st.caption(
    "**Heston** describes the whole surface with five parameters; **SABR** fits each expiry's smile separately "
    "with three."
)

c1, c2, _ = st.columns([1, 1, 2])
underlying = c1.selectbox("Underlying", data.UNDERLYINGS, key="models_underlying")
fits = data.load_or_stop(data.heston_fits, underlying)
if fits.empty:
    st.info(f"No model fits for {underlying} yet.")
    st.stop()
days = sorted(fits["trade_date"], reverse=True)
day = c2.selectbox(
    "Trading day", days, format_func=lambda d: f"{d:%a %d %b %Y}", key=f"models_day_{underlying}"
)
fit = fits.set_index("trade_date").loc[day]
params = data.heston_params(fit)
chain = data.load_or_stop(data.chain, underlying, day)
smiles = data.load_or_stop(data.sabr_fits, underlying, day)
if chain.empty:
    st.warning("No quotes stored for this day.")
    st.stop()
pal = theme.current()

# --- The Heston fit in numbers --------------------------------------------------------------
st.subheader("Heston fit")
half_life_days = np.log(2) / params.kappa * 365
k = st.columns(6)
k[0].metric(
    "Fit error (vol pts)",
    f"{100 * fit['rmse']:.2f}",
    help="Root-mean-square gap between Heston's implied vols and the market's.",
)
k[1].metric("Current vol", f"{np.sqrt(params.v0):.1%}", help="√v₀: today's instantaneous volatility.")
k[2].metric("Long-run vol", f"{np.sqrt(params.theta):.1%}", help="√θ: the level volatility drifts back to.")
k[3].metric(
    "Mean reversion",
    f"{params.kappa:.2f}",
    help=f"κ: how fast volatility returns to its long-run level. A shock halves in {half_life_days:.0f} days.",
)
k[4].metric(
    "Vol of vol", f"{params.xi:.2f}", help="ξ: how much the variance itself moves; it curves the smile."
)
k[5].metric(
    "Spot-vol correlation",
    f"{params.rho:+.2f}",
    help="ρ. Negative: vol rises when the market falls, which tilts the smile.",
)
feller = params.feller_ratio
st.caption(
    f"Fitted to {int(fit['n_quotes'])} quotes across {int(fit['n_expiries'])} expiries. Feller ratio {feller:.2f}: "
    + (
        "the variance stays positive."
        if feller >= 1
        else "the variance can touch zero, as usual for an index skew."
    )
)
if params.kappa >= 29.9:
    st.info(
        "Mean reversion is at its cap of 30: with only a few monthly expiries listed, Heston matches the front "
        "month's skew by making vol shocks fade within days. Read it as a fit, not a forecast."
    )

# --- One expiry: market, Heston, SABR -------------------------------------------------------
st.subheader("Smile: market and models")
metrics = expiry_metrics(chain)
expiries = list(metrics.loc[metrics["n_quotes"] >= 5, "expiry"])
in_window = [e for e in expiries if HESTON_MIN_DAYS <= (e - day).days <= HESTON_MAX_DAYS]
expiry = st.selectbox(
    "Expiry",
    expiries,
    index=expiries.index(in_window[0]) if in_window else 0,
    format_func=lambda e: (
        f"{e:%d %b %Y} ({(e - day).days}d)" + ("" if e in in_window else ", not in Heston's fit")
    ),
    key=f"models_expiry_{underlying}_{day}",
)
quotes = chain[chain["expiry"] == expiry].sort_values("strike")
F, T, r = (float(quotes[c].iloc[0]) for c in ("forward", "T", "r"))
grid = np.linspace(quotes["strike"].min(), quotes["strike"].max(), 120)
kinds = np.where(grid < F, "put", "call")
curves = {"Heston": (grid, implied_vol(heston.price(F, grid, T, r, params, kinds), F, grid, T, r, r, kinds))}
smile_row = smiles[smiles["expiry"] == expiry]
if len(smile_row):
    s = smile_row.iloc[0]
    curves["SABR"] = (
        grid,
        sabr.implied_vol(F, grid, T, SabrParams(s["alpha"], s["rho"], s["nu"], s["beta"])),
    )
in_fit = quotes["delta"].abs().to_numpy() >= MIN_ABS_DELTA
st.plotly_chart(charts.model_smile(quotes, curves, pal, fitted=in_fit), width="stretch")
st.caption(
    "Hollow points are far out-of-the-money options (under 5% delta) that trade for a few rupees, often at vols no "
    "model reaches. Both fits leave them out."
)

# --- Where Heston misses --------------------------------------------------------------------
st.subheader("Where Heston misses")
window = calibration_quotes(chain, 0, HESTON_MAX_DAYS).copy()
window["heston"] = heston_vols(window, params)
window["sabr"] = sabr_vols(window, smiles)
atm = metrics.set_index("expiry")["atm_iv"]
fallback = window.groupby("expiry")["iv"].transform("median")
level = window["expiry"].map(atm).fillna(fallback)
window["sd"] = window["log_moneyness"] / (level * np.sqrt(window["T"]))
edges = np.arange(-2.5, 2.51, 0.25)
labels = [f"{(a + b) / 2:+.2f}".replace("-", "−") for a, b in zip(edges[:-1], edges[1:], strict=True)]
window["bucket"] = pd.cut(window["sd"], edges, labels=labels)
window["row"] = [
    f"{e:%d %b} ({(e - day).days}d)" + ("" if HESTON_MIN_DAYS <= (e - day).days else " · not fitted")
    for e in window["expiry"]
]
window["miss"] = window["heston"] - window["iv"]
order = window.sort_values("T")["row"].unique()
table = window.pivot_table(
    index="row", columns="bucket", values="miss", aggfunc="mean", observed=False
).reindex(order)
counts = (
    window.pivot_table(index="row", columns="bucket", values="miss", aggfunc="count", observed=False)
    .reindex(order)
    .fillna(0)
    .astype(int)
)
st.plotly_chart(charts.residual_heatmap(table, counts, pal), width="stretch")
st.caption(
    "Heston's implied vol minus the market's, averaged over the quotes the fit used. Red: Heston too high; blue: too "
    "low. Columns are strikes in standard deviations from the forward, so −1 is a one-standard-deviation put. "
    "Expiries under a week aren't fitted."
)
rmse_by = (
    window.groupby("row", sort=False)
    .apply(
        lambda g: pd.Series(
            {
                "Heston": np.sqrt(np.nanmean((g["heston"] - g["iv"]) ** 2)),
                "SABR": np.sqrt(np.nanmean((g["sabr"] - g["iv"]) ** 2))
                if g["sabr"].notna().any()
                else np.nan,
                "quotes": len(g),
            }
        ),
        include_groups=False,
    )
    .reindex(order)
)

# --- SABR, expiry by expiry -----------------------------------------------------------------
st.subheader("SABR, expiry by expiry")
if smiles.empty:
    st.info("No SABR fits for this day.")
else:
    days_out = (smiles["T"] * 365).round()
    cols = st.columns(3)
    for col, (name, column, fmt, note) in zip(
        cols,
        [
            ("Vol level", "alpha", ".1%", "α: close to the at-the-money vol, since β = 1."),
            ("Skew", "rho", "+.2f", "ρ: more negative means puts dearer relative to calls."),
            ("Vol of vol", "nu", ".2f", "ν: the curvature of the wings; large for short expiries."),
        ],
        strict=True,
    ):
        with col:
            fig = charts.lines(
                days_out, {name: smiles[column]}, pal, x_title="Days to expiry", y_title=name, height=280
            )
            fig.update_traces(mode="lines+markers", marker=dict(size=charts.MARKER))
            fig.update_yaxes(tickformat=fmt)
            st.plotly_chart(fig, width="stretch")
            st.caption(note)

with st.expander("Data: fit errors by expiry, SABR parameters, every quote"):
    st.dataframe(
        rmse_by.rename(columns={"Heston": "Heston RMSE", "SABR": "SABR RMSE"}),
        column_config={
            c: st.column_config.NumberColumn(format="percent") for c in ("Heston RMSE", "SABR RMSE")
        },
    )
    st.dataframe(smiles.drop(columns=["trade_date", "underlying"]), hide_index=True)
    st.dataframe(
        window[["expiry", "strike", "option_type", "close", "iv", "heston", "sabr", "sd"]],
        hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="percent") for c in ("iv", "heston", "sabr")},
    )

# --- How the fitted parameters moved --------------------------------------------------------
st.subheader("Parameters over time")
RANGES = {"6 months": 182, "1 year": 365, "All": None}
span = st.segmented_control("Range", list(RANGES), default="All", required=True, key="models_range")
history = (
    fits
    if RANGES[span] is None
    else fits[fits["trade_date"] >= fits["trade_date"].max() - timedelta(days=RANGES[span])]
)
x = pd.to_datetime(history["trade_date"])


def chart(series: dict, y_title: str, percent=False, fmt=None):
    fig = charts.lines(x, series, pal, y_title=y_title, percent_y=percent, height=280, hover_format=fmt)
    fig.add_vline(x=pd.Timestamp(day), line=dict(color=pal.muted, width=1))
    st.plotly_chart(fig, width="stretch")


c1, c2 = st.columns(2)
with c1:
    chart(
        {"Current vol": np.sqrt(history["v0"]), "Long-run vol": np.sqrt(history["theta"])},
        "Volatility",
        True,
    )
    chart({"Vol of vol": history["xi"]}, "Vol of vol", fmt=".2f")
    chart({"Fit error": history["rmse"]}, "Fit error (vol)", True)
with c2:
    chart({"Spot-vol correlation": history["rho"]}, "Spot-vol correlation", fmt="+.2f")
    chart({"Mean reversion": history["kappa"]}, "Mean reversion", fmt=".2f")
    chart({"Feller ratio": history["feller_ratio"]}, "Feller ratio", fmt=".2f")
st.caption(
    "One fit per trading day, each starting from the previous day's. Current vol tracks the market. Mean-reversion "
    "speed, long-run vol and vol of vol trade off against each other (several combinations fit a surface almost "
    "equally well), so they're noisier."
)

with st.expander("How this is computed"):
    st.markdown(
        f"""
- Heston prices come from its characteristic function (Lewis's single-integral formula, 512-point
  Gauss-Legendre); implied vols are Black's formula inverted on those prices.
- The fit minimises vega-weighted price errors, with each expiry weighted equally. It uses expiries
  {HESTON_MIN_DAYS} days to a year out, and quotes between the {MIN_ABS_DELTA:.0%}-delta put and call.
  Mean reversion κ is capped at 30. Several starting points are tried, including the previous day's fit.
- Parameters, in Heston's notation: current variance v₀, long-run variance θ, mean reversion κ, vol of vol ξ
  and spot-vol correlation ρ. The Feller ratio is 2κθ/ξ²; below 1, the variance can reach zero.
- SABR uses Hagan's formula with β = 1 (α, ρ, ν), fitted to each expiry separately.
- The fits run in the daily data job; this page only reads them.
"""
    )
