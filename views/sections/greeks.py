import numpy as np
import streamlit as st

from optlab.models import black_scholes
from ui import charts, theme

spec = st.session_state["spec"]
pal = theme.current()

st.caption("How each sensitivity changes with spot, volatility or time (Black-Scholes, European exercise).")

# name -> (display scale, unit label)
GREEKS = {
    "Delta": (1, "per 1 unit of spot"),
    "Gamma": (1, "per 1 unit of spot"),
    "Vega": (1 / 100, "per 1 vol point"),
    "Theta": (1 / 365, "per calendar day"),
    "Rho": (1 / 100, "per 1 rate point"),
}
c1, c2, _ = st.columns([2, 2, 1])
greek = c1.segmented_control("Greek", list(GREEKS), default="Delta", required=True)
axis = c2.segmented_control(
    "Against", ["Spot", "Volatility", "Days to expiry"], default="Spot", required=True
)
scale, unit = GREEKS[greek]
fn = black_scholes.GREEKS[greek.lower()]

days = max(round(spec.T * 365), 1)
if axis == "Days to expiry":
    x = np.linspace(1, max(days * 2, 30), 200)
    curves = {
        f"Strike {K:,.2f}": fn(spec.S, K, x / 365, spec.r, spec.sigma, spec.q, spec.kind)
        for K in (spec.K * 0.95, spec.K, spec.K * 1.05)
    }
    x_title = "Days to expiry"
else:
    horizons = sorted({max(days // 4, 1), max(days // 2, 1), days})
    if axis == "Spot":
        x = np.linspace(spec.S * 0.7, spec.S * 1.3, 241)
        curves = {
            f"{d} days": fn(x, spec.K, d / 365, spec.r, spec.sigma, spec.q, spec.kind) for d in horizons
        }
        x_title = "Spot"
    else:
        x = np.linspace(0.02, max(1.0, spec.sigma * 2), 200)
        curves = {f"{d} days": fn(spec.S, spec.K, d / 365, spec.r, x, spec.q, spec.kind) for d in horizons}
        x_title = "Volatility"

fig = charts.lines(
    x, {k: v * scale for k, v in curves.items()}, pal, x_title=x_title, y_title=f"{greek} ({unit})"
)
if axis == "Volatility":
    fig.update_xaxes(tickformat=".0%")
    fig.add_vline(x=spec.sigma, line=dict(color=pal.muted, width=1))
elif axis == "Spot":
    fig.add_vline(x=spec.K, line=dict(color=pal.muted, width=1), annotation_text="Strike")
st.plotly_chart(fig, width="stretch")

NOTES = {
    "Delta": "Delta runs from 0 to 1 for a call (−1 to 0 for a put) and sharpens into a step as expiry nears.",
    "Gamma": "Gamma peaks at the money and spikes as expiry nears: short-dated at-the-money options flip delta fast.",
    "Vega": "Vega is largest at the money and for long-dated options; it shrinks like √T.",
    "Theta": "Time decay accelerates into expiry for at-the-money options.",
    "Rho": "Rho grows with time to expiry; it's small for the short-dated options that dominate NSE volume.",
}
st.caption(NOTES[greek])
