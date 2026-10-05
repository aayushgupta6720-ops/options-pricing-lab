import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from optlab.models import binomial, black_scholes, monte_carlo
from ui import charts, theme

spec = st.session_state["spec"].bump(style="european")
pal = theme.current()

st.title("Model convergence")
st.caption(
    "Black-Scholes is exact for a European option, so it's the yardstick: the binomial tree and Monte Carlo "
    "should home in on it as steps and paths grow. American exercise is ignored on this page."
)
if spec.T == 0 or spec.sigma == 0:
    st.info(
        "Pick an option with time and volatility left; at expiry or zero vol there's nothing to converge."
    )
    st.stop()

exact = black_scholes.price_spec(spec)
st.metric("Black-Scholes price", f"{exact:,.6f}")

st.subheader("Binomial tree")
steps = np.unique(np.logspace(1, np.log10(2000), 70).astype(int))
start = time.perf_counter()
prices = np.array([binomial.price_spec(spec, steps=int(n)) for n in steps])
elapsed = time.perf_counter() - start
c1, c2 = st.columns(2)
with c1:
    fig = charts.lines(
        steps, {"Tree price": prices}, pal, x_title="Steps", y_title="Price", hover_format=",.5f"
    )
    fig.update_xaxes(type="log")
    charts.reference_line(fig, exact, "Black-Scholes", pal)
    st.plotly_chart(fig, width="stretch")
with c2:
    error = np.abs(prices - exact)
    guide = error[0] * steps[0] / steps
    fig = charts.lines(
        steps,
        {"|Tree − Black-Scholes|": error, "1 / steps": guide},
        pal,
        x_title="Steps",
        y_title="Absolute error",
        hover_format=".2e",
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log", tickformat=".0e")
    st.plotly_chart(fig, width="stretch")
st.caption(
    f"{len(steps)} trees, {steps[0]} to {steps[-1]} steps, in {elapsed:.2f}s. The error falls roughly like 1/steps, "
    "zig-zagging as the strike falls on or between nodes."
)

st.subheader("Monte Carlo")
n_paths = 200_000
paths = monte_carlo.gbm_paths(spec.S, spec.T, spec.r, spec.sigma, spec.q, n_paths=n_paths, rng=11)
discounted = np.exp(-spec.r * spec.T) * spec.payoff(paths[:, -1])
n = np.arange(1, n_paths + 1)
running_mean = np.cumsum(discounted) / n
running_sq = np.cumsum(discounted**2) / n
running_se = np.sqrt(np.maximum(running_sq - running_mean**2, 0) / np.maximum(n - 1, 1))
idx = np.unique(np.logspace(2, np.log10(n_paths), 200).astype(int)) - 1
x, mid, se = n[idx], running_mean[idx], running_se[idx]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=mid + 1.96 * se, line=dict(width=0), hoverinfo="skip", showlegend=False))
fig.add_trace(
    go.Scatter(
        x=x,
        y=mid - 1.96 * se,
        fill="tonexty",
        fillcolor=pal.series[0] + "33",
        line=dict(width=0),
        name="95% interval",
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=x,
        y=mid,
        name="Running estimate",
        line=dict(color=pal.series[0], width=charts.LINE),
        hovertemplate="%{x:,} paths<br>%{y:,.5f}<extra></extra>",
    )
)
charts.reference_line(fig, exact, "Black-Scholes", pal)
fig.update_xaxes(type="log", title_text="Paths")
fig.update_yaxes(title_text="Price")
st.plotly_chart(theme.style(fig, pal, 380), width="stretch")
final = monte_carlo.MCResult(float(running_mean[-1]), float(running_se[-1]), n_paths)
lo, hi = final.ci95
st.caption(
    f"After {n_paths:,} paths: {final.price:,.5f} ± {1.96 * final.std_error:,.5f} (95%), "
    f"{'containing' if lo <= exact <= hi else 'missing'} the exact price. The interval narrows like 1/√paths: "
    "100× more paths for 10× more accuracy, which is why variance reduction matters."
)
