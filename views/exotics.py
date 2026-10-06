import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optlab import exotics as ex
from optlab.exotics import Exotic
from ui import charts, data, theme

spec = st.session_state["spec"]
pal = theme.current()

st.title("Exotic options")
st.caption(
    "Options whose payoff depends on the whole price path, priced by Monte Carlo under three models: Black-Scholes "
    "at the sidebar's volatility, and Heston and rough Bergomi at their latest NIFTY fits. The rest of the option "
    "comes from the sidebar."
)

days = round(spec.T * 365)
if days < 2:
    st.info("Pick an option with at least two days to expiry in the sidebar.")
    st.stop()

PRODUCTS = {
    "Asian (arithmetic average)": "asian",
    "Asian (geometric average)": "geometric_asian",
    "Barrier": "barrier",
    "Lookback (floating strike)": "lookback",
}
c1, c2, c3, c4 = st.columns(4)
label = c1.selectbox("Product", list(PRODUCTS), key="ex_product")
product = PRODUCTS[label]
kind = c2.segmented_control(
    "Type", ["call", "put"], default=spec.kind, format_func=str.title, required=True, key="ex_kind"
)
barrier, knock = None, None
if product == "barrier":
    default_barrier = round(spec.S * (0.92 if kind == "call" else 1.08), 2)
    barrier = c3.number_input(
        "Barrier", min_value=0.01, value=default_barrier, key=f"ex_barrier_{kind}",
        help="Below spot: a down barrier; above: an up barrier.",
    )  # fmt: skip
    knock = c4.segmented_control("Knock", ["out", "in"], default="out", required=True, key="ex_knock")
    if abs(barrier - spec.S) < 1e-9:
        st.warning("A barrier at the spot is touched immediately; move it away from the spot.")
        st.stop()
exotic = Exotic(product, kind, None if product == "lookback" else spec.K, barrier, knock, fixings=days)
S, K, T, r, q, sigma = spec.S, spec.K, spec.T, spec.r, spec.q, spec.sigma

# --- Price under each model -----------------------------------------------------------------
BS_PATHS, HESTON_PATHS = 100_000, 40_000
rough_paths = 20_000  # simulated 2,000 at a time (ex.stats_rough_bergomi), so memory stays flat


@st.cache_data(show_spinner=False, max_entries=16)
def priced(model: str, exotic: Exotic, S, T, r, q, sigma, model_row: dict | None):
    started = time.perf_counter()
    if model == "Black-Scholes":
        stats = ex.stats_black_scholes_numba(
            S, T, r, q, sigma, BS_PATHS, exotic.fixings, seed=11, barrier=exotic.barrier
        )
    elif model == "Heston":
        params = data.heston_params(model_row)
        stats = ex.stats_heston_numba(
            S, T, r, q, params, HESTON_PATHS, exotic.fixings, seed=12, barrier=exotic.barrier
        )
    else:
        params, xi = data.rough_model(model_row)
        stats = ex.stats_rough_bergomi(S, T, r, q, params, xi, rough_paths, rng=13, barrier=exotic.barrier)
    result = ex.price_from_stats(exotic, stats, r, T)
    return result.price, result.std_error, result.n_paths, time.perf_counter() - started


def latest(rows: pd.DataFrame) -> dict | None:
    return rows.iloc[-1].to_dict() if len(rows) else None


try:
    heston_row = latest(data.heston_fits("NIFTY"))
    rough_row = latest(data.rough_fits("NIFTY"))
except data.DataUnavailable:
    heston_row = rough_row = None
    st.warning("Market data can't be reached right now, so only Black-Scholes is priced.")

models = [("Black-Scholes", None, f"{sigma:.2%} vol")]
if heston_row:
    models.append(("Heston", heston_row, f"NIFTY fit of {heston_row['trade_date']:%d %b %Y}"))
if rough_row:
    if days <= 365:
        models.append(("Rough Bergomi", rough_row, f"NIFTY fit of {rough_row['trade_date']:%d %b %Y}"))
    else:
        st.info("Rough Bergomi is priced for expiries up to a year.")
if sigma <= 0:
    models = models[1:]

rows = []
with st.spinner("Simulating..."):
    for name, row, source in models:
        price, error, n_paths, seconds = priced(name, exotic, S, T, r, q, sigma, row)
        rows.append(
            {
                "Model": name,
                "Inputs": source,
                "Price": price,
                "± 95%": 1.96 * error,
                "Paths": n_paths,
                "Seconds": seconds,
            }
        )
table = pd.DataFrame(rows)
exact = ex.closed_form_bs(exotic, S, T, r, q, sigma) if sigma > 0 else None
if exact is not None:
    table.insert(
        3, "Black-Scholes formula", [exact if m == "Black-Scholes" else np.nan for m in table["Model"]]
    )
st.dataframe(
    table,
    hide_index=True,
    column_config={
        "Price": st.column_config.NumberColumn(format="%.4f"),
        "± 95%": st.column_config.NumberColumn(format="%.4f"),
        "Black-Scholes formula": st.column_config.NumberColumn(format="%.4f"),
        "Paths": st.column_config.NumberColumn(format="%,d"),
        "Seconds": st.column_config.NumberColumn(format="%.2f"),
    },
)
notes = []
if exact is not None:
    notes.append("The Black-Scholes row's Monte Carlo price should sit within its interval of the formula.")
if len(table) > 1:
    spread = table["Price"].max() - table["Price"].min()
    notes.append(
        f"The models disagree by {spread:,.4f} ({spread / max(table['Price'].mean(), 1e-12):.0%} of the average): "
        "for path-dependent payoffs the smile's dynamics matter, not just today's prices."
    )
st.caption(" ".join(notes))

fig = go.Figure(
    go.Bar(
        x=table["Price"],
        y=table["Model"],
        orientation="h",
        marker_color=[pal.series[charts.MODEL_SLOTS[m]] for m in table["Model"]],
        error_x=dict(type="data", array=table["± 95%"], color=pal.ink_secondary, thickness=1.5, width=6),
        hovertemplate="%{y}: %{x:,.4f}<extra></extra>",
    )
)
if exact is not None:
    fig.add_vline(x=exact, line=dict(color=pal.muted, width=1), annotation_text="Black-Scholes formula")
fig.update_xaxes(title_text="Price", rangemode="tozero")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(theme.style(fig, pal, 220), width="stretch")

# --- Variance reduction ---------------------------------------------------------------------
st.subheader("Getting more accuracy per path")
st.caption(
    "The same arithmetic Asian (the sidebar's strike and expiry, Black-Scholes) priced five ways. Variance "
    "reduction is per path, against plain Monte Carlo: 100× means plain Monte Carlo would need 100 times as "
    "many paths for the same accuracy."
)
if sigma <= 0:
    st.info("Set a volatility above zero in the sidebar.")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=8)
def reduction(S, K, T, r, q, sigma, kind, fixings):
    return pd.DataFrame(ex.variance_reduction(S, K, T, r, q, sigma, fixings, 32_000, kind, seed=21))


vr = reduction(S, K, T, r, q, sigma, kind, min(days, 64))
st.dataframe(
    vr.rename(
        columns={
            "method": "Method",
            "price": "Price",
            "std_error": "Std error",
            "paths": "Paths",
            "seconds": "Seconds",
            "variance_reduction": "Variance reduction",
        }
    ),  # fmt: skip
    hide_index=True,
    column_config={
        "Price": st.column_config.NumberColumn(format="%.5f"),
        "Std error": st.column_config.NumberColumn(format="%.5f"),
        "Paths": st.column_config.NumberColumn(format="%,d"),
        "Seconds": st.column_config.NumberColumn(format="%.2f"),
        "Variance reduction": st.column_config.NumberColumn(format="%,.0f×"),
    },
)
fig = go.Figure(
    go.Bar(
        x=vr["variance_reduction"],
        y=vr["method"],
        orientation="h",
        marker_color=pal.series[0],
        hovertemplate="%{y}: %{x:,.0f}×<extra></extra>",
    )
)
charts.log_ticks(
    fig, "x", [10**k for k in range(int(np.log10(max(vr["variance_reduction"].max(), 10))) + 2)], "×"
)
fig.update_xaxes(title_text="Variance reduction per path (log scale)")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(theme.style(fig, pal, 260), width="stretch")
st.caption(
    "The geometric average has a closed form and moves almost in lockstep with the arithmetic one, which makes it "
    "an excellent control variate. Sobol points fill the space of paths far more evenly than random ones, and "
    "building paths from principal components puts most of the variance in the first few coordinates, where "
    "Sobol points are best."
)

with st.expander("How this is computed"):
    st.markdown(
        f"""
- Prices are fixed once a calendar day until expiry. Asians average those fixings; barriers and lookbacks are monitored
  continuously, with a Brownian-bridge correction between fixings.
- Paths: {BS_PATHS:,} under Black-Scholes and {HESTON_PATHS:,} under Heston (full-truncation Euler), both
  compiled with numba so no path is stored; {rough_paths:,} under rough Bergomi (the hybrid scheme, simulated
  2,000 at a time to keep memory flat).
- The Black-Scholes check uses closed forms: geometric Asians, barriers (Haug) and floating lookbacks
  (Goldman-Sosin-Gatto). There is none for the arithmetic Asian.
- The variance-reduction table uses 32,000 paths per method and up to 64 fixings.
"""
    )
