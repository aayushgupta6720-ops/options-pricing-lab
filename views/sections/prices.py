import time

import numpy as np
import pandas as pd
import streamlit as st

from optlab.greeks import binomial_greeks
from optlab.implied_vol import american_implied_vol, implied_vol
from optlab.models import binomial, black_scholes, monte_carlo
from optlab.models.registry import MODELS
from ui import charts, theme

spec = st.session_state["spec"]
pal = theme.current()

st.caption("The same option priced by a formula, a tree and a simulation.")

if spec.T == 0:
    st.info(f"At expiry the option is worth its payoff: {float(spec.payoff(spec.S)):,.4f}.")
    st.stop()

rows = []
bs_price = black_scholes.price_spec(spec.bump(style="european"))
for model in MODELS.values():
    if spec.style == "american" and not model.american:
        rows.append({"Model": model.name, "Price": None, "95% interval": "European only", "Time (ms)": None})
        continue
    start = time.perf_counter()
    if model.name == "Monte Carlo":
        result = monte_carlo.price_spec(spec, n_paths=200_000, seed=7)
        price, interval = result.price, "{:,.4f} – {:,.4f}".format(*result.ci95)
    else:
        price, interval = model.price(spec), ""
    rows.append(
        {
            "Model": model.name,
            "Price": price,
            "95% interval": interval,
            "vs Black-Scholes": price - bs_price if spec.style == "european" else None,
            "Time (ms)": 1000 * (time.perf_counter() - start),
            "Method": model.description,
        }
    )
table = pd.DataFrame(rows)
st.dataframe(
    table,
    hide_index=True,
    column_config={
        "Price": st.column_config.NumberColumn(format="%.4f"),
        "vs Black-Scholes": st.column_config.NumberColumn(format="%+.5f"),
        "Time (ms)": st.column_config.NumberColumn(format="%.1f"),
    },
)

if spec.style == "american":
    european = binomial.price_spec(spec.bump(style="european"), steps=1000)
    american = table.loc[table["Model"] == "Binomial (CRR)", "Price"].iloc[0]
    st.metric(
        "Early-exercise premium",
        f"{american - european:,.4f}",
        help="American minus European value, both from the same 1,000-step tree.",
    )

market = st.session_state.get("market")
unchanged = market and all(st.session_state.get(key) == value for key, value in market["inputs"].items())
if unchanged and spec.style == "european":
    c1, c2 = st.columns(2)
    c1.metric("NSE close", f"₹{market['market_price']:,.2f}")
    c2.metric("Black-Scholes at the market's implied vol", f"₹{bs_price:,.2f}")
    st.caption(
        "They agree (up to rounding of the inputs) because the volatility input is that option's implied vol: "
        "implied vol is the market's price, restated in volatility units."
    )

st.subheader("Greeks")
if spec.style == "european":
    greeks = black_scholes.greeks_spec(spec)
    source = "Analytic Black-Scholes."
else:
    greeks = binomial_greeks(spec, steps=400)
    source = (
        "400-step binomial tree: delta and gamma from its first nodes, vega, theta and rho by "
        "re-pricing with bumped inputs."
    )
g1, g2, g3, g4, g5 = st.columns(5)
g1.metric("Delta", f"{greeks['delta']:.4f}", help="Change in price per 1 unit move in spot.")
g2.metric("Gamma", f"{greeks['gamma']:.6f}", help="Change in delta per 1 unit move in spot.")
g3.metric("Vega", f"{greeks['vega'] / 100:.4f}", help="Change in price per 1 volatility point.")
g4.metric("Theta", f"{greeks['theta'] / 365:.4f}", help="Change in price per calendar day that passes.")
g5.metric("Rho", f"{greeks['rho'] / 100:.4f}", help="Change in price per 1 point move in the rate.")
st.caption(source)

st.subheader("Value against spot")
spread = min(max(4 * spec.sigma * np.sqrt(spec.T), 0.05), 0.6)  # about +/- 4 standard deviations
spots = np.linspace(spec.S * np.exp(-spread), spec.S * np.exp(spread), 121)
if spec.style == "european":
    today = black_scholes.price(spots, spec.K, spec.T, spec.r, spec.sigma, spec.q, spec.kind)
else:
    today = [binomial.price_spec(spec.bump(S=s), steps=200) for s in spots]
fig = charts.lines(
    spots,
    {f"Today ({round(spec.T * 365)} days left)": today, "At expiry": spec.payoff(spots)},
    pal,
    x_title="Spot",
    y_title="Option value",
)
fig.add_vline(x=spec.S, line=dict(color=pal.muted, width=1))
st.plotly_chart(fig, width="stretch")

st.subheader("Implied volatility from a price")
american = spec.style == "american"
model_price = binomial.price_spec(spec, steps=300) if american else bs_price
price_in = st.number_input("Market price", min_value=0.0, value=round(model_price, 4), format="%.4f")
if american:
    iv = american_implied_vol(price_in, spec)
    st.caption("American exercise: solved on a 300-step binomial tree.")
else:
    iv = implied_vol(price_in, spec.S, spec.K, spec.T, spec.r, spec.q, spec.kind)
if np.isnan(iv):
    st.warning("No volatility reproduces that price: it's outside the no-arbitrage bounds for this option.")
else:
    st.metric("Implied volatility", f"{iv:.4%}")
