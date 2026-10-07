import numpy as np
import pandas as pd
import streamlit as st

from optlab import strategy
from optlab.strategy import Leg
from ui import charts, data, theme

st.title("Strategy builder")
st.caption(
    "Build a multi-leg position at the live price (the last NSE close when there's nothing newer). Each leg is "
    "priced at the market's implied vol for its strike, so premiums match the smile rather than one flat volatility."
)

c1, c2, c3 = st.columns(3)
underlying = c1.selectbox("Underlying", data.UNDERLYINGS, key="strategy_underlying")
as_of, chain = data.load_or_stop(data.latest_chain, underlying)
if as_of is None:
    st.info(f"No quotes for {underlying} yet.")
    st.stop()
# The live price is taken once per visit, so the strikes and legs don't shift under the viewer
# while they edit them; the Refresh button below takes it again.
pinned = st.session_state.setdefault("strategy_live", {})
if underlying not in pinned:
    pinned[underlying] = data.live_spot(underlying, as_of, float(chain["spot"].iloc[0]))
quote = pinned[underlying]
today = quote.time.date() if quote else as_of
expiries = sorted(e for e in chain["expiry"].unique() if e > today)
if not expiries:
    st.info(f"Every {underlying} expiry in the last close has passed; the next close will list new ones.")
    st.stop()
first_fortnight = next((i for i, e in enumerate(expiries) if (e - today).days >= 14), 0)
# Keys include the underlying (and expiry below) so a value chosen for one doesn't carry over to
# another where it makes no sense, e.g. NIFTY's 450-point wings on a ₹1,200 stock.
expiry = c2.selectbox(
    "Expiry",
    expiries,
    index=first_fortnight,
    format_func=lambda e: f"{e:%d %b %Y} ({(e - today).days}d)",
    key=f"strategy_expiry_{underlying}",
)
presets = list(strategy.PRESETS)
preset = c3.selectbox("Strategy", presets, index=presets.index("Iron condor"), key="strategy_preset")

quotes = chain[chain["expiry"] == expiry].sort_values("log_moneyness")
spot, forward, T, r = (float(quotes[c].iloc[0]) for c in ("spot", "forward", "T", "r"))
if quote:
    # Live: the same carry rate over the time that's left now, applied to the live price. Vols are
    # read off the close's smile at each strike's moneyness against this forward (sticky moneyness).
    days_now = (expiry - today).days
    forward = quote.price * strategy.forward_ratio_at(forward / spot, T, days_now / 365)
    spot, T = quote.price, days_now / 365
lot = int(quotes["lot_size"].iloc[0])

price_line = (
    f"{underlying} {spot:,.2f}, live from {quote.source} at {quote.time:%H:%M} IST on {quote.time:%d %b %Y}. "
    f"Vols from the {as_of:%d %b} close's smile."
    if quote
    else f"{underlying} {spot:,.2f}, the close on {as_of:%d %b %Y}."
)
c1, c2 = st.columns([5, 1], vertical_alignment="center")
c1.caption(price_line)
if data.has_live(underlying):
    c2.button(
        "Refresh price",
        key="strategy_refresh",
        on_click=pinned.pop,
        args=(underlying, None),
        help="Take the live price again (NSE, else Yahoo Finance; at most once a minute).",
    )

strikes = np.sort(quotes["strike"].unique())
step = float(np.min(np.diff(strikes))) if len(strikes) > 1 else max(round(spot * 0.01), 1)
atm = float(strikes[np.abs(strikes - forward).argmin()])
default_width = max(step, round(spot * 0.02 / step) * step)
max_width = max(step, np.floor(0.45 * atm / step) * step)  # keeps a condor's outer put strike > 0

width = st.number_input(
    "Wing width (distance between strikes)",
    min_value=step,
    max_value=max_width,
    value=min(default_width, max_width),
    step=step,
    key=f"strategy_width_{underlying}",
)
legs_df = pd.DataFrame(
    [
        {
            "Side": "Buy" if leg.side > 0 else "Sell",
            "Type": leg.kind.title(),
            "Strike": leg.strike,
            "Lots": leg.lots,
        }
        for leg in strategy.preset_legs(preset, atm, width)
    ]
)
edited = st.data_editor(
    legs_df,
    key=f"legs_{underlying}_{expiry}_{preset}_{width}",
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "Side": st.column_config.SelectboxColumn(options=["Buy", "Sell"], required=True),
        "Type": st.column_config.SelectboxColumn(options=["Call", "Put"], required=True),
        "Strike": st.column_config.NumberColumn(min_value=step, step=step, format="%.2f", required=True),
        "Lots": st.column_config.NumberColumn(min_value=1, step=1, required=True),
    },
)
edited = edited.dropna()
if edited.empty:
    st.info("Add at least one leg.")
    st.stop()
legs = [
    Leg(1 if row.Side == "Buy" else -1, row.Type.lower(), float(row.Strike), int(row.Lots))
    for row in edited.itertuples()
]

# Each strike's vol from the smile (flat beyond the quoted range); calls and puts at one strike share it.
log_k = np.log(np.array([leg.strike for leg in legs]) / forward)
vols = np.interp(log_k, quotes["log_moneyness"], quotes["iv"])
ratio = forward / spot
prices = strategy.leg_prices(legs, spot, ratio, T, r, vols)
net = strategy.premium(legs, prices)

pal = theme.current()
lo, hi = min(spot, *(leg.strike for leg in legs)), max(spot, *(leg.strike for leg in legs))
spots = np.linspace(lo - 0.12 * spot, hi + 0.12 * spot, 601)
days_left = max((expiry - today).days, 1)
horizon = st.slider(
    "Days from now for the model curve", 0, days_left - 1, 0, key=f"strategy_horizon_{underlying}_{expiry}"
)
at_expiry = (strategy.payoff(legs, spots) - net) * lot
remaining = T - horizon / 365
later_ratio = strategy.forward_ratio_at(ratio, T, remaining)  # the forward rolls down towards spot
before = (strategy.value(legs, spots, later_ratio, remaining, r, vols) - net) * lot
bes = strategy.breakevens(spots, at_expiry)


def rupees(x: float, decimals: int = 0) -> str:
    return f"{'−' if x < 0 else ''}₹{abs(x):,.{decimals}f}"


best, worst = strategy.extremes(legs, net)
max_profit = "Unlimited" if np.isinf(best) else rupees(best * lot)
max_loss = "Unlimited" if np.isinf(worst) else rupees(worst * lot)
k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Net premium {'paid' if net > 0 else 'received'}", f"₹{abs(net) * lot:,.0f}")
k2.metric("Max profit at expiry", max_profit)
k3.metric("Max loss at expiry", max_loss)
k4.metric("Breakevens", ", ".join(f"{b:,.0f}" for b in bes) or "None")

fig = charts.pnl(spots, at_expiry, before, pal, spot, bes)
fig.data[1].name = f"In {horizon} days (model)" if horizon else "Today (model)"
st.plotly_chart(fig, width="stretch")

g = strategy.greeks(legs, spot, ratio, T, r, vols)
g1, g2, g3, g4 = st.columns(4)
g1.metric(
    "Delta",
    rupees(g["delta"] * lot, 1) + "/pt",
    help="Change in position value per 1-point move in the underlying.",
)
g2.metric("Gamma", f"{g['gamma'] * lot:,.4f}", help="Change in delta (₹/pt) per 1-point move.")
g3.metric(
    "Vega", rupees(g["vega"] * lot / 100), help="Change in value per 1 vol point, all strikes together."
)
g4.metric(
    "Theta", rupees(g["theta"] * lot / 365) + "/day", help="Change in value per calendar day, spot unchanged."
)

st.caption(
    f"Forward {forward:,.2f}; lot size {lot}. P&L is for the whole position in rupees, before brokerage and taxes."
)
with st.expander("Data: legs"):
    st.dataframe(
        pd.DataFrame(
            {
                "Side": edited["Side"].to_numpy(),
                "Type": edited["Type"].to_numpy(),
                "Strike": edited["Strike"].to_numpy(),
                "Lots": edited["Lots"].to_numpy(),
                "Implied vol": vols,
                "Premium per unit (₹)": prices,
            }
        ),
        hide_index=True,
        column_config={"Implied vol": st.column_config.NumberColumn(format="percent")},
    )
