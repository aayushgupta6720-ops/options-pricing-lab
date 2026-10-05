import numpy as np
import plotly.graph_objects as go
import streamlit as st

from optlab.implied_vol import implied_vol
from optlab.models import black_scholes, heston
from optlab.models.heston import HestonParams
from ui import charts, data, theme

spec = st.session_state["spec"]
pal = theme.current()

st.title("Heston model")
st.caption(
    "Black-Scholes holds volatility fixed. Heston lets the variance wander, mean-revert and move against the "
    "market, which is what bends a flat line into a skewed smile. Move the parameters and watch the smile "
    "change; the option comes from the sidebar."
)

TEXTBOOK = HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)  # Fang & Oosterlee


def latest_nifty() -> HestonParams | None:
    try:
        fits = data.heston_fits("NIFTY")
    except data.DataUnavailable:
        return None
    return data.heston_params(fits.iloc[-1]) if len(fits) else None


def load(p: HestonParams):
    st.session_state.update(
        {
            "hs_vol0": round(100 * np.sqrt(p.v0), 1),
            "hs_volbar": round(100 * np.sqrt(p.theta), 1),
            "hs_kappa": round(float(np.clip(p.kappa, 0.1, 30.0)), 2),
            "hs_xi": round(float(np.clip(p.xi, 0.05, 5.0)), 2),
            "hs_rho": round(float(p.rho), 2),
        }
    )


if "hs_vol0" not in st.session_state:
    load(latest_nifty() or TEXTBOOK)

b1, b2, _ = st.columns([1, 1, 3])
nifty_fit = latest_nifty()
b1.button("Latest NIFTY fit", key="hs_load_fit", on_click=load, args=(nifty_fit,), disabled=nifty_fit is None)
b2.button(
    "Textbook example",
    key="hs_load_textbook",
    on_click=load,
    args=(TEXTBOOK,),
    help="Fang & Oosterlee (2008)'s test parameters.",
)

c = st.columns(5)
c[0].slider("Spot vol √v₀ (%)", 1.0, 80.0, step=0.5, key="hs_vol0")
c[1].slider("Long-run vol √θ (%)", 1.0, 80.0, step=0.5, key="hs_volbar")
c[2].slider("Mean reversion κ", 0.1, 30.0, step=0.1, key="hs_kappa")
c[3].slider("Vol of vol ξ", 0.05, 5.0, step=0.05, key="hs_xi")
c[4].slider("Correlation ρ", -0.99, 0.99, step=0.01, key="hs_rho")
ss = st.session_state
p = HestonParams((ss.hs_vol0 / 100) ** 2, ss.hs_kappa, (ss.hs_volbar / 100) ** 2, ss.hs_xi, ss.hs_rho)
st.caption(
    f"Feller ratio 2κθ/ξ² = {p.feller_ratio:.2f} ({'variance stays positive' if p.feller_ratio >= 1 else 'variance can touch zero'}); "
    f"half-life of a vol shock {np.log(2) / p.kappa * 365:.0f} days."
)

if spec.T == 0:
    st.info("Pick an option with time to expiry in the sidebar.")
    st.stop()
if spec.style == "american":
    st.info(
        "Heston is priced here for European exercise; the sidebar's American setting is ignored on this page."
    )

S, K, T, r, q = spec.S, spec.K, spec.T, spec.r, spec.q
F = S * np.exp((r - q) * T)
price = float(heston.price(F, K, T, r, p, spec.kind))
iv = float(implied_vol(price, F, K, T, r, r, spec.kind))
bs = float(black_scholes.price(S, K, T, r, spec.sigma, q, spec.kind)) if spec.sigma > 0 else np.nan

m = st.columns(3)
m[0].metric(f"Heston {spec.kind}", f"{price:,.4f}")
m[1].metric(
    "Its implied vol",
    "—" if np.isnan(iv) else f"{iv:.2%}",
    help="The Black-Scholes vol that gives the same price.",
)
m[2].metric(f"Black-Scholes at σ = {spec.sigma:.2%}", "—" if np.isnan(bs) else f"{bs:,.4f}")

c1, c2 = st.columns(2)
with c1:
    st.subheader("The smile Heston makes")
    width = max(np.sqrt(p.v0 * T), 0.02) * 3
    strikes = F * np.exp(np.linspace(-width, width, 81))
    kinds = np.where(strikes < F, "put", "call")
    vols = implied_vol(heston.price(F, strikes, T, r, p, kinds), F, strikes, T, r, r, kinds)
    series = {f"Heston, {round(T * 365)} days": vols}
    if spec.sigma > 0:
        series["Black-Scholes (flat)"] = np.full_like(strikes, spec.sigma)
    fig = charts.lines(strikes, series, pal, x_title="Strike", y_title="Implied volatility", percent_y=True)
    fig.add_vline(x=K, line=dict(color=pal.muted, width=1), annotation_text="Your strike")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "ρ tilts it (negative: downside strikes dearer); ξ curves it; both matter most for short expiries."
    )
with c2:
    st.subheader("At-the-money vol by maturity")
    maturities = np.geomspace(7 / 365, 3.0, 40)
    atm = [float(implied_vol(heston.price(1.0, 1.0, t, 0.0, p), 1.0, 1.0, t, 0.0, 0.0)) for t in maturities]
    fig = charts.lines(
        maturities * 365,
        {"ATM implied vol": atm},
        pal,
        x_title="Days to expiry",
        y_title="Implied volatility",
        percent_y=True,
    )
    fig.update_xaxes(type="log")
    charts.reference_line(fig, np.sqrt(p.theta), "√θ", pal)
    st.plotly_chart(fig, width="stretch")
    st.caption("Short-dated vol starts near √v₀ and drifts towards √θ, faster for a larger κ.")

st.subheader("Check: Monte Carlo against the formula")
paths = st.select_slider(
    "Paths", [5_000, 20_000, 50_000, 100_000], value=20_000, format_func="{:,}".format, key="hs_paths"
)
terminal = heston.simulate(S, T, r, q, p, n_paths=paths, n_steps=100, rng=7)
payoff = spec.payoff(terminal) * np.exp(-r * T)
estimate, error = payoff.mean(), payoff.std(ddof=1) / np.sqrt(paths)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[estimate],
        y=["Monte Carlo"],
        error_x=dict(type="constant", value=1.96 * error, color=pal.series[0], thickness=2, width=8),
        mode="markers",
        marker=dict(size=charts.MARKER + 2, color=pal.series[0]),
        name="Monte Carlo, 95% interval",
        hovertemplate=f"%{{x:,.4f}} ± {1.96 * error:,.4f}<extra></extra>",
    )
)
fig.add_vline(x=price, line=dict(color=pal.muted, width=1), annotation_text="Formula")
fig.update_xaxes(title_text="Option value")
st.plotly_chart(theme.style(fig, pal, 180), width="stretch")
inside = abs(estimate - price) <= 1.96 * error
st.caption(
    f"{paths:,} paths, 100 time steps (full-truncation Euler): {estimate:,.4f} ± {1.96 * error:,.4f}, "
    f"{'containing' if inside else 'missing'} the formula's {price:,.4f}. The formula integrates Heston's "
    "characteristic function; the simulation is an independent check on it."
)
