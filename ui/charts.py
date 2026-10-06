"""Plotly figure builders. No Streamlit here, so they can be tested and reused."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ui.theme import Palette, style

LINE = 2
MARKER = 8
# Each model keeps one colour on every page. Only the first three palette slots are used, since
# they're the ones that stay distinguishable in any combination (slot 4, yellow, is too close to
# orange). Market data and Black-Scholes never share a chart, nor do SABR and rough Bergomi.
MODEL_SLOTS = {"Market": 0, "Black-Scholes": 0, "SABR": 1, "Rough Bergomi": 1, "Heston": 2}


def log_ticks(fig: go.Figure, axis: str, values, suffix=""):
    """Plain labels on a log axis, instead of Plotly's 2-5-10 minor labels."""
    values = list(values)
    labels = [f"{v:,.0f}{suffix}" if v >= 1 else f"{v:g}{suffix}" for v in values]
    getattr(fig, f"update_{axis}axes")(type="log", tickvals=values, ticktext=labels)


def lines(
    x,
    series: dict,
    pal: Palette,
    x_title="",
    y_title="",
    percent_y=False,
    height=360,
    hover_format=None,
    slots: list[int] | None = None,
) -> go.Figure:
    """One line per entry of `series` (name -> y values), coloured in fixed slot order, or by
    `slots` when the lines are entities with a colour of their own (a model is always the same
    colour; see MODEL_SLOTS)."""
    fig = go.Figure()
    fmt = hover_format or (".2%" if percent_y else ",.2f")
    for i, (name, y) in enumerate(series.items()):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                mode="lines",
                line=dict(color=pal.series[slots[i] if slots else i], width=LINE),
                hovertemplate=f"%{{y:{fmt}}}<extra>{name}</extra>",
            )
        )
    fig.update_layout(hovermode="x unified", showlegend=len(series) > 1)
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    return style(fig, pal, height, percent_y)


def reference_line(fig: go.Figure, y: float, label: str, pal: Palette) -> go.Figure:
    """A thin muted horizontal reference (e.g. the exact Black-Scholes price), labelled in place."""
    fig.add_hline(
        y=y,
        line=dict(color=pal.muted, width=1),
        annotation_text=label,
        annotation_position="top left",
        annotation_font_color=pal.ink_secondary,
    )
    return fig


def smile(chain: pd.DataFrame, expiries: list, pal: Palette, x: str = "strike") -> go.Figure:
    """IV against strike (or log-moneyness), one colour per expiry; circles are puts, diamonds calls."""
    fig = go.Figure()
    for i, expiry in enumerate(expiries):
        g = chain[chain["expiry"] == expiry].sort_values("strike")
        days = int(round(g["T"].iloc[0] * 365))
        name = f"{expiry:%d %b %Y} ({days}d)"
        fig.add_trace(
            go.Scatter(
                x=g[x],
                y=g["iv"],
                name=name,
                mode="lines+markers",
                line=dict(color=pal.series[i], width=LINE),
                marker=dict(
                    size=MARKER,
                    symbol=np.where(g["option_type"] == "put", "circle", "diamond"),
                    line=dict(width=0),
                ),
                customdata=np.stack([g["option_type"], g["close"], g["n_trades"], g["strike"]], axis=1),
                hovertemplate=(
                    "Strike %{customdata[3]:,.0f} %{customdata[0]}<br>IV %{y:.2%}<br>"
                    "Close ₹%{customdata[1]:,.2f} · %{customdata[2]:,} trades<extra>" + name + "</extra>"
                ),
            )
        )
    title = "Strike" if x == "strike" else "Log-moneyness ln(K / F)"
    fig.update_xaxes(title_text=title, tickformat=",.0f" if x == "strike" else ".0%")
    fig.update_yaxes(title_text="Implied volatility")
    return style(fig, pal, 420, percent_y=True)


def term_structure(metrics: pd.DataFrame, pal: Palette) -> go.Figure:
    days = (metrics["T"] * 365).round()
    fig = lines(
        days,
        {
            "At the money": metrics["atm_iv"],
            "25-delta put": metrics["put_25d_iv"],
            "25-delta call": metrics["call_25d_iv"],
        },
        pal,
        x_title="Days to expiry",
        y_title="Implied volatility",
        percent_y=True,
    )
    fig.update_traces(mode="lines+markers", marker=dict(size=MARKER))
    return fig


def surface(grid: pd.DataFrame, days: np.ndarray, pal: Palette) -> go.Figure:
    """grid: rows = expiries, columns = standardised moneyness (standard deviations), values = IV."""
    k = np.asarray(grid.columns, dtype=float)
    scale = [[i / (len(pal.sequential) - 1), c] for i, c in enumerate(pal.sequential)]
    fig = go.Figure(
        go.Surface(
            x=k,
            y=days,
            z=grid.to_numpy(dtype=float),
            colorscale=scale,
            colorbar=dict(title=dict(text="IV"), tickformat=".0%", thickness=12),
            hovertemplate="%{x:+.1f} SD from the forward<br>%{y:.0f} days<br>IV %{z:.2%}<extra></extra>",
            contours=dict(z=dict(show=False)),
        )
    )
    axis = dict(gridcolor=pal.grid, backgroundcolor="rgba(0,0,0,0)", color=pal.muted)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Moneyness (SD)", **axis),
            yaxis=dict(title="Days to expiry", **axis),
            zaxis=dict(title="IV", tickformat=".0%", **axis),
            camera=dict(eye=dict(x=-1.75, y=-1.75, z=1.0)),
        )
    )
    return style(fig, pal, 520)


def pnl(spots, at_expiry, today, pal: Palette, spot_now: float, breakevens=()) -> go.Figure:
    fig = lines(
        spots,
        {"At expiry": at_expiry, "Today (model value)": today},
        pal,
        x_title="Underlying price",
        y_title="Profit / loss (₹)",
        hover_format=",.0f",
    )
    fig.add_hline(y=0, line=dict(color=pal.axis, width=1))
    fig.add_vline(
        x=spot_now,
        line=dict(color=pal.muted, width=1),
        annotation_text=f"Spot {spot_now:,.0f}",
        annotation_font_color=pal.ink_secondary,
    )
    for b in breakevens:
        fig.add_annotation(
            x=b,
            y=0,
            text=f"{b:,.0f}",
            showarrow=True,
            arrowcolor=pal.muted,
            ay=-28,
            font_color=pal.ink_secondary,
        )
    fig.update_xaxes(tickformat=",.0f")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def model_smile(quotes: pd.DataFrame, curves: dict, pal: Palette, fitted=None) -> go.Figure:
    """One expiry: market quotes as dots, each model's smile (name -> (strikes, vols)) as a line.

    `fitted` marks the quotes the models were fitted to; the others are drawn hollow.
    """
    fitted = np.ones(len(quotes), dtype=bool) if fitted is None else np.asarray(fitted)
    fig = go.Figure()
    for i, (name, (strikes, vols)) in enumerate(curves.items(), start=1):
        fig.add_trace(
            go.Scatter(
                x=strikes,
                y=vols,
                name=name,
                mode="lines",
                line=dict(color=pal.series[MODEL_SLOTS.get(name, i)], width=LINE),
                hovertemplate=f"Strike %{{x:,.0f}}<br>IV %{{y:.2%}}<extra>{name}</extra>",
            )
        )
    for name, mask, marker in (
        ("Market", fitted, dict(color=pal.series[0], line=dict(width=0))),
        (
            "Market, far wings (not fitted)",
            ~fitted,
            dict(color="rgba(0,0,0,0)", line=dict(width=1.5, color=pal.series[0])),
        ),
    ):
        q = quotes[mask]
        if q.empty:
            continue
        detailed = {"close", "n_trades"} <= set(q.columns)
        fig.add_trace(
            go.Scatter(
                x=q["strike"],
                y=q["iv"],
                name=name,
                mode="markers",
                marker=dict(size=MARKER, **marker),
                customdata=np.stack(
                    [q["option_type"], q["close"], q["n_trades"]] if detailed else [q["option_type"]], axis=1
                ),
                hovertemplate=(
                    "Strike %{x:,.0f} %{customdata[0]}<br>IV %{y:.2%}"
                    + ("<br>Close ₹%{customdata[1]:,.2f} · %{customdata[2]:,} trades" if detailed else "")
                    + f"<extra>{name}</extra>"
                ),
            )
        )
    fig.update_xaxes(title_text="Strike", tickformat=",.0f")
    fig.update_yaxes(title_text="Implied volatility")
    return style(fig, pal, 400, percent_y=True)


def residual_heatmap(
    table: pd.DataFrame, counts: pd.DataFrame, pal: Palette, limit: float = 0.03
) -> go.Figure:
    """Rows are expiries, columns moneyness buckets, cells the mean model-minus-market vol."""
    n = len(pal.diverging) - 1
    scale = [[i / n, c] for i, c in enumerate(pal.diverging)]
    fig = go.Figure(
        go.Heatmap(
            z=table.to_numpy(dtype=float),
            x=list(table.columns),
            y=list(table.index),
            customdata=counts.to_numpy(),
            colorscale=scale,
            zmin=-limit,
            zmax=limit,
            zmid=0,
            xgap=2,
            ygap=2,
            colorbar=dict(title=dict(text="Model − market"), tickformat="+.1%", thickness=12),
            hovertemplate="%{y}<br>%{x}<br>Model − market: %{z:+.2%}<br>%{customdata} quotes<extra></extra>",
        )
    )
    # Category axis: the bucket labels use a typographic minus, which a numeric axis would drop.
    fig.update_xaxes(
        title_text="Moneyness (standard deviations from the forward)", showgrid=False, type="category"
    )
    fig.update_yaxes(title_text="Expiry", autorange="reversed", showgrid=False, type="category")
    return style(fig, pal, max(260, 44 * len(table) + 120))
