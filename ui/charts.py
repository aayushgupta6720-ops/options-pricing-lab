"""Plotly figure builders. No Streamlit here, so they can be tested and reused."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ui.theme import Palette, style

LINE = 2
MARKER = 8


def lines(
    x, series: dict, pal: Palette, x_title="", y_title="", percent_y=False, height=360, hover_format=None
) -> go.Figure:
    """One line per entry of `series` (name -> y values), coloured in fixed slot order."""
    fig = go.Figure()
    fmt = hover_format or (".2%" if percent_y else ",.2f")
    for i, (name, y) in enumerate(series.items()):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                mode="lines",
                line=dict(color=pal.series[i], width=LINE),
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
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.9)),
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
