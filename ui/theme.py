"""Chart colours and the shared Plotly styling.

The categorical palette is the validated eight-slot reference palette (adjacent-pair CVD
separation >= 8.4 in both modes), assigned in fixed order and never cycled. Series past the first
four only appear where they're also direct-labelled. Light-mode slots 3 and 4 are under 3:1
contrast on the surface, so every chart also has a table view.
"""

from dataclasses import dataclass

import plotly.graph_objects as go

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


@dataclass(frozen=True)
class Palette:
    series: tuple[str, ...]
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    sequential: tuple[str, ...]


LIGHT = Palette(
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)
DARK = Palette(
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    # Dark mode runs the same ramp the other way, so "more" is still "more contrast".
    sequential=("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
)


def current() -> Palette:
    """The palette for the viewer's Streamlit theme."""
    import streamlit as st

    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except AttributeError:
        return LIGHT


def style(fig: go.Figure, pal: Palette, height: int = 380, percent_y: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        font=dict(family=FONT, color=pal.ink_secondary, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=36, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(font_family=FONT),
    )
    axis = dict(
        gridcolor=pal.grid,
        linecolor=pal.axis,
        zerolinecolor=pal.axis,
        tickfont_color=pal.muted,
        showline=True,
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if percent_y:
        fig.update_yaxes(tickformat=".1~%")
    return fig
