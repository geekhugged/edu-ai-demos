"""Shared palette and Plotly helpers.

We keep colors in one place so every demo looks like one system. The palette is
chosen to read well on both light and dark themes, and charts follow whichever
theme is active (via ``st.context.theme``).
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Semantic colors: each role has one stable color across all charts. The accent
# hues (blue/amber/teal/red/purple) read on both light and dark backgrounds;
# "actual" is a mid slate that stays visible on white and on near-black.
COLORS = {
    "actual": "#7C8698",      # actual — mid slate (readable on both themes)
    "zero_shot": "#E8963A",   # zero-shot — amber
    "fine_tuned": "#2CA58D",  # fine-tuned — teal-green
    "accent": "#4C7DF0",      # accent / highlight
    "lunch": "#E8963A",       # lunch peak
    "dinner": "#C4472E",      # dinner peak (the larger one)
    "grid": "rgba(128,128,128,0.18)",  # subtle on both themes
}

# Categorical palette for multivariate series (order = priority).
CATEGORICAL = ["#4C7DF0", "#E8963A", "#2CA58D", "#C4472E", "#9B6FD4"]

_TEXT = {"dark": "#E6EAF1", "light": "#1B2430"}


def theme_type() -> str:
    """Return the active theme, "dark" or "light".

    Reads ``st.context.theme`` (updates when the viewer switches theme) and
    falls back to the configured ``theme.base`` when the type is unavailable
    (e.g. on first load or outside a script run).
    """
    typ = None
    try:
        theme = st.context.theme
        typ = theme.get("type") if hasattr(theme, "get") else theme["type"]
    except Exception:
        typ = None
    if typ in ("dark", "light"):
        return typ
    try:
        return "dark" if st.get_option("theme.base") == "dark" else "light"
    except Exception:
        return "dark"


def plotly_template() -> str:
    """Plotly template matching the active theme."""
    return "plotly_dark" if theme_type() == "dark" else "plotly_white"


def base_layout(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    """Bring a chart to the common minimalist style, adapting to the theme."""
    dark = theme_type() == "dark"
    fig.update_layout(
        height=height,
        # always an explicit dict — a null title can render as "undefined"
        title=dict(text=title or ""),
        template="plotly_dark" if dark else "plotly_white",
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        font=dict(size=13, color=_TEXT["dark"] if dark else _TEXT["light"]),
        # transparent so charts sit on the Streamlit background of either theme
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], zeroline=False)
    return fig
