"""Shared palette and Plotly helpers.

We keep colors in one place so every demo looks like one system. The palette is
chosen to read well on both light and dark themes.
"""
from __future__ import annotations

import plotly.graph_objects as go

# Semantic colors: each role has one stable color across all charts.
COLORS = {
    "actual": "#5B6472",      # actual — calm graphite
    "zero_shot": "#E8963A",   # zero-shot — amber
    "fine_tuned": "#2CA58D",  # fine-tuned — teal-green
    "accent": "#4C7DF0",      # accent / highlight
    "lunch": "#E8963A",       # lunch peak
    "dinner": "#C4472E",      # dinner peak (the larger one)
    "grid": "rgba(128,128,128,0.18)",
}

# Categorical palette for multivariate series (order = priority).
CATEGORICAL = ["#4C7DF0", "#E8963A", "#2CA58D", "#C4472E", "#9B6FD4"]


def base_layout(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    """Bring a chart to the common minimalist style."""
    fig.update_layout(
        height=height,
        # always an explicit dict — a null title can render as "undefined"
        title=dict(text=title or ""),
        template="plotly_white",
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        font=dict(size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], zeroline=False)
    return fig
