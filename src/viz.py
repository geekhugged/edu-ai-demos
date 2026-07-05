"""Единая палитра и хелперы для графиков Plotly.

Держим цвета в одном месте, чтобы все демо выглядели как одна система.
Палитра подобрана так, чтобы читаться и на светлой, и на тёмной теме.
"""
from __future__ import annotations

import plotly.graph_objects as go

# Семантические цвета: у каждой роли — свой устойчивый цвет во всех графиках.
COLORS = {
    "actual": "#5B6472",      # факт — спокойный графитовый
    "zero_shot": "#E8963A",   # zero-shot — янтарный
    "fine_tuned": "#2CA58D",  # fine-tuned — бирюзово-зелёный
    "accent": "#4C7DF0",      # акцент/выделение
    "lunch": "#E8963A",       # обеденный пик
    "dinner": "#C4472E",      # ужинный пик (побольше)
    "grid": "rgba(128,128,128,0.18)",
}

# Категориальная палитра для многомерных рядов (порядок = приоритет).
CATEGORICAL = ["#4C7DF0", "#E8963A", "#2CA58D", "#C4472E", "#9B6FD4"]


def base_layout(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    """Привести график к общему минималистичному стилю."""
    fig.update_layout(
        height=height,
        title=title,
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
