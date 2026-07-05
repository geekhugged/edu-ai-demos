"""Transformer & attention mechanism — a general, model-agnostic explainer."""
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import general_theory as gt
from src.attention import softmax
from src.viz import COLORS, base_layout

st.title("🧠 Transformer & attention mechanism")
st.markdown(
    "A short, model-agnostic tour of how Transformers work and what the "
    "**attention** mechanism actually computes — the ideas behind the Chronos "
    "demo (and behind LLMs like ChatGPT). Click through the sections."
)

tabs = st.tabs([
    "🧠 Transformer",
    "🎯 Attention",
    "🧩 Multi-head & positions",
    "🏗️ Encoder / decoder",
    "👥 Group attention",
])

with tabs[0]:
    st.markdown(gt.TRANSFORMER)

with tabs[1]:
    st.markdown(gt.ATTENTION)
    st.divider()
    st.markdown("#### 🕹️ Feel the softmax")
    st.markdown(
        "Below are the raw **scores** ($QK^\\top/\\sqrt{d_k}$) of a query against "
        "five keys. Move the **temperature** to see how softmax turns them into "
        "attention weights — low temperature = sharp focus on the best key, high "
        "temperature = spread out."
    )
    keys = ["key 1", "key 2", "key 3", "key 4", "key 5"]
    base_scores = np.array([2.0, 1.2, 0.6, 0.0, -0.6])
    c1, c2 = st.columns([1, 2])
    with c1:
        temp = st.slider("Temperature", 0.2, 3.0, 1.0, 0.1)
        boost = st.slider("Boost key 3's score", -2.0, 4.0, 0.6, 0.2,
                          help="Raise/lower one key's similarity to the query.")
        st.caption("Raw scores come from the query·key dot products; softmax makes "
                   "them a distribution that sums to 1.")
    scores = base_scores.copy()
    scores[2] = boost
    weights = softmax(scores, temperature=temp)
    with c2:
        fig = go.Figure(
            go.Bar(x=keys, y=weights,
                   marker=dict(color=weights, colorscale="Blues", cmin=0,
                               line=dict(width=0)),
                   customdata=scores,
                   hovertemplate="%{x}<br>score: %{customdata:.2f}"
                                 "<br>weight: %{y:.2f}<extra></extra>")
        )
        fig.update_yaxes(title="Attention weight (sums to 1)", range=[0, 1])
        base_layout(fig, height=320, title="softmax(scores / temperature)")
        st.plotly_chart(fig, width="stretch")
    top = int(np.argmax(weights))
    st.success(
        f"🔦 Most of the attention (**{weights[top]*100:.0f}%**) goes to "
        f"**{keys[top]}** — the highest-scoring key. Raise the temperature and the "
        f"weights flatten toward equal (1/5 each); lower it and they collapse onto "
        f"the single best key."
    )

with tabs[2]:
    st.markdown(gt.MULTIHEAD)

with tabs[3]:
    st.markdown(gt.ENCODER_DECODER)

with tabs[4]:
    st.markdown(gt.GROUP_ATTENTION)
    st.info(
        "See it in action: the Chronos v2 demo's **Multivariate series** tab "
        "simulates covariates (promo, rain, temperature, holidays) feeding the "
        "target — the job group attention does inside the model."
    )
