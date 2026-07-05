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
from src.attention import (
    EAT_CAKE_SENTENCE,
    EAT_SENTENCE,
    next_word_attention,
    softmax,
    word_self_attention,
)
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
    "🍰 A tiny example",
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
    st.markdown("#### 🍰 The simplest example: two sentences")
    st.markdown(
        "Attention is easiest to *feel* on words. Compare:\n\n"
        "> **“I want to eat”**  → the sentence isn't finished — what word comes next?\n\n"
        "> **“I want to eat cake”**  → now it's complete.\n\n"
        "Each word carries a tiny meaning vector (on the axes *subject/desire · "
        "action · food*), and attention is just the softmax of how well a **query** "
        "matches each word's **key**."
    )

    st.divider()
    st.markdown("##### 1️⃣ Predicting the next word — “I want to eat ___”")
    st.markdown(
        "To guess the blank, the model's *next-word* query looks back over the "
        "sentence. Watch **where it looks**:"
    )
    e1, e2 = st.columns([1, 2])
    with e1:
        t1 = st.slider("Attention sharpness", 0.2, 1.5, 0.5, 0.05, key="eat_t1")
        st.caption("The query is 'what object am I about to eat?' — it should land "
                   "on the verb **eat**.")
    w_next = next_word_attention(EAT_SENTENCE, temperature=t1)
    with e2:
        fig1 = go.Figure(go.Bar(
            x=EAT_SENTENCE, y=w_next,
            marker=dict(color=w_next, colorscale="Blues", cmin=0, line=dict(width=0)),
        ))
        fig1.update_yaxes(title="Attention weight", range=[0, 1])
        base_layout(fig1, height=300, title="next-word query → “I want to eat …”")
        st.plotly_chart(fig1, width="stretch")
    top1 = EAT_SENTENCE[int(np.argmax(w_next))]
    st.success(
        f"🔦 The query attends most to **{top1}** — the verb. So the model expects "
        f"the next word to be **something you eat**: *cake*, pizza, sushi… That's "
        f"attention picking the relevant context to make a prediction."
    )

    st.divider()
    st.markdown("##### 2️⃣ The finished sentence — “I want to eat cake”")
    st.markdown(
        "Now every word can look at the others (**self-attention**). Pick a word "
        "and see which *other* words it links to (its own position is hidden):"
    )
    s1, s2 = st.columns([1, 2])
    with s1:
        focus = st.select_slider(
            "Selected word", options=list(range(len(EAT_CAKE_SENTENCE))),
            value=EAT_CAKE_SENTENCE.index("cake"),
            format_func=lambda i: EAT_CAKE_SENTENCE[i], key="eat_focus",
        )
        t2 = st.slider("Attention sharpness", 0.2, 1.5, 0.5, 0.05, key="eat_t2")
    w_self = word_self_attention(EAT_CAKE_SENTENCE, focus, temperature=t2)
    with s2:
        fig2 = go.Figure(go.Bar(
            x=EAT_CAKE_SENTENCE, y=w_self,
            marker=dict(color=w_self, colorscale="Teal", cmin=0, line=dict(width=0)),
        ))
        fig2.update_yaxes(title="Attention weight", range=[0, 1])
        base_layout(fig2, height=300,
                    title=f"“{EAT_CAKE_SENTENCE[focus]}” looks at…")
        st.plotly_chart(fig2, width="stretch")
    top2 = EAT_CAKE_SENTENCE[int(np.argmax(w_self))]
    st.info(
        f"🔗 **{EAT_CAKE_SENTENCE[focus]}** attends most to **{top2}**. "
        "Try **cake** → it points back at **eat** (the verb it's the object of); "
        "try **eat** → it points at **cake** and **want**. Attention wires the "
        "related words together — no rules, just learned meaning vectors."
    )

with tabs[3]:
    st.markdown(gt.MULTIHEAD)

with tabs[4]:
    st.markdown(gt.ENCODER_DECODER)

with tabs[5]:
    st.markdown(gt.GROUP_ATTENTION)
    st.info(
        "See it in action: the Chronos v2 demo's **Multivariate series** tab "
        "simulates covariates (promo, rain, temperature, holidays) feeding the "
        "target — the job group attention does inside the model."
    )
