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
    "demo (and behind LLMs like ChatGPT). **Start at 🌱 The big idea** for the "
    "beginner version, then dig deeper in the later tabs."
)


def _rnn_diagram(words: list[str]) -> go.Figure:
    """RNN as a left→right chain: info hops from word to word."""
    n = len(words)
    xs = list(range(n))
    fig = go.Figure()
    for i in range(n - 1):  # consecutive arrows
        fig.add_annotation(x=xs[i + 1], y=0, ax=xs[i], ay=0, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=3,
                           arrowsize=1.1, arrowwidth=2, arrowcolor=COLORS["zero_shot"])
    fig.add_trace(go.Scatter(
        x=xs, y=[0] * n, mode="markers+text", text=words, textposition="top center",
        textfont=dict(size=13), marker=dict(size=24, color=COLORS["accent"]),
        hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, range=[-0.6, n - 0.4])
    fig.update_yaxes(visible=False, range=[-0.9, 1.0])
    base_layout(fig, height=240, title="🔁 RNN — whisper down the line")
    fig.add_annotation(x=(n - 1) / 2, y=-0.7, showarrow=False,
                       text=f"word 1 → word {n}: {n - 1} hops (signal fades)",
                       font=dict(color=COLORS["zero_shot"], size=13))
    return fig


def _attention_diagram(words: list[str]) -> go.Figure:
    """Attention: the last word reaches every other word in one direct hop (arcs)."""
    n = len(words)
    xs = list(range(n))
    last = n - 1
    fig = go.Figure()
    for i in range(n - 1):  # arc from each word up to the last word
        t = np.linspace(0, 1, 24)
        ax = xs[i] + t * (xs[last] - xs[i])
        ay = 0.7 * 4 * t * (1 - t)
        fig.add_trace(go.Scatter(x=ax, y=ay, mode="lines", hoverinfo="skip",
                                 line=dict(color=COLORS["fine_tuned"], width=1.6),
                                 showlegend=False))
    fig.add_trace(go.Scatter(
        x=xs, y=[0] * n, mode="markers+text", text=words, textposition="bottom center",
        textfont=dict(size=13), marker=dict(size=24, color=COLORS["accent"]),
        hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, range=[-0.6, n - 0.4])
    fig.update_yaxes(visible=False, range=[-0.95, 1.0])
    base_layout(fig, height=240, title="⚡ Attention — look at all at once")
    fig.add_annotation(x=(n - 1) / 2, y=-0.8, showarrow=False,
                       text="last word ← every word: 1 hop each",
                       font=dict(color=COLORS["fine_tuned"], size=13))
    return fig

tabs = st.tabs([
    "🌱 The big idea",
    "🧠 Transformer",
    "🎯 Attention",
    "🍰 A tiny example",
    "🧩 Multi-head & positions",
    "🏗️ Encoder / decoder",
    "👥 Group attention",
])

with tabs[0]:
    st.markdown(gt.BIG_IDEA)
    st.divider()
    st.markdown("#### 🎲 Try it: predict the next word")
    st.markdown(
        "Pick the start of a sentence. The model gives a **probability to each "
        "possible next word** and would pick a likely one — then repeat to write "
        "the whole sentence."
    )
    ctx = st.selectbox("Beginning of a sentence…", list(gt.NEXT_WORD_EXAMPLES.keys()))
    dist = gt.NEXT_WORD_EXAMPLES[ctx]
    words = list(dist.keys())
    probs = np.array(list(dist.values()), dtype=float)
    n1, n2 = st.columns([1, 2])
    with n1:
        creativity = st.slider("Creativity (temperature)", 0.2, 2.0, 1.0, 0.1,
                               key="nw_temp",
                               help="Low = always the safest word; high = more "
                                    "surprising, varied words.")
        st.caption("Real models choose from ~50,000 words, not five — but the game "
                   "is exactly this.")
    # temperature reshaping: softmax(log p / T)
    scaled = softmax(np.log(probs), temperature=creativity)
    with n2:
        fig_nw = go.Figure(go.Bar(
            x=words, y=scaled,
            marker=dict(color=scaled, colorscale="Blues", cmin=0, line=dict(width=0)),
        ))
        fig_nw.update_yaxes(title="probability", range=[0, 1])
        base_layout(fig_nw, height=300, title=f"“{ctx} ___”  →  next word")
        st.plotly_chart(fig_nw, width="stretch")
    best = words[int(np.argmax(scaled))]
    st.success(
        f"🔮 Most likely next word: **{best}**  →  “{ctx} **{best}**”. A model adds "
        f"it, then predicts *again* — that's how it writes whole sentences. **How** "
        f"it decides which earlier words matter is **attention** → next tabs."
    )
    with st.expander("🌡️ What is “creativity” (temperature)?"):
        st.markdown(gt.CREATIVITY_NOTE)

with tabs[1]:
    st.markdown(gt.TRANSFORMER)
    st.divider()
    st.markdown(gt.RNN_VS_ATTENTION)
    _words = ["The", "cat", "I", "saw", "was", "black"]
    nwords = st.slider("Sentence length", 4, 6, 6, key="rnn_n",
                       help="Grow the sentence and watch the RNN's path get longer "
                            "while attention stays at one hop.")
    ws = _words[:nwords]
    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(_rnn_diagram(ws), width="stretch")
    with d2:
        st.plotly_chart(_attention_diagram(ws), width="stretch")
    st.caption(
        "As the sentence grows, the RNN's path from the first word to the last "
        "gets **longer** (and its memory weaker), while attention stays at a "
        "**single hop** — that's the whole reason Transformers won."
    )

with tabs[2]:
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

with tabs[3]:
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

with tabs[4]:
    st.markdown(gt.MULTIHEAD)

with tabs[5]:
    st.markdown(gt.ENCODER_DECODER)

with tabs[6]:
    st.markdown(gt.GROUP_ATTENTION)
    st.info(
        "See it in action: the Chronos v2 demo's **Multivariate series** tab "
        "simulates covariates (promo, rain, temperature, holidays) feeding the "
        "target — the job group attention does inside the model."
    )
