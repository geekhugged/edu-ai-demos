"""General (model-agnostic) explanations of Transformers and attention.

Reused by the standalone "Transformer & Attention" page. Kept separate from the
Chronos-specific theory so it can back any demo.
"""

BIG_IDEA = r"""
### 🌱 The big idea — a Transformer is a "next-word machine"

You already use one every day: **phone autocomplete** and **ChatGPT** are
Transformers. Strip away the jargon and the whole job is astonishingly simple:

> **Read the words so far → guess the *next* word → repeat.**

That's it. To write a sentence, the model predicts one word, adds it, and predicts
again — over and over. "Understanding," "answering," "translating" all fall out of
doing this next-word game extremely well, on a model trained on a huge slice of
the internet.

**Two questions make the whole thing work:**

1. **Which word comes next?** The model outputs a **probability for every possible
   word** and picks a likely one. Try the little predictor below 👇
2. **Which earlier words should it look at to decide?** That's the **attention**
   mechanism — the star of this page. Intuition: to finish *"I want to eat ___"*
   the model **pays attention to "eat"** and expects a food. It highlights the
   *relevant* words and ignores the rest.

Everything else — layers, multi-head, positions, encoders/decoders — is machinery
that makes these two steps work well at scale. The other tabs unpack each piece;
this one is the 30-second version.
"""

# Beginner "predict the next word" demo: context → {next word: rough probability}.
NEXT_WORD_EXAMPLES: dict[str, dict[str, float]] = {
    "The weather today is": {
        "sunny": 0.34, "cold": 0.22, "rainy": 0.20, "nice": 0.14, "hot": 0.10,
    },
    "I want to eat": {
        "pizza": 0.30, "cake": 0.25, "sushi": 0.20, "lunch": 0.15, "something": 0.10,
    },
    "Machine learning is": {
        "powerful": 0.30, "hard": 0.22, "fun": 0.20, "everywhere": 0.16, "math": 0.12,
    },
    "Once upon a": {
        "time": 0.86, "dream": 0.06, "day": 0.05, "star": 0.03,
    },
    "The cat sat on the": {
        "mat": 0.40, "sofa": 0.24, "floor": 0.20, "roof": 0.10, "table": 0.06,
    },
}


TRANSFORMER = r"""
### 🧠 What is a Transformer?

A **Transformer** is a neural network made of a stack of identical layers, each
with two parts:

1. an **attention** sub-layer — lets every position in the sequence look at every
   other position and pull in what's relevant;
2. a **feed-forward (MLP)** sub-layer — transforms each position on its own.

Around both sit **residual connections** and **layer norm** for stable training.

**Why it replaced RNNs.** An RNN reads a sequence step by step, so information
from far back has to survive many hops (it fades). A Transformer lets any position
attend **directly** to any other in a single step — so "the same hour yesterday"
or "the promo three weeks ago" is one hop away, not thirty. It also processes the
whole sequence **in parallel**, which makes it fast to train on huge datasets —
exactly what makes *foundation models* possible.

> For time series this is the whole point: the model can reach straight for the
> relevant past — the matching hour, weekday, month — instead of letting it decay.
"""

ATTENTION = r"""
### 🎯 The attention mechanism

Attention answers one question for each position: **"which other positions should
I pull information from, and how much?"**

Every token produces three vectors:

- **Query (Q)** — *what am I looking for?*
- **Key (K)** — *what do I offer?*
- **Value (V)** — *what I'll actually hand over if attended to.*

The recipe:

$$\mathrm{Attention}(Q,K,V)=\underbrace{\mathrm{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}}\right)}_{\text{weights that sum to 1}}\,V$$

1. **Score** every query against every key with a dot product $QK^\top$ — a big
   value means "similar / relevant."
2. **Scale** by $\sqrt{d_k}$ so the scores don't explode as the dimension grows.
3. **Softmax** each row → weights between 0 and 1 that sum to 1.
4. **Weighted sum** of the values → the output for that position.

That's it — a *soft, learnable lookup*. The interactive demo below lets you feel
how the softmax turns raw similarities into a distribution, and how the
**temperature** sharpens or softens the focus.
"""

MULTIHEAD = r"""
### 🧩 Multi-head attention & positions

**Multiple heads.** One attention operation can only track one kind of
relationship. So Transformers run several in parallel — **heads** — each looking
in a different learned subspace (one head might track "same hour", another "same
weekday"), then concatenate the results. This is **multi-head attention**.

**Positions.** Attention is *order-agnostic*: shuffle the inputs and the maths is
unchanged. So we add **positional information**:

- classic Transformers add fixed sinusoidal **positional encodings**;
- Chronos-2 uses **RoPE** (rotary position embeddings) — it *rotates* the Q/K
  vectors by an angle proportional to position, which cleanly encodes **relative**
  distance (how far apart two moments are).

> In our forecasting demo we encode the hour/weekday/month **on a circle**
> (sin/cos) for the same reason RoPE exists: so "23:00 and 00:00" or "Dec and Jan"
> come out as neighbours, not opposite extremes.
"""

ENCODER_DECODER = r"""
### 🏗️ Encoder, decoder, and encoder-only

A Transformer stack can be wired in a few ways:

| Shape | What it does | Examples |
|---|---|---|
| **Encoder-only** | reads the input into rich representations, then an output head reads off the answer | BERT, **Chronos-2** |
| **Decoder-only** | generates a sequence left-to-right, each token attending to the ones before | GPT-style LLMs |
| **Encoder-decoder** | the encoder reads the input; the decoder generates the output while attending back to it (**cross-attention**) | T5, **Chronos v1 / Bolt** |

- **Autoregressive** generation (decoder-style) produces one step at a time — accurate but slow for long horizons.
- **Direct** output (encoder-only + a head) predicts the whole horizon at once — what Chronos-2 does for speed.

So the "encoder vs decoder" choice is really about **how the answer comes out**,
not about whether attention is used — both use it.
"""

GROUP_ATTENTION = r"""
### 👥 Group attention (how Chronos-2 sees many series at once)

Standard self-attention mixes information **within one sequence** (across time).
But real forecasting problems have **several related series and covariates** —
orders, weather, a promo flag, a holiday flag — and they influence each other.

**Chronos-2 adds a second kind of attention** on top of the temporal one:

- **Time attention** — self-attention *along time*, across the patches of a single
  series. "What in this series' past is relevant to its future?"
- **Group attention** — attention *across the group* of series/covariates at each
  time position. "How do the other channels (promo, rain, temperature) move this
  target right now?"

Stacking the two lets the model do **in-context learning across channels**: give
it a target plus its covariates as a group, and it learns — at inference, no
retraining — that e.g. a holiday flag lifts demand, or rain nudges it up. That is
exactly what the Chronos demo's **Multivariate series** tab simulates, and why v2
can be *covariate-aware* where v1 (single-series) could not.

> Intuition: **time attention** looks *back*; **group attention** looks *sideways*
> at the other signals happening at the same moment.
"""
