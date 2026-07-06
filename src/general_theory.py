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

CREATIVITY_NOTE = r"""
**“Creativity” is the softmax _temperature_ — the very same dial that controls attention.**

After the model scores every possible next word, it turns those scores $s_i$ into
probabilities with a **softmax**, divided by a **temperature** $T$:

$$p_i = \frac{e^{\,s_i/T}}{\sum_j e^{\,s_j/T}}$$

- 🧊 **Low $T$ (→ 0) — cautious.** The biggest score dominates, so the model almost
  always picks the *single most likely* word. Output is safe, predictable, and can
  get repetitive.
- ⚖️ **$T = 1$** — the model's raw probabilities, untouched.
- 🔥 **High $T$ (→ ∞) — adventurous.** The scores flatten toward equal, so rarer
  words get a real chance. Output is varied and surprising — but can wander into
  nonsense.

So "creativity" isn't a separate idea — it's simply **how sharp or soft the
softmax is**. And it is *literally the same knob* as **attention sharpness** on the
other tabs: low temperature = laser-focus on the single best key; high temperature
= attention spread thinly across many. **One mechanism, two uses** — choosing the
next word, and choosing what to pay attention to.
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

RNN_VS_ATTENTION = r"""
#### 🔁 RNN vs ⚡ Attention — the one picture to remember

Before Transformers, sequences were read by **RNNs**: one step at a time,
left→right, passing a little "memory" from each word to the next — like a
**game of telephone (whisper down the line)**. To let word #1 influence word
#50, its signal has to survive **49 whispers** — and it usually **fades**.

**Attention** throws that out. Every word can look at **every other word
directly, in a single step** — like everyone in a room being able to talk to
anyone instantly. Distance stops mattering.

|  | 🔁 **RNN** (telephone line) | ⚡ **Attention** (everyone in a room) |
|---|---|---|
| Reads the sequence | one step at a time, in order | all positions at once (in parallel) |
| Word 1 → word N | signal hops through **N−1** cells, fading | **1 direct look**, any distance |
| Long-range memory | weak — it forgets | strong — direct access |
| Training speed | slow (must go in order) | fast (all at once) |

**Why it matters for our forecast:** to use *"the same hour last week"* an RNN
must carry that memory through **168 hourly steps**; attention just **jumps
straight to it**. That direct reach is exactly what the attention simulations in
the Chronos demo show.
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

**The simplest way to picture it — a *reader* and a *writer*:**

- 📖 **Encoder = the reader.** It reads the whole input and *understands* it. It
  doesn't produce any new text — it just "gets the gist."
- ✍️ **Decoder = the writer.** It *produces* the output one piece at a time, each
  new piece based on what it has written so far.

**Three everyday examples make it click:**

- 🌍 **Translate English → French** = *read **and** write* → **encoder-decoder**.
  First **read** the whole English sentence to understand it (encoder), then
  **write** the French one word by word (decoder), glancing back at the original
  (**cross-attention**).  *“The cat is black” → “Le chat est noir.”*
- 💬 **ChatGPT finishing your sentence** = *only writing* → **decoder-only**. There's
  nothing separate to "read" — it just keeps guessing the next word, over and over.
- 👍 **“Is this review positive or negative?”** or **“forecast next week's sales”**
  = *read everything, give one answer* → **encoder-only**. No word-by-word writing:
  read the input, output a label or a number. **Chronos-2 works exactly like
  this** — read the history → produce the whole forecast in one go.

---

Same idea as a table:

| Shape | In one line | Everyday example | Models |
|---|---|---|---|
| 📖 **Encoder-only** | read & understand → one answer | sentiment of a review; a sales forecast | BERT, **Chronos-2** |
| ✍️ **Decoder-only** | write left-to-right, one token at a time | ChatGPT autocomplete | GPT-style LLMs |
| 📖✍️ **Encoder-decoder** | read the input, then write the output | translation, summarization | T5, **Chronos v1 / Bolt** |

- **Autoregressive** (decoder-style) writing produces one step at a time —
  accurate but slow for long outputs.
- **Direct** output (encoder-only + a small head) gives the whole answer at once —
  what Chronos-2 does, which is a big reason it's fast.

So "encoder vs decoder" is really just **reading vs writing** — *how the answer
comes out*. It's **not** about attention: **all three use attention.**
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
