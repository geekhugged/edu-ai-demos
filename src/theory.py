"""Theory material for Chronos v2 at three difficulty levels.

Kept separate from the UI so it's easy to edit and reuse.
"""

BEGINNER = r"""
### 🟢 Level 1 — Beginner: the plain-English version

**What does Chronos actually do?**
Think of a weather forecast, but instead of rain we predict **how many food
orders will arrive each hour**. Chronos is a large neural network from Amazon
that has "read" millions of different time series (sales, traffic, weather,
energy) and learned to guess *how a series continues*.

**The key idea — language instead of numbers.**
Chronos treats a series of numbers the same way a language model (like ChatGPT)
treats text. Each value in the series becomes a "word" (a token), and the model
simply predicts the **next word**. It's just that the "words" here are order
levels.

> 📌 Analogy: you read "quiet in the morning → a spike at lunch → a big spike in
> the evening" so many times that you can finish the sentence without even
> knowing which restaurant it is. That's how Chronos "finishes" your day.

**Zero-shot vs Fine-tuned — what's the difference?**
- 🟦 **Zero-shot** — the model sees your history for the *first time* and
  forecasts right away. Like an experienced courier in a new city: knows the
  general rules, but not the local quirks yet.
- 🟩 **Fine-tuned** — the same model, lightly "re-trained" on your data. Now the
  courier has worked for you for a month and knows that *Saturday evenings get
  slammed*.

**WAPE** is simply "on average, by what percentage are we wrong."
Lower is better. An 8% error is better than 20%.
"""

INTERMEDIATE = r"""
### 🟡 Level 2 — Intermediate: how it works

**1. Tokenizing the series.**
Chronos turns continuous values into discrete tokens in three steps:
1. **Scaling** — the series is divided by its mean (mean scaling), so that
   "5 orders" and "5000 orders" look the same to the model in terms of shape.
2. **Quantization** — the value range is cut into `B` bins. Each value lands in
   its bin → we get a token index, like a letter in an alphabet.
3. From here it's an ordinary sequence of tokens, just like a sentence.

**2. Architecture — a language model (T5).**
The first version of Chronos uses an encoder-decoder transformer (the T5
family). It predicts a probability distribution over the next token, not a
single number. This lets us sample **many trajectories** and build confidence
intervals (P10 / P50 / P90) rather than just a point forecast.

**3. Zero-shot vs Fine-tuning.**
- *Zero-shot* works through **in-context learning**: your entire history is fed
  in as "context," and the model continues it, relying on patterns learned from
  a huge corpus of series. No weight training is required.
- *Fine-tuning* — a few epochs of extra training on your series. The model
  shifts its "prior" toward your specifics (for example, the strength of the
  weekly seasonality). It usually gives a noticeable accuracy boost for a small
  cost.

**4. The WAPE metric.**
$$\mathrm{WAPE} = \frac{\sum_t |y_t - \hat{y}_t|}{\sum_t |y_t|}$$
Unlike MAPE, it doesn't blow up on zero/small values (and hourly orders are
sometimes zero at night). That's why WAPE is a standard in retail and logistics.

**5. What's new in Chronos v2.**
The second version made the model more practical for business:
- support for **covariates** (promos, weather, price) and **multivariate** series;
- group / in-context attention between related series;
- fast inference (in the spirit of Chronos-Bolt: patches + direct multi-step
  forecasting).
"""

ADVANCED = r"""
### 🔴 Level 3 — Advanced: details and math

**Tokenization and the distribution head.**
Let the series $x_{1:C}$ be the context. Mean-scaling: $\tilde{x}_t = x_t / s$,
where $s = \frac{1}{C}\sum |x_t|$. Then quantization into $B$ levels (typically
$B=4096$) yields tokens $z_t \in \{1,\dots,B\}$. The model learns
$p_\theta(z_{t+1} \mid z_{1:t})$ via cross-entropy — i.e. classification over
bins, not regression. Forecasts are obtained by autoregressively sampling $k$
trajectories and taking per-quantile estimates.

**Attention (the transformer core).**
$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
Each token forms a query $Q$, keys $K$, and values $V$. The matrix
$QK^\top/\sqrt{d_k}$ holds pairwise "similarities"; a row-wise softmax gives the
weights; the output is a weighted sum of $V$. The $\sqrt{d_k}$ divisor
stabilizes gradients. For time series this lets the model "reach" directly for
the relevant past (the same hour yesterday, the same week) without the decay you
get in an RNN.

**From autoregression to patches (Chronos-Bolt / v2).**
Token-by-token autoregressive decoding is slow (O(H) passes for a horizon $H$).
The Bolt approach cuts the series into **patches** (a window of several points →
one token) and predicts the whole horizon at once (**direct multi-horizon**),
which gives a multiplicative speedup at inference and often better accuracy on
long horizons.

**Multivariate and covariates in v2.**
Chronos v2 extends the model from univariate to **multivariate /
covariate-aware** forecasting: several related series and external features are
processed jointly through attention *between* series (cross-series / group
attention). The model learns that, e.g., a promo flag lifts the target series
rather than forecasting it in a vacuum. This is exactly what the "Multivariate
series" tab simulates.

**Fine-tuning in practice.**
- Freeze/unfreeze parts of the network; small learning rate (adapting the prior).
- Validate over time (rolling / expanding origin), NOT a random split — otherwise
  you leak information from the future.
- Metrics on quantiles: besides WAPE, look at the **weighted quantile loss (WQL)**
  to assess interval quality, not just the median.

**Pitfalls.**
Leakage through scaling (compute scaling on the train window only), distribution
shift (covariate shift) when the season or promo policy changes, and a
"too-smooth" zero-shot that underestimates rare spikes — which is exactly what
fine-tuning fixes.
"""
