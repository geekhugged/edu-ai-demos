"""Key numbers about the Chronos family — as data + sourced prose.

Figures are attributed to the specific model they belong to (v1 / Bolt / v2),
because they differ a lot. Sources are listed at the bottom of the page.
"""

# Headline metric cards: (label, value, note)
HEADLINE: list[tuple[str, str, str]] = [
    ("Parameters · Chronos-2", "≈120M", "encoder-only (T5 encoder)"),
    ("Model-family range", "8M – 710M", "Chronos v1: Tiny → Large"),
    ("Pretraining datasets", "28", "real-world datasets (v1 corpus)"),
    ("Pretraining series", "≈890K", "univariate series (v1)"),
    ("Pretraining observations", "≈84B", "data points (v1); Bolt ≈100B"),
    ("Token vocabulary", "4096", "quantization bins (v1)"),
    ("Inference speed", ">300 series/s", "Chronos-2 on one A10G GPU"),
    ("Bolt vs v1 speedup", "up to 250×", "and ≈20× less memory"),
]

# Q&A sections (title, markdown body)
QA: list[tuple[str, str]] = [
    ("1️⃣ How many parameters (weights)?", r"""
- **Chronos-2** — the model this demo is about — has **≈120M parameters** and is
  **encoder-only** (it keeps just the T5 *encoder*).
- The **original Chronos (v1)** shipped as a T5 family of five sizes:
  **Tiny ≈8M · Mini ≈20M · Small ≈46M · Base ≈200M · Large ≈710M** (plus a
  GPT-2-base ≈90M variant). They use a small **4096-token** vocabulary instead of
  T5's 32,128 — which is why they have fewer parameters than a same-size T5.
"""),
    ("2️⃣ How much time-series data was used?", r"""
- **Chronos v1** was pretrained on **28 real-world datasets** — about
  **≈890,000 univariate time series** totalling **≈84 billion observations**
  (data points). On top of that it added **10M** `TSMixup` augmentations and
  **1M** purely-synthetic series (`KernelSynth`, from Gaussian processes).
- **Chronos-Bolt** was trained on **≈100 billion** time-series observations.
- **Chronos-2** draws from the **Chronos + GIFT-Eval** pretraining corpora and
  adds synthetic data (a *TSI* trend/seasonality generator and a *TCM* temporal
  causal-model generator) — the synthetic part is what teaches it multivariate
  and covariate tasks.
"""),
    ("3️⃣ How many sectors / domains were used?", r"""
The pretraining corpus spans **roughly seven broad domains** so the model sees
many kinds of dynamics, e.g.:

- ⚡ **energy** (electricity, load)   ·   🚗 **transport / mobility**
- 🌦️ **nature / weather**   ·   🌐 **web & cloud-ops** (traffic, metrics)
- 🏥 **healthcare**   ·   💰 **finance / economics**   ·   🛒 **retail / sales**

Breadth is the point: a model that has seen electricity, traffic *and* sales
learns the **shared shapes** (daily/weekly cycles, spikes, trends) that transfer
to a brand-new series it has never seen — like our food-delivery example.
"""),
    ("4️⃣ What's the benefit of pretraining?", r"""
Pretraining once on a huge, diverse corpus buys you:

- 🎯 **Zero-shot forecasting** — point it at a new series and get a forecast with
  **no training at all** (just feed the history as context).
- ⚡ **Deploy in seconds** — no per-dataset model to build, tune and maintain.
- 📈 **Strong accuracy** — often matches or beats models trained specifically on
  the target dataset.
- 📊 **Uncertainty for free** — probabilistic **quantile** forecasts (P10/P50/P90),
  not just a single line.
- 🔧 **A great starting point** — when you want more, **fine-tuning** on your own
  data adapts the prior cheaply (see the Chronos v2 demo's Zero-shot vs Fine-tuned
  tab).
"""),
    ("5️⃣ How long does pretraining take?", r"""
It's a **one-time, offline** cost paid by Amazon — you never repeat it:

- **Chronos-T5-Large (710M)** took about **63 hours** on **8× A100 (40 GB)** GPUs
  (an AWS `p4d.24xlarge`), trained for **200,000 steps** with AdamW and an
  effective **batch size of 256**. Smaller sizes train faster.
- Chronos-2's exact figure isn't published, but it's the same shape: **on the
  order of days on a multi-GPU node**.

The whole idea: pay this **once**, then everyone gets **zero-shot** forecasts in
milliseconds — or a short **fine-tune** (minutes–hours on a single GPU) for extra
accuracy.
"""),
    ("➕ A few more numbers", r"""
- **Architecture**: Chronos-2 adds **time attention** (along time) + **group
  attention** (across related series/covariates) with **RoPE** positions.
- **Output**: v1 samples tokens autoregressively; **Bolt & v2** predict the whole
  horizon **directly** as quantiles — far fewer forward passes.
- **Efficiency**: Chronos-Bolt is **up to 250× faster** and **≈20× more
  memory-efficient** than the same-size v1.
"""),
]

# (label, url)
SOURCES: list[tuple[str, str]] = [
    ("Chronos: Learning the Language of Time Series (paper, arXiv 2403.07815)",
     "https://arxiv.org/abs/2403.07815"),
    ("Chronos-2: From Univariate to Universal Forecasting (paper, arXiv 2510.15821)",
     "https://arxiv.org/abs/2510.15821"),
    ("amazon/chronos-2 model card (Hugging Face)",
     "https://huggingface.co/amazon/chronos-2"),
    ("Fast and accurate zero-shot forecasting with Chronos-Bolt (AWS blog)",
     "https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/"),
]
