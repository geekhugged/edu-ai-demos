"""A general glossary of terms used across the demos.

Kept as plain data (term, category, definition) so it's easy to extend, test,
and reuse. The Glossary page renders and filters it.
"""
from __future__ import annotations

# (term, category, definition)
GLOSSARY: list[tuple[str, str, str]] = [
    # ── Attention & Transformers ────────────────────────────────────────────
    ("Transformer", "Attention & Transformers",
     "A neural network built from stacked attention + feed-forward layers. It "
     "processes a whole sequence in parallel and lets any position directly "
     "influence any other."),
    ("Attention", "Attention & Transformers",
     "The mechanism that weighs how much each element of the input matters for a "
     "given position. Weights come from the similarity between a *query* and "
     "*keys*, normalized by softmax, then used to average the *values*."),
    ("Self-attention", "Attention & Transformers",
     "Attention where the queries, keys and values all come from the same "
     "sequence — every position can look at every other position."),
    ("Cross-attention", "Attention & Transformers",
     "Attention where queries come from one sequence and keys/values from "
     "another (e.g. a decoder attending to the encoder's output)."),
    ("Multi-head attention", "Attention & Transformers",
     "Running several attention operations ('heads') in parallel, each in a "
     "different subspace, then concatenating — so the model can attend to several "
     "kinds of relationship at once."),
    ("Query / Key / Value (Q/K/V)", "Attention & Transformers",
     "Three vectors derived from each token. The query is matched against every "
     "key to get attention weights; those weights then average the values."),
    ("Softmax", "Attention & Transformers",
     "Turns a vector of scores into positive weights that sum to 1. Used to "
     "normalize attention scores (and control sharpness via a temperature)."),
    ("Positional encoding", "Attention & Transformers",
     "Extra signal added so the model knows the *order* of tokens — attention on "
     "its own is order-agnostic."),
    ("RoPE (Rotary Position Embedding)", "Attention & Transformers",
     "A positional scheme that rotates the query/key vectors by an angle "
     "proportional to position, encoding *relative* distance. Used by Chronos-2."),
    ("Encoder", "Attention & Transformers",
     "The stack that reads the input sequence into contextual representations."),
    ("Decoder", "Attention & Transformers",
     "The stack that generates the output sequence, usually autoregressively, "
     "attending back to the encoder."),
    ("Encoder-only vs encoder-decoder", "Attention & Transformers",
     "Encoder-only models (Chronos-2, BERT) just encode and read off an output "
     "head; encoder-decoder models (T5, Chronos v1) add a decoder that generates "
     "step by step."),
    ("Feed-forward (MLP) block", "Attention & Transformers",
     "The per-position two-layer network applied after attention in each "
     "transformer layer."),
    ("Layer norm", "Attention & Transformers",
     "Normalization applied within each layer to keep activations stable during "
     "training."),
    ("Embedding", "Attention & Transformers",
     "A learned vector representation of a token, patch, or position."),
    ("T5", "Attention & Transformers",
     "A well-known encoder-decoder transformer family; the backbone Chronos is "
     "built on (v1/Bolt use encoder+decoder, Chronos-2 uses the encoder only)."),
    ("Time attention", "Attention & Transformers",
     "In Chronos-2, self-attention applied along the *time* axis — across the "
     "patches of a single series."),
    ("Group attention", "Attention & Transformers",
     "In Chronos-2, attention *across a group* of related series and covariates, "
     "enabling in-context learning between them (how a promo or the weather "
     "moves the target)."),
    ("Patch / patching", "Attention & Transformers",
     "Grouping several consecutive time steps into one token/embedding, so the "
     "model processes fewer, richer units (Chronos-Bolt / v2)."),
    ("Autoregressive", "Attention & Transformers",
     "Generating one step at a time, feeding each prediction back as input for "
     "the next step (Chronos v1)."),
    ("Direct multi-step forecasting", "Attention & Transformers",
     "Predicting the whole horizon in a single shot instead of step by step "
     "(Chronos-Bolt / v2) — much faster."),
    ("Foundation model", "Attention & Transformers",
     "A large model pretrained on massive data, usable zero-shot or fine-tuned "
     "for many downstream tasks."),

    # ── Time-series forecasting ────────────────────────────────────────────
    ("Zero-shot", "Time-series forecasting",
     "Using a pretrained model on new data with no training — just feed the "
     "history as context and read off the forecast."),
    ("Fine-tuning", "Time-series forecasting",
     "Continuing training of a pretrained model on your data so its weights adapt "
     "to your specific patterns."),
    ("In-context learning", "Time-series forecasting",
     "The model 'learns' the pattern from the context it is given at inference, "
     "without any weight updates — the basis of zero-shot."),
    ("Context length", "Time-series forecasting",
     "How many past points the model reads as input. Too short and it can't see "
     "a weekly/yearly cycle."),
    ("Horizon (prediction length)", "Time-series forecasting",
     "How many future points to forecast. Longer horizons compound error."),
    ("Covariate", "Time-series forecasting",
     "An external feature that helps explain the target (weather, promo, holiday, "
     "price)."),
    ("Univariate vs multivariate", "Time-series forecasting",
     "Univariate = one series; multivariate = several related series forecast "
     "jointly (with cross-series information)."),
    ("Seasonality", "Time-series forecasting",
     "Repeating patterns at fixed periods — daily, weekly, yearly."),
    ("Trend", "Time-series forecasting",
     "A slow, long-term drift of the level up or down."),
    ("Quantile (P10 / P50 / P90)", "Time-series forecasting",
     "The value below which a given fraction of outcomes fall. Used to express "
     "uncertainty as bands (P50 = median)."),
    ("Mean scaling", "Time-series forecasting",
     "Dividing a series by its mean so that magnitude stops mattering — only the "
     "shape does."),
    ("Tokenization (time series)", "Time-series forecasting",
     "Turning continuous values into discrete tokens (Chronos v1, via "
     "quantization). Bolt/v2 skip this and read raw patches."),
    ("Quantization / bins", "Time-series forecasting",
     "Cutting the value range into B buckets; each value becomes a bucket index "
     "(a token)."),

    # ── Training & fine-tuning ─────────────────────────────────────────────
    ("LoRA (Low-Rank Adaptation)", "Training & fine-tuning",
     "A parameter-efficient fine-tuning method: freeze the base weights and train "
     "small low-rank matrices injected into (usually) the attention projections. "
     "Only ~0.1–1% of parameters train — cheap, fast, resists forgetting."),
    ("PEFT (Parameter-Efficient Fine-Tuning)", "Training & fine-tuning",
     "Umbrella term for methods (LoRA, adapters, prefix-tuning) that adapt a model "
     "by training a tiny fraction of its parameters."),
    ("Adapter", "Training & fine-tuning",
     "Small trainable modules inserted between otherwise-frozen layers."),
    ("Learning rate", "Training & fine-tuning",
     "The step size of each weight update — the single most important training "
     "knob. Too high → instability/forgetting; too low → barely adapts."),
    ("Epoch / step", "Training & fine-tuning",
     "One full pass over the training data / one gradient update."),
    ("Batch size", "Training & fine-tuning",
     "How many examples per gradient update. Larger = smoother, more memory."),
    ("Warmup", "Training & fine-tuning",
     "Ramping the learning rate up gradually at the start of training to avoid a "
     "destructive first step."),
    ("Weight decay", "Training & fine-tuning",
     "L2 regularization that shrinks weights to fight overfitting."),
    ("Early stopping", "Training & fine-tuning",
     "Halting training once validation performance stops improving."),
    ("Validation split (time-based)", "Training & fine-tuning",
     "Holding out the most recent slice chronologically to evaluate and pick "
     "models. A *random* split leaks the future and inflates every metric."),
    ("Catastrophic forgetting", "Training & fine-tuning",
     "When fine-tuning erases the general knowledge the model had — usually from "
     "too high a learning rate."),
    ("Cross-entropy loss", "Training & fine-tuning",
     "The classification loss used to train token-predicting models (Chronos v1)."),
    ("Overfitting / underfitting", "Training & fine-tuning",
     "Memorizing noise in the training set vs failing to learn the pattern at all."),
    ("Temperature (sampling)", "Training & fine-tuning",
     "Scales randomness when sampling — higher = more diverse/spread-out outputs, "
     "lower = sharper/more typical."),
    ("top-k / top-p (nucleus)", "Training & fine-tuning",
     "Truncate the next-token distribution to the k most likely tokens / to the "
     "smallest set covering probability p."),

    # ── Metrics ────────────────────────────────────────────────────────────
    ("WAPE", "Metrics",
     "Weighted Absolute Percentage Error = Σ|y−ŷ| / Σ|y|. Robust to zeros; a "
     "standard in retail and logistics."),
    ("WQL", "Metrics",
     "Weighted Quantile Loss — evaluates the quality of *quantile* (interval) "
     "forecasts, not just the median."),
    ("MAPE", "Metrics",
     "Mean Absolute Percentage Error — the average of |y−ŷ|/|y|. Blows up near "
     "zero values, which is why WAPE is often preferred."),

    # ── Chronos models ─────────────────────────────────────────────────────
    ("Chronos (v1)", "Chronos models",
     "Amazon's original time-series foundation model: tokenize values, T5 "
     "encoder-decoder, autoregressive decoding."),
    ("Chronos-Bolt", "Chronos models",
     "A faster Chronos: patch-based, still T5 encoder-decoder, but direct "
     "multi-step quantile forecasts (~250× faster than v1)."),
    ("Chronos-2", "Chronos models",
     "Encoder-only foundation model (the T5 encoder) with time + group attention "
     "and RoPE; supports multivariate series and covariates. The model this demo "
     "is about."),
]


def categories() -> list[str]:
    """Category names in first-seen order."""
    seen: list[str] = []
    for _term, cat, _defn in GLOSSARY:
        if cat not in seen:
            seen.append(cat)
    return seen


def search(query: str) -> list[tuple[str, str, str]]:
    """Filter entries whose term or definition contains the query (case-insensitive)."""
    q = query.strip().lower()
    if not q:
        return list(GLOSSARY)
    return [e for e in GLOSSARY if q in e[0].lower() or q in e[2].lower()]
