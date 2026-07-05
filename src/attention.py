"""A mini-simulation of the attention mechanism, for beginners.

The idea we want to show intuitively:

    To predict the orders in a particular hour (say 19:00 — dinner), the model
    "looks back" at similar moments in the past. It looks CAREFULLY at similar
    hours (a large weight) and barely looks at dissimilar ones (a small weight).
    The forecast = a weighted sum of the past.

That's attention: a query (what we're looking for) is compared with keys (what's
in the past), the similarity is turned into weights via softmax, and we take a
weighted sum of the values. No heavy math — just a dot product and a softmax.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with a "temperature" (sharpness).

    temperature < 1  → sharper attention (almost all-in on one key);
    temperature > 1  → more diffuse attention (looking at everything a little).
    """
    z = np.asarray(x, dtype=float) / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def cyclic_embedding(value: float, period: float) -> np.ndarray:
    """A cyclic representation of a position within a period as (sin, cos).

    Thanks to this, the two ends of the cycle end up "close" rather than at
    opposite ends: 23:00 ↔ 00:00 (period 24), Sunday ↔ Monday (period 7),
    December ↔ January (period 12).
    """
    ang = 2 * np.pi * value / period
    return np.array([np.sin(ang), np.cos(ang)])


def hour_embedding(hour: float) -> np.ndarray:
    """Cyclic embedding of the hour of day (period 24). Kept for convenience."""
    return cyclic_embedding(hour, 24.0)


@dataclass
class AttentionResult:
    key_positions: np.ndarray  # positions of the key tokens (past slots in the cycle)
    key_values: np.ndarray     # values (orders) at those positions
    scores: np.ndarray         # "raw" query·key similarity
    weights: np.ndarray        # attention weights after softmax (sum = 1)
    prediction: float          # final forecast = weighted sum of values
    query_position: float
    period: float


def run_attention(
    key_positions: np.ndarray,
    key_values: np.ndarray,
    query_position: float,
    period: float = 24.0,
    temperature: float = 1.0,
) -> AttentionResult:
    """Compute cyclic attention for a single query.

    Works for any period: hour of day (24), day of week (7), month of year (12).

    * Query — the vector of the target position (which slot we're forecasting).
    * Keys — the vectors of positions from the past.
    * Score = Q · K (dot product: the closer within the cycle, the larger it is).
    * Weights = softmax(Score).
    * Prediction = sum of weights * values.
    """
    key_positions = np.asarray(key_positions, dtype=float)
    key_values = np.asarray(key_values, dtype=float)

    q = cyclic_embedding(query_position, period)
    keys = np.stack([cyclic_embedding(p, period) for p in key_positions])

    scores = keys @ q  # (N,) dot products
    # scale, as in the transformer (divide by sqrt(d)), d=2
    scores = scores / np.sqrt(keys.shape[1])
    weights = softmax(scores, temperature=temperature)
    prediction = float((weights * key_values).sum())

    return AttentionResult(
        key_positions=key_positions,
        key_values=key_values,
        scores=scores,
        weights=weights,
        prediction=prediction,
        query_position=query_position,
        period=period,
    )


# tokens used by the word-level attention example (kept here so UI and logic agree)
SENTENCE_TOKENS = ["rain", "falls", "so", "food", "orders", "grow"]


def sentence_attention(tokens: list[str], focus_idx: int, temperature: float = 1.0):
    """A very simple attention example on words (for intuition).

    We give every word a small "meaning" vector, and attention shows which words
    a chosen word looks at. The weights are set up so semantically related words
    attract each other (food ↔ orders, rain ↔ orders).
    """
    # toy 3-d embeddings: [food, weather, action]
    vocab = {
        "rain":   np.array([0.0, 1.0, 0.1]),
        "falls":  np.array([0.0, 0.4, 0.9]),
        "so":     np.array([0.1, 0.2, 0.5]),
        "food":   np.array([1.0, 0.0, 0.1]),
        "orders": np.array([0.9, 0.3, 0.6]),
        "grow":   np.array([0.4, 0.1, 0.9]),
    }
    vecs = np.stack([vocab.get(t, np.array([0.3, 0.3, 0.3])) for t in tokens])
    q = vecs[focus_idx]
    scores = (vecs @ q) / np.sqrt(vecs.shape[1])
    weights = softmax(scores, temperature=temperature)
    return weights
