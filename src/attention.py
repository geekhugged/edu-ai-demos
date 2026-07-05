"""Мини-симуляция механизма внимания (attention) для новичков.

Идея, которую хотим показать «на пальцах»:

    Чтобы предсказать заказы в конкретный час (например, 19:00 — ужин),
    модель «оглядывается назад» и смотрит на похожие моменты в прошлом.
    На похожие часы она смотрит ВНИМАТЕЛЬНО (большой вес), на непохожие —
    почти не смотрит (маленький вес). Прогноз = взвешенная сумма прошлого.

Это и есть attention: query (что ищем) сравнивается с keys (что есть в
прошлом), из сходства получаем веса через softmax, и берём взвешенную сумму
values. Никакой тяжёлой математики — только скалярное произведение и softmax.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax с «температурой» (резкостью).

    temperature < 1  → внимание острее (почти all-in в один ключ);
    temperature > 1  → внимание размазаннее (смотрим на всё понемногу).
    """
    z = np.asarray(x, dtype=float) / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def hour_embedding(hour: float) -> np.ndarray:
    """Циклическое представление часа суток двумя числами (sin, cos).

    Благодаря этому 23:00 и 00:00 оказываются «рядом», а не на разных концах.
    """
    ang = 2 * np.pi * hour / 24.0
    return np.array([np.sin(ang), np.cos(ang)])


@dataclass
class AttentionResult:
    key_hours: np.ndarray      # часы токенов-ключей (что было в прошлом)
    key_values: np.ndarray     # значения (заказы) в эти часы
    scores: np.ndarray         # «сырое» сходство query·key
    weights: np.ndarray        # веса внимания после softmax (сумма = 1)
    prediction: float          # итоговый прогноз = взвешенная сумма values
    query_hour: float


def run_attention(
    key_hours: np.ndarray,
    key_values: np.ndarray,
    query_hour: float,
    temperature: float = 1.0,
) -> AttentionResult:
    """Посчитать attention для одного запроса.

    * Query — вектор целевого часа (какой момент прогнозируем).
    * Keys — векторы часов из прошлого.
    * Score = Q · K (скалярное произведение: чем ближе час, тем больше).
    * Weights = softmax(Score).
    * Prediction = сумма weights * values.
    """
    key_hours = np.asarray(key_hours, dtype=float)
    key_values = np.asarray(key_values, dtype=float)

    q = hour_embedding(query_hour)
    keys = np.stack([hour_embedding(h) for h in key_hours])

    scores = keys @ q  # (N,) скалярные произведения
    # масштаб, как в трансформере (делим на sqrt(d)), d=2
    scores = scores / np.sqrt(keys.shape[1])
    weights = softmax(scores, temperature=temperature)
    prediction = float((weights * key_values).sum())

    return AttentionResult(
        key_hours=key_hours,
        key_values=key_values,
        scores=scores,
        weights=weights,
        prediction=prediction,
        query_hour=query_hour,
    )


def sentence_attention(tokens: list[str], focus_idx: int, temperature: float = 1.0):
    """Совсем простой пример внимания на словах (для интуиции).

    Каждому слову даём маленький «смысловой» вектор, а вниманием показываем,
    на какие слова смотрит выбранное слово. Веса подобраны так, чтобы
    связанные по смыслу слова притягивались (еда↔вкусно, дождь↔заказ).
    """
    # игрушечные 3-мерные эмбеддинги: [еда, погода, действие]
    vocab = {
        "дождь":   np.array([0.0, 1.0, 0.1]),
        "идёт":    np.array([0.0, 0.4, 0.9]),
        "поэтому": np.array([0.1, 0.2, 0.5]),
        "заказы":  np.array([0.9, 0.3, 0.6]),
        "еды":     np.array([1.0, 0.0, 0.1]),
        "растут":  np.array([0.4, 0.1, 0.9]),
    }
    vecs = np.stack([vocab.get(t, np.array([0.3, 0.3, 0.3])) for t in tokens])
    q = vecs[focus_idx]
    scores = (vecs @ q) / np.sqrt(vecs.shape[1])
    weights = softmax(scores, temperature=temperature)
    return weights
