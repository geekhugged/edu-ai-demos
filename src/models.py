"""Симуляция двух режимов Chronos v2: zero-shot и fine-tuned.

Мы НЕ запускаем настоящую нейросеть (она тяжёлая и требует GPU/загрузки весов).
Вместо этого мы честно моделируем *принцип* разницы между режимами:

* **Zero-shot** — предобученная модель знает «типичную форму суток» (утро тихо,
  обед — холм, ужин — холм побольше), но не видела ИМЕННО ваш бизнес. Поэтому
  она хорошо ловит форму дня, но промахивается по недельной специфике
  (насколько именно у вас взлетают выходные).

* **Fine-tuned** — та же модель, дообученная на вашей истории. Она уже выучила
  профиль «час × день недели», поэтому точно попадает и в будни, и в выходные.

Это ровно тот эффект, ради которого делают fine-tuning: адаптация общего
предобученного «прайора» под конкретный ряд. Метрика качества — WAPE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error.

    WAPE = sum(|y - ŷ|) / sum(|y|).

    В отличие от MAPE, устойчива к нулям и малым значениям, поэтому её любят
    в ритейле и логистике. Возвращаем в процентах.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return 100.0 * np.abs(y_true - y_pred).sum() / denom


@dataclass
class ForecastResult:
    """Результат прогноза на тестовом окне."""

    timestamps: pd.DatetimeIndex
    y_true: np.ndarray
    zero_shot: np.ndarray
    fine_tuned: np.ndarray
    wape_zero_shot: float
    wape_fine_tuned: float

    @property
    def improvement_pct(self) -> float:
        """На сколько процентов fine-tuning снизил ошибку относительно zero-shot."""
        if self.wape_zero_shot == 0:
            return 0.0
        return 100.0 * (self.wape_zero_shot - self.wape_fine_tuned) / self.wape_zero_shot


def make_forecasts(
    df: pd.DataFrame,
    horizon_days: int = 14,
    ft_noise: float = 0.05,
    seed: int = 7,
) -> ForecastResult:
    """Построить прогнозы zero-shot и fine-tuned на последних ``horizon_days``.

    Логика (осознанно простая и интерпретируемая):

    * профиль обучаем на ИСТОРИИ (всё, что раньше тестового окна);
    * zero-shot ≈ профиль «по часу суток» (общий прайор формы дня);
    * fine-tuned ≈ профиль «час × день недели» + мягкий тренд (выучил ваш ряд).
    """
    rng = np.random.default_rng(seed)

    work = df.copy()
    split = len(work) - horizon_days * 24
    if split <= 0:
        raise ValueError("Слишком большой горизонт для данного объёма данных")

    train = work.iloc[:split]
    test = work.iloc[split:]

    # --- Zero-shot: только форма суток (час), без дня недели ---
    prof_hour = train.groupby("hour")["orders"].mean()
    zero_shot = test["hour"].map(prof_hour).to_numpy(dtype=float)

    # --- Fine-tuned: час × день недели (выучил недельную сезонность) ---
    prof_hd = train.groupby(["dow", "hour"])["orders"].mean()
    ft = np.array(
        [prof_hd.loc[(int(r.dow), int(r.hour))] for r in test.itertuples()],
        dtype=float,
    )

    # учёт мягкого тренда роста: масштабируем под уровень последней недели истории
    recent_level = train["orders"].iloc[-7 * 24 :].mean()
    hist_level = train["orders"].mean()
    trend_scale = recent_level / max(hist_level, 1e-6)
    zero_shot = zero_shot  # zero-shot тренд не «чувствует»
    ft = ft * trend_scale

    # немного неустранимого шума у fine-tuned (идеала не бывает)
    ft = ft * rng.normal(1.0, ft_noise, size=len(ft))
    ft = np.clip(ft, 0, None)

    y_true = test["orders"].to_numpy(dtype=float)

    return ForecastResult(
        timestamps=test.index,
        y_true=y_true,
        zero_shot=zero_shot,
        fine_tuned=ft,
        wape_zero_shot=wape(y_true, zero_shot),
        wape_fine_tuned=wape(y_true, ft),
    )
