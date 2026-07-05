"""Симуляция многомерных (multivariate) временных рядов с ковариатами.

Chronos v2, в отличие от первой версии, умеет учитывать не только сам ряд
заказов, но и внешние признаки (covariates): погода, промо-акции, выходные.
Здесь мы генерируем связанные ряды и показываем, как ковариаты «объясняют»
поведение целевого ряда заказов.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_multivariate(
    days: int = 21,
    seed: int = 3,
    beta_promo: float = 0.35,
    beta_rain: float = 0.25,
    beta_temp: float = -0.15,
) -> pd.DataFrame:
    """Сгенерировать целевой ряд заказов + три ковариаты.

    Ковариаты:
      * temperature — температура (°C), сезонно-суточная;
      * rain — интенсивность дождя (0..1), случайные ливни;
      * promo — флаг промо-акции (0/1), несколько кампаний.

    Целевой ряд строится как базовый суточный профиль, усиленный/ослабленный
    ковариатами через коэффициенты beta_*. Пользователь в UI может менять
    beta и видеть, как меняется связь.
    """
    from .data import daily_profile  # локальный импорт, чтобы избежать циклов

    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-06-01", periods=days * 24, freq="h")
    hours = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()

    # --- ковариаты ---
    temperature = (
        18
        + 8 * np.sin(2 * np.pi * (hours - 15) / 24)  # днём теплее
        + 3 * np.sin(2 * np.pi * doy / 365)
        + rng.normal(0, 1.0, len(index))
    )

    rain = np.zeros(len(index))
    n_showers = rng.integers(6, 12)
    for _ in range(n_showers):
        start = rng.integers(0, len(index) - 6)
        length = rng.integers(2, 6)
        rain[start : start + length] = np.clip(rng.uniform(0.3, 1.0), 0, 1)

    promo = np.zeros(len(index))
    n_campaigns = rng.integers(2, 4)
    for _ in range(n_campaigns):
        day = rng.integers(0, days)
        promo[day * 24 : (day + 1) * 24] = 1.0

    weekend = (dow >= 5).astype(float)

    # --- целевой ряд ---
    base = daily_profile(hours) * (1.0 + 0.20 * weekend)
    # нормируем ковариаты к ~[-1..1] для честного влияния beta
    temp_z = (temperature - temperature.mean()) / temperature.std()

    signal = base * (
        1.0
        + beta_promo * promo
        + beta_rain * rain
        + beta_temp * temp_z
    )
    noise = rng.normal(1.0, 0.06, len(index))
    orders = np.clip(np.round(signal * noise), 0, None)

    return pd.DataFrame(
        {
            "orders": orders,
            "temperature": temperature.round(1),
            "rain": rain.round(2),
            "promo": promo,
            "weekend": weekend,
        },
        index=index,
    )


def reconstruct(
    df: pd.DataFrame,
    use_promo: bool,
    use_rain: bool,
    use_temp: bool,
    beta_promo: float,
    beta_rain: float,
    beta_temp: float,
) -> tuple[np.ndarray, float]:
    """Простейшая «модель», которая пытается восстановить orders из ковариат.

    Возвращает восстановленный ряд и WAPE. Показывает: чем больше релевантных
    ковариат мы включаем, тем ближе восстановление к реальности.
    """
    from .data import daily_profile
    from .models import wape

    hours = df.index.hour.to_numpy()
    dow = df.index.dayofweek.to_numpy()
    weekend = (dow >= 5).astype(float)

    base = daily_profile(hours) * (1.0 + 0.20 * weekend)
    temp = df["temperature"].to_numpy()
    temp_z = (temp - temp.mean()) / temp.std()

    factor = np.ones(len(df))
    if use_promo:
        factor = factor + beta_promo * df["promo"].to_numpy()
    if use_rain:
        factor = factor + beta_rain * df["rain"].to_numpy()
    if use_temp:
        factor = factor + beta_temp * temp_z

    recon = base * factor
    return recon, wape(df["orders"].to_numpy(), recon)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Корреляции целевого ряда с ковариатами (для heatmap)."""
    cols = ["orders", "temperature", "rain", "promo", "weekend"]
    return df[cols].corr()
