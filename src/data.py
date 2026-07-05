"""Генерация синтетических данных food-delivery заказов.

Данные почасовые за год. Внутри суток — два «холма»:
  * обеденный пик около 13:00 (поменьше);
  * ужинный пик около 19:30 (побольше).

Плюс недельная сезонность (выходные активнее), плавный годовой тренд роста,
мягкая сезонность по месяцам и реалистичный шум. Всё детерминировано через
seed, чтобы демо было воспроизводимым.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _gaussian_peak(hours: np.ndarray, center: float, width: float, height: float) -> np.ndarray:
    """Один колоколообразный «холм» активности в течение суток."""
    return height * np.exp(-0.5 * ((hours - center) / width) ** 2)


def daily_profile(hours: np.ndarray) -> np.ndarray:
    """Базовый профиль заказов по часам суток (без учёта дня недели/сезона).

    Возвращает средний уровень заказов для каждого часа [0..23].
    """
    lunch = _gaussian_peak(hours, center=13.0, width=1.1, height=45.0)   # обед — холм поменьше
    dinner = _gaussian_peak(hours, center=19.5, width=1.6, height=75.0)  # ужин — холм побольше
    # небольшая ночная/утренняя база, чтобы линия не падала в ноль
    base = 6.0 + 3.0 * np.exp(-0.5 * ((hours - 22.0) / 3.0) ** 2)
    return lunch + dinner + base


def _weekday_factor(dow: np.ndarray) -> np.ndarray:
    """Множитель активности по дню недели (0=Пн ... 6=Вс).

    Пятница/суббота/воскресенье заметно активнее — люди чаще заказывают еду.
    """
    factors = np.array([0.92, 0.90, 0.95, 1.00, 1.18, 1.30, 1.22])
    return factors[dow]


def _seasonal_factor(day_of_year: np.ndarray) -> np.ndarray:
    """Мягкая годовая сезонность: зима активнее лета (готовить лень / холодно)."""
    # косинус с минимумом летом (день ~200) и максимумом зимой
    phase = 2 * np.pi * (day_of_year - 200) / 365.0
    return 1.0 + 0.12 * np.cos(phase)


def generate_food_delivery(
    year: int = 2025,
    seed: int = 42,
    noise_level: float = 0.10,
) -> pd.DataFrame:
    """Сгенерировать почасовой ряд заказов food-delivery за год.

    Parameters
    ----------
    year : год для индекса дат.
    seed : зерно генератора случайных чисел.
    noise_level : относительный уровень мультипликативного шума (0..1).

    Returns
    -------
    DataFrame с DatetimeIndex (частота 1 час) и колонкой ``orders`` (int).
    Дополнительно — служебные колонки hour / dow / date для группировок.
    """
    rng = np.random.default_rng(seed)

    index = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    hours = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()

    base = daily_profile(hours)
    base = base * _weekday_factor(dow)
    base = base * _seasonal_factor(doy)

    # плавный годовой тренд роста бизнеса (+25% к концу года)
    t = np.arange(len(index)) / len(index)
    base = base * (1.0 + 0.25 * t)

    # мультипликативный шум — реалистичнее, чем аддитивный
    noise = rng.normal(1.0, noise_level, size=len(index))
    orders = base * np.clip(noise, 0.4, 1.8)

    orders = np.clip(np.round(orders), 0, None).astype(int)

    df = pd.DataFrame(
        {
            "orders": orders,
            "hour": hours,
            "dow": dow,
            "date": index.date,
        },
        index=index,
    )
    df.index.name = "timestamp"
    return df


def one_day(df: pd.DataFrame, date: str | None = None) -> pd.DataFrame:
    """Вырезать один день (по умолчанию — типичную субботу) для показа двух пиков."""
    if date is None:
        # берём первую субботу в данных
        saturdays = df[df["dow"] == 5]
        date = str(saturdays["date"].iloc[0])
    return df[df["date"].astype(str) == str(date)]
