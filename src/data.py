"""Synthetic food-delivery order data generation.

Hourly data across a year. Within each day there are two "hills":
  * a lunch peak around 13:00 (smaller);
  * a dinner peak around 19:30 (larger).

Plus weekly seasonality (weekends are busier), a smooth yearly growth trend,
mild monthly seasonality, and realistic noise. Everything is deterministic via
a seed, so the demo is reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _gaussian_peak(hours: np.ndarray, center: float, width: float, height: float) -> np.ndarray:
    """A single bell-shaped "hill" of activity during the day."""
    return height * np.exp(-0.5 * ((hours - center) / width) ** 2)


def daily_profile(hours: np.ndarray) -> np.ndarray:
    """Base order profile by hour of day (ignoring weekday/season).

    Returns the average order level for each hour [0..23].
    """
    lunch = _gaussian_peak(hours, center=11.5, width=1.1, height=45.0)   # lunch — smaller hill (~11-12)
    dinner = _gaussian_peak(hours, center=18.5, width=1.6, height=75.0)  # dinner — larger hill (~18-19)
    # small night/morning baseline so the line doesn't drop to zero
    base = 6.0 + 3.0 * np.exp(-0.5 * ((hours - 22.0) / 3.0) ** 2)
    return lunch + dinner + base


def _weekday_factor(dow: np.ndarray) -> np.ndarray:
    """Activity multiplier by day of week (0=Mon ... 6=Sun).

    Friday/Saturday/Sunday are noticeably busier — people order food more often.
    """
    factors = np.array([0.92, 0.90, 0.95, 1.00, 1.18, 1.30, 1.22])
    return factors[dow]


def _seasonal_factor(day_of_year: np.ndarray) -> np.ndarray:
    """Mild yearly seasonality: winter is busier than summer (cold / lazy to cook)."""
    # cosine peaking in winter (day ~0 / 365) and dipping in summer (day ~182)
    phase = 2 * np.pi * day_of_year / 365.0
    return 1.0 + 0.12 * np.cos(phase)


def generate_food_delivery(
    year: int = 2025,
    seed: int = 42,
    noise_level: float = 0.10,
) -> pd.DataFrame:
    """Generate an hourly food-delivery order series for one year.

    Parameters
    ----------
    year : year for the date index.
    seed : random number generator seed.
    noise_level : relative level of multiplicative noise (0..1).

    Returns
    -------
    DataFrame with a DatetimeIndex (hourly frequency) and an ``orders`` column
    (int). Plus helper columns hour / dow / date for grouping.
    """
    rng = np.random.default_rng(seed)

    index = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    hours = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()

    base = daily_profile(hours)
    base = base * _weekday_factor(dow)
    base = base * _seasonal_factor(doy)

    # smooth yearly business growth trend (+25% by year end)
    t = np.arange(len(index)) / len(index)
    base = base * (1.0 + 0.25 * t)

    # multiplicative noise — more realistic than additive
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


def weekday_profile(df: pd.DataFrame) -> np.ndarray:
    """Average total orders for each day of week (0=Mon ... 6=Sun).

    Computed from daily totals — a typical "how busy is each weekday" curve.
    """
    daily = df["orders"].resample("D").sum()
    prof = daily.groupby(daily.index.dayofweek).mean()
    return prof.reindex(range(7)).to_numpy(dtype=float)


def month_profile(df: pd.DataFrame) -> np.ndarray:
    """Average daily orders for each month of the year (index 0=Jan ... 11=Dec).

    Computed from daily totals — a typical "how busy is each month" curve.
    """
    daily = df["orders"].resample("D").sum()
    prof = daily.groupby(daily.index.month).mean()
    return prof.reindex(range(1, 13)).to_numpy(dtype=float)


def one_day(df: pd.DataFrame, date: str | None = None) -> pd.DataFrame:
    """Slice out a single day (a typical Saturday by default) to show the two peaks."""
    if date is None:
        # take the first Saturday in the data
        saturdays = df[df["dow"] == 5]
        date = str(saturdays["date"].iloc[0])
    return df[df["date"].astype(str) == str(date)]
