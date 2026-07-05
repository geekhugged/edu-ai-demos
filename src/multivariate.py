"""Simulation of multivariate time series with covariates.

Unlike the first version, Chronos v2 can account not only for the order series
itself but also for external features (covariates): weather, promotions,
weekends. Here we generate related series and show how covariates "explain" the
behavior of the target order series.
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
    beta_holiday: float = 0.55,
) -> pd.DataFrame:
    """Generate a target order series + four covariates.

    Covariates:
      * temperature — temperature (°C), seasonal-diurnal;
      * rain — rain intensity (0..1), random showers;
      * promo — promotion flag (0/1), a few campaigns;
      * holiday — special-day flag (0/1), a couple of public holidays that cause
        a strong demand spike (people stay in and order food).

    The target series is built as a base daily profile amplified/dampened by the
    covariates through the beta_* coefficients. In the UI the user can change the
    betas and watch how the relationship changes.
    """
    from .data import daily_profile  # local import to avoid cycles

    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-06-01", periods=days * 24, freq="h")
    hours = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()

    # --- covariates ---
    temperature = (
        18
        + 8 * np.sin(2 * np.pi * (hours - 15) / 24)  # warmer during the day
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

    # special days / public holidays — a couple of standout days with a big spike
    holiday = np.zeros(len(index))
    holiday_days = rng.choice(np.arange(days), size=2, replace=False)
    for day in holiday_days:
        holiday[day * 24 : (day + 1) * 24] = 1.0

    weekend = (dow >= 5).astype(float)

    # --- target series ---
    base = daily_profile(hours) * (1.0 + 0.20 * weekend)
    # normalize covariates to ~[-1..1] for a fair beta influence
    temp_z = (temperature - temperature.mean()) / temperature.std()

    signal = base * (
        1.0
        + beta_promo * promo
        + beta_rain * rain
        + beta_temp * temp_z
        + beta_holiday * holiday
    )
    noise = rng.normal(1.0, 0.06, len(index))
    orders = np.clip(np.round(signal * noise), 0, None)

    return pd.DataFrame(
        {
            "orders": orders,
            "temperature": temperature.round(1),
            "rain": rain.round(2),
            "promo": promo,
            "holiday": holiday,
            "weekend": weekend,
        },
        index=index,
    )


def reconstruct(
    df: pd.DataFrame,
    use_promo: bool,
    use_rain: bool,
    use_temp: bool,
    use_holiday: bool,
    beta_promo: float,
    beta_rain: float,
    beta_temp: float,
    beta_holiday: float,
) -> tuple[np.ndarray, float]:
    """A minimal "model" that tries to reconstruct orders from the covariates.

    Returns the reconstructed series and the WAPE. It shows: the more relevant
    covariates we include, the closer the reconstruction is to reality.
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
    if use_holiday:
        factor = factor + beta_holiday * df["holiday"].to_numpy()

    recon = base * factor
    return recon, wape(df["orders"].to_numpy(), recon)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Correlations of the target series with the covariates (for the heatmap)."""
    cols = ["orders", "temperature", "rain", "promo", "holiday", "weekend"]
    return df[cols].corr()
