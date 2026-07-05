"""Simulation of the two Chronos v2 modes: zero-shot and fine-tuned.

We do NOT run the real neural network (it's heavy and needs a GPU / weight
downloads). Instead we honestly model the *principle* behind the difference
between the modes:

* **Zero-shot** — the pretrained model knows the "typical shape of a day"
  (quiet morning, a lunch hill, a bigger dinner hill), but has not seen YOUR
  specific business. So it captures the daily shape well but misses the weekly
  specifics (exactly how much your weekends spike).

* **Fine-tuned** — the same model, further trained on your history. It has
  already learned the "hour × day-of-week" profile, so it nails both weekdays
  and weekends.

This is exactly the effect fine-tuning is done for: adapting a general
pretrained "prior" to a specific series. The quality metric is WAPE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error.

    WAPE = sum(|y - ŷ|) / sum(|y|).

    Unlike MAPE, it is robust to zeros and small values, which is why it's
    popular in retail and logistics. Returned as a percentage.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return 100.0 * np.abs(y_true - y_pred).sum() / denom


@dataclass
class ForecastResult:
    """Forecast result over the test window."""

    timestamps: pd.DatetimeIndex
    y_true: np.ndarray
    zero_shot: np.ndarray
    fine_tuned: np.ndarray
    wape_zero_shot: float
    wape_fine_tuned: float

    @property
    def improvement_pct(self) -> float:
        """By what percentage fine-tuning reduced the error relative to zero-shot."""
        if self.wape_zero_shot == 0:
            return 0.0
        return 100.0 * (self.wape_zero_shot - self.wape_fine_tuned) / self.wape_zero_shot


def make_forecasts(
    df: pd.DataFrame,
    horizon_days: int = 14,
    ft_noise: float = 0.05,
    seed: int = 7,
) -> ForecastResult:
    """Build zero-shot and fine-tuned forecasts over the last ``horizon_days``.

    The logic is deliberately simple and interpretable:

    * the profile is learned on HISTORY (everything before the test window);
    * zero-shot ≈ an "hour-of-day" profile (a general prior for the day's shape);
    * fine-tuned ≈ an "hour × day-of-week" profile + a mild trend (learned your
      series).
    """
    rng = np.random.default_rng(seed)

    work = df.copy()
    split = len(work) - horizon_days * 24
    if split <= 0:
        raise ValueError("Horizon is too large for the available amount of data")

    train = work.iloc[:split]
    test = work.iloc[split:]

    # --- Zero-shot: day shape only (hour), no day of week ---
    prof_hour = train.groupby("hour")["orders"].mean()
    zero_shot = test["hour"].map(prof_hour).to_numpy(dtype=float)

    # --- Fine-tuned: hour × day-of-week (learned the weekly seasonality) ---
    prof_hd = train.groupby(["dow", "hour"])["orders"].mean()
    ft = np.array(
        [prof_hd.loc[(int(r.dow), int(r.hour))] for r in test.itertuples()],
        dtype=float,
    )

    # account for the mild growth trend: scale to the level of the last week of history
    recent_level = train["orders"].iloc[-7 * 24 :].mean()
    hist_level = train["orders"].mean()
    trend_scale = recent_level / max(hist_level, 1e-6)
    zero_shot = zero_shot  # zero-shot doesn't "feel" the trend
    ft = ft * trend_scale

    # a bit of irreducible noise for fine-tuned (nothing is perfect)
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
