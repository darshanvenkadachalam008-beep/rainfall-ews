"""Feature engineering for rainfall nowcasting and NWP post-processing.

Extracts:
1. Temporal lags (t-30m, t-60m, t-120m, t-180m).
2. Rolling window accumulations (1h, 3h, 6h, 24h).
3. IMD day accumulation so far (from 08:30 IST to current timestamp, for Amendment G).
4. Cyclical diurnal and seasonal harmonics.
5. Topographic gradients and elevation features.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr

from rews.utils.logging import get_logger
from rews.utils.timeutils import get_imd_accumulation_window, to_utc

logger = get_logger("rews.preprocessing.features")


def compute_lag_features(
    precip_array: np.ndarray,
    lag_steps: List[int] = [1, 2, 4, 6],  # 30m, 60m, 120m, 180m at 30-min cadence
) -> Dict[str, np.ndarray]:
    """Computes past lag frames for temporal nowcasting models.

    Args:
        precip_array: Array of shape (T, H, W)
        lag_steps: List of integer step delays

    Returns:
        Dict mapping lag name (e.g. 'lag_30m') to array of shape (T, H, W)
    """
    T, H, W = precip_array.shape
    features: Dict[str, np.ndarray] = {}

    for step in lag_steps:
        lag_arr = np.zeros_like(precip_array)
        if step < T:
            lag_arr[step:] = precip_array[:-step]
        # Pad initial steps with first valid frame
        for i in range(min(step, T)):
            lag_arr[i] = precip_array[0]
        lag_minutes = step * 30
        features[f"lag_{lag_minutes}m"] = lag_arr

    return features


def compute_rolling_accumulations(
    precip_rates_mm_h: np.ndarray,
    window_hours: List[int] = [1, 3, 6, 24],
    step_duration_hours: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Computes backward rolling accumulations from rain rate fields.

    Step precipitation = rain_rate * step_duration_hours.
    Window accumulation is the sum of step precipitation within the past N hours.
    """
    T, H, W = precip_rates_mm_h.shape
    step_precip = precip_rates_mm_h * step_duration_hours
    accumulations: Dict[str, np.ndarray] = {}

    for w_h in window_hours:
        window_steps = max(1, int(round(w_h / step_duration_hours)))
        acc_arr = np.zeros_like(precip_rates_mm_h)

        # Vectorized cumsum for fast rolling calculation along time axis
        cumsum = np.cumsum(step_precip, axis=0)
        # For steps >= window_steps: cumsum[t] - cumsum[t - window_steps]
        acc_arr[:window_steps] = cumsum[:window_steps]
        if T > window_steps:
            acc_arr[window_steps:] = cumsum[window_steps:] - cumsum[:-window_steps]

        accumulations[f"acc_{w_h}h"] = acc_arr

    return accumulations


def compute_imd_day_acc_so_far(
    precip_rates_mm_h: np.ndarray,
    timestamps: List[pd.Timestamp],
    step_duration_hours: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes observed accumulation so far and remaining hours in the active IMD day (Amendment G).

    The IMD day runs from 08:30 IST (03:00 UTC) on Day D to 08:30 IST on Day D+1.
    """
    T, H, W = precip_rates_mm_h.shape
    acc_so_far = np.zeros_like(precip_rates_mm_h)
    remaining_hours = np.zeros(T, dtype=np.float32)

    current_window_start = None
    rolling_acc = np.zeros((H, W), dtype=np.float32)

    for t, ts in enumerate(timestamps):
        dt = to_utc(ts)
        win_start, win_end = get_imd_accumulation_window(dt)

        if current_window_start != win_start:
            # New IMD day started; reset accumulator
            current_window_start = win_start
            rolling_acc = np.zeros((H, W), dtype=np.float32)

        # Add this step's precipitation
        step_mm = precip_rates_mm_h[t] * step_duration_hours
        rolling_acc += step_mm
        acc_so_far[t] = rolling_acc

        rem_sec = max(0.0, (win_end - dt).total_seconds())
        remaining_hours[t] = rem_sec / 3600.0

    return acc_so_far, remaining_hours


def compute_cyclical_time_features(timestamps: List[pd.Timestamp]) -> Dict[str, np.ndarray]:
    """Generates continuous sine and cosine diurnal and seasonal harmonics."""
    hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps], dtype=np.float32)
    doys = np.array([ts.dayofyear for ts in timestamps], dtype=np.float32)

    return {
        "sin_hour": np.sin(2 * np.pi * hours / 24.0),
        "cos_hour": np.cos(2 * np.pi * hours / 24.0),
        "sin_doy": np.sin(2 * np.pi * doys / 365.25),
        "cos_doy": np.cos(2 * np.pi * doys / 365.25),
    }


def compute_topographic_gradients(elevation_2d: np.ndarray, res_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Computes east-west and north-south elevation gradients (slope)."""
    # 1 deg latitude approx 111 km = 111,000 m
    dy_m = res_deg * 111000.0
    dx_m = res_deg * 111000.0 * np.cos(np.radians(13.0))

    grad_y, grad_x = np.gradient(elevation_2d, dy_m, dx_m)
    slope = np.sqrt(grad_x**2 + grad_y**2)
    return slope.astype(np.float32), np.arctan2(-grad_y, -grad_x).astype(np.float32)
