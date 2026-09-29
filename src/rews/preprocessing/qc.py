"""Quality control (QC) and anomaly filtering for meteorological datasets.

Provides:
1. Physical boundary and range validation.
2. Isolated spatial spike detection (sensor hot pixels).
3. Missingness masking and short-gap interpolation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.ndimage import median_filter
import xarray as xr

from rews.utils.logging import get_logger

logger = get_logger("rews.preprocessing.qc")


# Physical meteorological bounds
PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "precipitation": (0.0, 300.0),       # mm/h
    "precipitationCal": (0.0, 300.0),    # IMERG mm/h
    "rain_rate": (0.0, 300.0),           # radar mm/h
    "total_precipitation": (0.0, 0.3),   # ERA5 m/h
    "temperature_2m": (260.0, 330.0),    # Kelvin (~-13 C to 57 C)
    "mslp": (90000.0, 105000.0),         # Pascals (900 to 1050 hPa)
    "wind_speed": (0.0, 100.0),          # m/s
    "tir1_bt": (180.0, 340.0),           # INSAT Brightness Temperature (Kelvin)
}


def flag_spatial_spikes(data_2d: np.ndarray, threshold_val: float = 25.0, factor: float = 5.0) -> np.ndarray:
    """Identifies isolated spatial spikes (e.g. radar hot pixels, single-cell artifacts).

    A pixel is flagged as a spatial spike if its value exceeds `threshold_val`
    AND is greater than `factor` times the local 3x3 median.
    """
    clean = np.nan_to_num(data_2d, nan=0.0)
    local_median = median_filter(clean, size=3)
    # A single isolated pixel with 0 or low median around it
    spike_mask = (clean > threshold_val) & (clean > factor * np.maximum(local_median, 0.1))
    return spike_mask


def filter_precipitation_qc(
    arr: np.ndarray,
    max_rate_mm_h: float = 300.0,
    detect_spikes: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies strict quality control to precipitation arrays.

    Returns:
        (clean_array, valid_mask)
    """
    valid_mask = np.isfinite(arr) & (arr >= 0.0) & (arr <= max_rate_mm_h)
    clean_arr = np.where(valid_mask, arr, 0.0)

    if detect_spikes:
        if arr.ndim == 2:
            spikes = flag_spatial_spikes(clean_arr)
            valid_mask &= ~spikes
            clean_arr = np.where(spikes, 0.0, clean_arr)
        elif arr.ndim == 3:
            for t in range(arr.shape[0]):
                spikes = flag_spatial_spikes(clean_arr[t])
                valid_mask[t] &= ~spikes
                clean_arr[t] = np.where(spikes, 0.0, clean_arr[t])

    return clean_arr.astype(np.float32), valid_mask


def qc_dataset(ds: xr.Dataset, var_name: str) -> xr.Dataset:
    """Applies QC checks and attaches qc_flags bitmask to an xarray Dataset."""
    da = ds[var_name]
    raw_vals = da.values

    var_type = "precipitation" if "precip" in var_name.lower() or "rain" in var_name.lower() else var_name
    bounds = PHYSICAL_BOUNDS.get(var_type, (None, None))

    clean_vals = raw_vals.copy()
    valid_mask = np.isfinite(raw_vals)

    if bounds[0] is not None:
        valid_mask &= (raw_vals >= bounds[0])
        clean_vals = np.maximum(bounds[0], clean_vals)
    if bounds[1] is not None:
        valid_mask &= (raw_vals <= bounds[1])
        clean_vals = np.minimum(bounds[1], clean_vals)

    # For precipitation, run isolated spike detection
    if "precip" in var_name.lower() or "rain" in var_name.lower():
        clean_vals, p_mask = filter_precipitation_qc(clean_vals)
        valid_mask &= p_mask

    ds_qc = ds.copy()
    ds_qc[var_name] = (da.dims, clean_vals)
    ds_qc[f"{var_name}_qc_valid"] = (da.dims, valid_mask)
    ds_qc[var_name].attrs["qc_applied"] = True
    ds_qc[var_name].attrs["physical_bounds"] = list(bounds)

    return ds_qc
