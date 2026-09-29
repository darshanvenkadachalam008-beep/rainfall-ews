"""Unit tests for REWS preprocessing: regridding conservation, QC, features, and split leakage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from rews.preprocessing.dataset import SplitDefinition
from rews.preprocessing.features import (
    compute_cyclical_time_features,
    compute_imd_day_acc_so_far,
    compute_lag_features,
    compute_rolling_accumulations,
)
from rews.preprocessing.qc import filter_precipitation_qc, flag_spatial_spikes
from rews.preprocessing.regrid import conservative_regrid_2d, compute_lat_weights
from rews.utils.geo import generate_grid_coords


def test_regrid_mass_conservation() -> None:
    """Verifies that 2D precipitation regridding preserves total water volume (< 0.1% error)."""
    # Source grid (0.25 deg, e.g. GFS/ERA5)
    src_lons, src_lats = generate_grid_coords(78.0, 11.0, 83.0, 15.0, 0.25)
    # Synthetic rain field with non-uniform convective core
    mesh_lat, mesh_lon = np.meshgrid(src_lats, src_lons, indexing="ij")
    src_rain = 50.0 * np.exp(-((mesh_lat - 13.0) ** 2 + (mesh_lon - 80.5) ** 2) / 0.5)

    # Destination grid (0.1 deg pilot grid)
    dst_lons, dst_lats = generate_grid_coords(78.5, 11.5, 82.5, 14.5, 0.1)

    regridded_rain = conservative_regrid_2d(
        src_rain, src_lons, src_lats, dst_lons, dst_lats, is_flux=True
    )

    # Compute overlapping spatial volume
    overlap_lon_min = max(src_lons.min(), dst_lons.min())
    overlap_lon_max = min(src_lons.max(), dst_lons.max())
    overlap_lat_min = max(src_lats.min(), dst_lats.min())
    overlap_lat_max = min(src_lats.max(), dst_lats.max())

    src_mask = (
        (src_lats[:, None] >= overlap_lat_min)
        & (src_lats[:, None] <= overlap_lat_max)
        & (src_lons[None, :] >= overlap_lon_min)
        & (src_lons[None, :] <= overlap_lon_max)
    )
    dst_mask = (
        (dst_lats[:, None] >= overlap_lat_min)
        & (dst_lats[:, None] <= overlap_lat_max)
        & (dst_lons[None, :] >= overlap_lon_min)
        & (dst_lons[None, :] <= overlap_lon_max)
    )

    src_areas = np.tile(compute_lat_weights(src_lats, 0.25)[:, None], (1, len(src_lons)))
    dst_areas = np.tile(compute_lat_weights(dst_lats, 0.1)[:, None], (1, len(dst_lons)))

    v_src = np.sum(src_rain[src_mask] * src_areas[src_mask])
    v_dst = np.sum(regridded_rain[dst_mask] * dst_areas[dst_mask])

    rel_error = abs(v_dst - v_src) / v_src
    # Conservation requirement: error < 0.1% (0.001)
    assert rel_error < 0.001, f"Volume mismatch: src={v_src:.2f}, dst={v_dst:.2f}, err={rel_error:.4%}"


def test_qc_range_and_hot_pixel_filtering() -> None:
    """Verifies that physical ranges are enforced and isolated sensor hot pixels suppressed."""
    # 5x5 grid with background light rain and one isolated sensor spike
    arr = np.ones((5, 5), dtype=np.float32) * 2.0
    arr[2, 2] = 120.0  # isolated hot pixel (> 5x local median of 2.0)
    arr[0, 0] = -5.0   # negative artifact
    arr[4, 4] = 600.0  # absurd rate (> 300 mm/h)

    clean_arr, valid_mask = filter_precipitation_qc(arr, max_rate_mm_h=300.0, detect_spikes=True)

    # Negative value clamped/masked
    assert clean_arr[0, 0] == 0.0
    assert not valid_mask[0, 0]

    # Absurd value clamped/masked
    assert clean_arr[4, 4] == 0.0
    assert not valid_mask[4, 4]

    # Hot pixel suppressed by spatial spike detection
    assert clean_arr[2, 2] == 0.0
    assert not valid_mask[2, 2]

    # Normal rain preserved
    assert clean_arr[1, 1] == pytest.approx(2.0)
    assert valid_mask[1, 1]


def test_feature_rolling_accumulations() -> None:
    """Verifies rolling accumulations across 1h, 3h, 6h, 24h at 30-min cadence."""
    T = 60  # 30 hours of 30-min steps
    # Constant 10.0 mm/h rain rate
    rain_rates = np.full((T, 4, 4), 10.0, dtype=np.float32)

    accs = compute_rolling_accumulations(rain_rates, window_hours=[1, 3, 24], step_duration_hours=0.5)

    # At step 2 (1 hour elapsed): acc_1h = 2 * (10.0 * 0.5) = 10.0 mm
    np.testing.assert_allclose(accs["acc_1h"][2:], 10.0, rtol=1e-5)

    # At step 6 (3 hours elapsed): acc_3h = 6 * (10.0 * 0.5) = 30.0 mm
    np.testing.assert_allclose(accs["acc_3h"][6:], 30.0, rtol=1e-5)

    # At step 48 (24 hours elapsed): acc_24h = 48 * (10.0 * 0.5) = 240.0 mm
    np.testing.assert_allclose(accs["acc_24h"][48:], 240.0, rtol=1e-5)


def test_feature_lag_construction() -> None:
    """Verifies temporal lag frame alignment."""
    T = 10
    rain_rates = np.arange(T, dtype=np.float32)[:, None, None] * np.ones((1, 3, 3), dtype=np.float32)

    lags = compute_lag_features(rain_rates, lag_steps=[1, 2])

    # lag_30m (step 1 delay): lag_30m[1] == rain_rates[0]
    assert lags["lag_30m"][1, 0, 0] == rain_rates[0, 0, 0]
    assert lags["lag_30m"][5, 0, 0] == rain_rates[4, 0, 0]

    # lag_60m (step 2 delay): lag_60m[2] == rain_rates[0]
    assert lags["lag_60m"][2, 0, 0] == rain_rates[0, 0, 0]
    assert lags["lag_60m"][6, 0, 0] == rain_rates[4, 0, 0]


def test_imd_day_acc_so_far() -> None:
    """Verifies that accumulation so far resets at 03:00 UTC (08:30 IST) and grows monotonically."""
    timestamps = [
        pd.Timestamp("2023-12-03 02:00:00+0000"),  # Day D-1 (07:30 IST)
        pd.Timestamp("2023-12-03 02:30:00+0000"),  # Day D-1 (08:00 IST)
        pd.Timestamp("2023-12-03 03:00:00+0000"),  # Day D START (08:30 IST - reset!)
        pd.Timestamp("2023-12-03 03:30:00+0000"),  # Day D (09:00 IST)
    ]
    # Rain rate = 20 mm/h (10 mm per 30-min step)
    rain = np.full((4, 2, 2), 20.0, dtype=np.float32)

    acc_so_far, rem_hours = compute_imd_day_acc_so_far(rain, timestamps, step_duration_hours=0.5)

    # Step 0: 10 mm
    assert acc_so_far[0, 0, 0] == pytest.approx(10.0)
    # Step 1: 20 mm
    assert acc_so_far[1, 0, 0] == pytest.approx(20.0)
    # Step 2: 30 mm (completing Day D-1 window ending at 03:00 UTC)
    assert acc_so_far[2, 0, 0] == pytest.approx(30.0)
    # Step 3: RESET at 03:30 UTC (new IMD Day D begins) -> 10 mm
    assert acc_so_far[3, 0, 0] == pytest.approx(10.0)


def test_split_leakage_and_purge_gap() -> None:
    """Verifies programmatic zero-leakage enforcement and purge gap checks (Amendment D)."""
    # Train: Nov 2021
    train_times = [datetime(2021, 11, 1, 0, 0, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(24)]
    # Val: Oct 2023
    val_times = [datetime(2023, 10, 15, 0, 0, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(24)]
    # Test: Cyclone Michaung Dec 2023 (well over 75h after Oct 2023 val)
    test_times = [datetime(2023, 12, 3, 0, 0, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(24)]

    # Clean splits with wide purge gaps
    report = SplitDefinition.verify_no_leakage(train_times, val_times, test_times, min_purge_gap_hours=75.0)
    assert report["leakage_free"]
    assert report["overlap_samples"] == 0
    assert report["min_separation_val_test_hours"] > 75.0

    # Test failure case 1: direct timestamp overlap
    leaky_val = val_times + [test_times[0]]
    with pytest.raises(ValueError, match="Data leakage detected"):
        SplitDefinition.verify_no_leakage(train_times, leaky_val, test_times, min_purge_gap_hours=75.0)

    # Test failure case 2: purge gap violation (< 75 hours)
    too_close_val = [datetime(2023, 12, 1, 0, 0, tzinfo=timezone.utc)]  # only 48h before test
    with pytest.raises(ValueError, match="Purge gap violation"):
        SplitDefinition.verify_no_leakage(train_times, too_close_val, test_times, min_purge_gap_hours=75.0)


def test_actual_datacube_splits_and_leakage() -> None:
    """Verifies that the generated Zarr datacube contains train, val, and test splits with zero leakage."""
    import xarray as xr
    from rews.config import get_settings

    settings = get_settings()
    cube_path = settings.processed_dir / "datacube.zarr"
    assert cube_path.exists(), f"Datacube not found at {cube_path}"

    ds = xr.open_zarr(str(cube_path))
    splits = ds["split"].values.astype(str)
    times = pd.to_datetime(ds["time"].values)

    train_mask = (splits == "train")
    val_mask = (splits == "val")
    test_mask = np.char.startswith(splits, "test")

    train_times = times[train_mask]
    val_times = times[val_mask]
    test_times = times[test_mask]

    # Verify each partition is populated with real timestamps
    assert len(train_times) >= 300, f"Expected >= 300 train steps, got {len(train_times)}"
    assert len(val_times) >= 200, f"Expected >= 200 val steps, got {len(val_times)}"
    assert len(test_times) >= 700, f"Expected >= 700 test steps, got {len(test_times)}"

    # Check temporal span
    assert times.min() <= pd.Timestamp("2021-11-05")
    assert times.max() >= pd.Timestamp("2023-12-08")

    # Run strict zero-leakage verification on the actual datacube timestamps
    report = SplitDefinition.verify_no_leakage(
        train_times=list(train_times),
        val_times=list(val_times),
        test_times=list(test_times),
        min_purge_gap_hours=75.0,
    )

    assert report["leakage_free"] is True
    assert report["overlap_samples"] == 0
    assert report["min_separation_train_val_hours"] >= 75.0
    assert report["min_separation_val_test_hours"] >= 75.0

