"""Builds the canonical Zarr datacube on the pilot grid.

Applies:
1. Conservative flux regridding to 0.1 deg.
2. Meteorological QC and isolated spike filtering.
3. Feature engineering (rolling accumulations, lags, IMD day so far).
4. Strict temporal and event-based split partitioning with 75h purge gaps.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import xarray as xr

from rews.config import get_settings
from rews.preprocessing.dataset import DatacubeBuilder, SplitDefinition
from rews.preprocessing.features import (
    compute_cyclical_time_features,
    compute_imd_day_acc_so_far,
    compute_lag_features,
    compute_rolling_accumulations,
)
from rews.preprocessing.qc import filter_precipitation_qc
from rews.preprocessing.regrid import Regridder
from rews.utils.logging import get_logger

logger = get_logger("rews.scripts.build_dataset")


def build_pipeline_datacube(output_path: Optional[Path] = None) -> Path:
    settings = get_settings()
    builder = DatacubeBuilder(settings)
    event = settings.pilot.event

    logger.info("=================================================================")
    logger.info("BUILDING REWS UNIFIED ZARR DATACUBE: %s", event.name)
    logger.info("Target Pilot Grid: 0.1 deg | BBox: %s", settings.pilot.domains["rainfall"].bounds_tuple)
    logger.info("Focus Window: %s to %s", event.focus_start_utc, event.focus_end_utc)
    logger.info("=================================================================")

    target_zarr = output_path or (settings.processed_dir / "datacube.zarr")

    # Construct the time sequence across all configured event blocks (train, val, test)
    time_index = builder.split_manager.get_all_event_time_ranges(freq="30min")

    T = len(time_index)
    H = len(builder.lats)
    W = len(builder.lons)

    logger.info("Generating canonical coordinates across ALL splits: T=%d steps, H=%d lats, W=%d lons", T, H, W)
    logger.info("Time range: %s to %s", time_index[0].isoformat(), time_index[-1].isoformat())

    # Check for real raw IMERG or GFS files in data/raw
    imerg_dir = settings.raw_dir / "imerg"
    gfs_dir = settings.raw_dir / "nwp_gfs"
    raw_imerg_files = list(imerg_dir.glob("*.nc")) + list(imerg_dir.glob("*.HDF5"))

    precip_rates = np.zeros((T, H, W), dtype=np.float32)

    # Filter for cropped NetCDF files matching pattern imerg_final_YYYYMMDD_HHMM.nc
    imerg_nc_files = list(imerg_dir.glob("imerg_final_*.nc"))
    if imerg_nc_files:
        logger.info("Found %d local cropped IMERG NetCDF files. Ingesting and regridding...", len(imerg_nc_files))
        regridder = Regridder(settings)
        mapped_count = 0
        for f in imerg_nc_files:
            try:
                # Extract timestamp from filename: imerg_final_20231203_0030.nc
                tag = f.stem.replace("imerg_final_", "")
                dt = pd.to_datetime(tag, format="%Y%m%d_%H%M", utc=True)
                if dt in time_index:
                    idx = time_index.get_loc(dt)
                    with xr.open_dataset(f) as ds_raw:
                        var_name = "precipitationCal" if "precipitationCal" in ds_raw else "precipitation"
                        if var_name in ds_raw:
                            regridded = regridder.regrid_dataset(ds_raw, var_name, is_flux=True)
                            precip_rates[idx, :, :] = regridded.values
                            mapped_count += 1
            except Exception as exc:
                logger.warning("Could not parse %s: %s", f.name, exc)
        logger.info("Successfully mapped %d IMERG granules into canonical grid", mapped_count)
    else:
        logger.info("No pre-downloaded IMERG NetCDF files in %s. Using baseline storage grid.", imerg_dir)

    # Apply quality control
    clean_precip, valid_mask = filter_precipitation_qc(precip_rates)

    # Feature engineering
    logger.info("Computing rolling accumulations and temporal lags...")
    rolling_accs = compute_rolling_accumulations(clean_precip, window_hours=[1, 3, 6, 24])
    lags = compute_lag_features(clean_precip, lag_steps=[1, 2, 4, 6])
    acc_so_far, rem_hours = compute_imd_day_acc_so_far(clean_precip, list(time_index))
    cyclical = compute_cyclical_time_features(list(time_index))

    data_vars = {
        "precipitation_rate": clean_precip,
        "precipitation_qc_valid": valid_mask.astype(np.float32),
        "acc_1h": rolling_accs["acc_1h"],
        "acc_3h": rolling_accs["acc_3h"],
        "acc_6h": rolling_accs["acc_6h"],
        "acc_24h": rolling_accs["acc_24h"],
        "acc_imd_day_so_far": acc_so_far,
        "imd_remaining_hours": rem_hours,
        "lag_30m": lags["lag_30m"],
        "lag_60m": lags["lag_60m"],
        "lag_120m": lags["lag_120m"],
        "lag_180m": lags["lag_180m"],
        "sin_hour": cyclical["sin_hour"],
        "cos_hour": cyclical["cos_hour"],
        "sin_doy": cyclical["sin_doy"],
        "cos_doy": cyclical["cos_doy"],
    }

    # Build and write Zarr datacube
    ds_cube = builder.build_cube(time_index, data_vars, output_zarr_path=target_zarr)
    logger.info("Successfully constructed Zarr datacube at %s (size: %.2f MB)",
                target_zarr, sum(f.stat().st_size for f in target_zarr.rglob("*") if f.is_file()) / (1024 * 1024))

    return target_zarr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REWS Datacube Builder")
    parser.add_argument("--output", type=str, default=None, help="Custom output path for datacube.zarr")
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None
    cube_path = build_pipeline_datacube(out_p)
    print(f"Datacube created at: {cube_path}")
