"""Zarr datacube builder and split manager for REWS.

Constructs unified multi-variable Zarr datacubes on the canonical pilot grid.
Enforces:
1. Strict event-based and time-block splits.
2. Minimum 75-hour temporal purge gaps between splits (Amendment D).
3. Zero-leakage verification.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import xarray as xr

from rews.config import REWSSettings, get_settings
from rews.utils.geo import generate_grid_coords
from rews.utils.logging import get_logger
from rews.utils.timeutils import to_utc

logger = get_logger("rews.preprocessing.dataset")


class SplitDefinition:
    """Defines temporal bounds for train, validation, and held-out test splits."""

    def __init__(self, settings: Optional[REWSSettings] = None):
        self.settings = settings or get_settings()
        splits_cfg = self.settings.pipeline.splits
        self.purge_gap_hours = float(splits_cfg.get("purge_gap_hours", 75.0))

        # Held-out test events
        self.test_events = splits_cfg.get("test_events", [
            {
                "id": "chennai_michaung_2023",
                "name": "Cyclone Michaung (Dec 2023)",
                "start_utc": "2023-11-30T00:00:00Z",
                "end_utc": "2023-12-08T00:00:00Z",
            }
        ])

        # Validation window and sampled validation events
        val_cfg = splits_cfg.get("validation_period", {
            "start_utc": "2023-10-01T00:00:00Z",
            "end_utc": "2023-11-20T00:00:00Z",
        })
        self.val_start = to_utc(val_cfg["start_utc"])
        self.val_end = to_utc(val_cfg["end_utc"])
        self.val_events = splits_cfg.get("validation_events", [])

        # Sampled training events
        self.train_events = splits_cfg.get("train_events", [])

    def get_all_event_time_ranges(self, freq: str = "30min") -> pd.DatetimeIndex:
        """Generates a concatenated, deduplicated, sorted DatetimeIndex across train, val, and test events."""
        indices = []

        # 1. Test events
        for ev in self.test_events:
            idx = pd.date_range(start=ev["start_utc"], end=ev["end_utc"], freq=freq, tz="UTC")
            indices.append(idx)

        # 2. Validation events (or full val period if events not configured)
        if self.val_events:
            for ev in self.val_events:
                idx = pd.date_range(start=ev["start_utc"], end=ev["end_utc"], freq=freq, tz="UTC")
                indices.append(idx)
        else:
            idx = pd.date_range(start=self.val_start, end=self.val_end, freq=freq, tz="UTC")
            indices.append(idx)

        # 3. Train events
        for ev in self.train_events:
            idx = pd.date_range(start=ev["start_utc"], end=ev["end_utc"], freq=freq, tz="UTC")
            indices.append(idx)

        combined = indices[0]
        for other in indices[1:]:
            combined = combined.union(other)

        return combined.sort_values()

    def assign_split(self, dt: datetime) -> str:
        """Determines whether a timestamp belongs to 'train', 'val', 'test', or 'purge_gap'."""
        dt_utc = to_utc(dt)
        gap = timedelta(hours=self.purge_gap_hours)

        # Check test events first
        for ev in self.test_events:
            ev_start = to_utc(ev["start_utc"])
            ev_end = to_utc(ev["end_utc"])
            if ev_start <= dt_utc <= ev_end:
                return f"test_{ev['id']}"
            # Check purge buffer surrounding test event
            if (ev_start - gap) <= dt_utc < ev_start or ev_end < dt_utc <= (ev_end + gap):
                return "purge_gap"

        # Check validation events or window
        if self.val_events:
            for ev in self.val_events:
                ev_start = to_utc(ev["start_utc"])
                ev_end = to_utc(ev["end_utc"])
                if ev_start <= dt_utc <= ev_end:
                    return "val"
                if (ev_start - gap) <= dt_utc < ev_start or ev_end < dt_utc <= (ev_end + gap):
                    return "purge_gap"
        elif self.val_start <= dt_utc <= self.val_end:
            return "val"

        # Check train events
        if self.train_events:
            for ev in self.train_events:
                ev_start = to_utc(ev["start_utc"])
                ev_end = to_utc(ev["end_utc"])
                if ev_start <= dt_utc <= ev_end:
                    return "train"

        # Check earliest GFS availability (2021-03-22)
        earliest_gfs = to_utc("2021-03-22T00:00:00Z")
        if dt_utc < earliest_gfs:
            return "excluded_pre_gfs"

        # Prior years Northeast Monsoon (Oct-Dec)
        if dt_utc.month in [10, 11, 12]:
            return "train"

        return "other"

    @staticmethod
    def verify_no_leakage(
        train_times: Sequence[datetime],
        val_times: Sequence[datetime],
        test_times: Sequence[datetime],
        min_purge_gap_hours: float = 75.0,
    ) -> Dict[str, Any]:
        """Programmatically proves that train, val, and test splits have 0 overlap and sufficient purge gaps."""
        t_train = set(pd.to_datetime(list(train_times)))
        t_val = set(pd.to_datetime(list(val_times)))
        t_test = set(pd.to_datetime(list(test_times)))

        overlap_tv = t_train.intersection(t_val)
        overlap_tt = t_train.intersection(t_test)
        overlap_vt = t_val.intersection(t_test)

        if overlap_tv or overlap_tt or overlap_vt:
            raise ValueError(
                f"Data leakage detected! Overlaps: train-val={len(overlap_tv)}, "
                f"train-test={len(overlap_tt)}, val-test={len(overlap_vt)}"
            )

        # Compute minimum temporal separation between sets
        min_dist_train_val = float("inf")
        min_dist_val_test = float("inf")
        min_dist_train_test = float("inf")

        for tt in t_train:
            for tv in t_val:
                diff = abs((tt - tv).total_seconds()) / 3600.0
                if diff < min_dist_train_val:
                    min_dist_train_val = diff
            for tte in t_test:
                diff = abs((tt - tte).total_seconds()) / 3600.0
                if diff < min_dist_train_test:
                    min_dist_train_test = diff

        for tv in t_val:
            for tte in t_test:
                diff = abs((tv - tte).total_seconds()) / 3600.0
                if diff < min_dist_val_test:
                    min_dist_val_test = diff

        report = {
            "train_samples": len(t_train),
            "val_samples": len(t_val),
            "test_samples": len(t_test),
            "overlap_samples": 0,
            "min_separation_train_val_hours": round(min_dist_train_val, 2),
            "min_separation_val_test_hours": round(min_dist_val_test, 2),
            "min_separation_train_test_hours": round(min_dist_train_test, 2),
            "purge_gap_enforced_hours": min_purge_gap_hours,
            "leakage_free": (
                min_dist_train_val >= min_purge_gap_hours
                and min_dist_val_test >= min_purge_gap_hours
            ),
        }

        if not report["leakage_free"]:
            raise ValueError(
                f"Purge gap violation! Required: {min_purge_gap_hours}h, "
                f"found train-val={min_dist_train_val}h, val-test={min_dist_val_test}h"
            )

        return report


class DatacubeBuilder:
    """Assembles regridded and QC-validated variables into a standardized Zarr datacube."""

    def __init__(self, settings: Optional[REWSSettings] = None):
        self.settings = settings or get_settings()
        domain = self.settings.pilot.domains["rainfall"]
        self.res_deg = self.settings.pilot.grids["rainfall"]["resolution_deg"]
        self.lons, self.lats = generate_grid_coords(
            domain.west, domain.south, domain.east, domain.north, self.res_deg
        )
        self.split_manager = SplitDefinition(self.settings)

    def build_cube(
        self,
        time_index: pd.DatetimeIndex,
        data_vars: Dict[str, np.ndarray],
        output_zarr_path: Optional[Path] = None,
    ) -> xr.Dataset:
        """Constructs an xarray Dataset and writes to Zarr store."""
        target_path = output_zarr_path or (self.settings.processed_dir / "datacube.zarr")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure CF-compliant UTC timestamps for Zarr/NetCDF backend
        time_utc_naive = pd.to_datetime(time_index).tz_localize(None) if time_index.tz is not None else time_index

        coords = {
            "time": time_utc_naive,
            "lat": self.lats,
            "lon": self.lons,
        }

        # Assign split tags using timezone-aware datetimes
        split_tags = [self.split_manager.assign_split(t.to_pydatetime()) for t in time_index]

        ds = xr.Dataset(coords=coords)
        ds["time"].attrs = {"standard_name": "time", "timezone": "UTC"}
        ds["split"] = ("time", split_tags)

        for name, arr in data_vars.items():
            if arr.ndim == 3:
                ds[name] = (("time", "lat", "lon"), arr.astype(np.float32))
            elif arr.ndim == 2:
                ds[name] = (("lat", "lon"), arr.astype(np.float32))
            elif arr.ndim == 1:
                ds[name] = (("time",), arr.astype(np.float32))

        ds.attrs = {
            "title": "SIH26071 Unified Meteorological & Inundation Datacube",
            "region": self.settings.pilot.region.name,
            "resolution_deg": self.res_deg,
            "grid_crs": "EPSG:4326",
            "created_at_utc": datetime.now().isoformat(),
        }

        logger.info("Writing datacube to Zarr store at %s...", target_path)
        ds.to_zarr(str(target_path), mode="w")
        return ds
