"""Audit precipitation statistics across all splits in datacube.zarr."""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import xarray as xr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rews.config import get_settings
from rews.preprocessing.dataset import SplitDefinition


def audit_datacube(zarr_path: str = "data/processed/datacube.zarr") -> None:
    settings = get_settings()

    print("=" * 80)
    print(f"DATACUBE PRECIPITATION AUDIT: {zarr_path}")
    print("=" * 80)

    try:
        ds = xr.open_zarr(zarr_path)
    except Exception as e:
        print(f"Failed to open Zarr datacube at {zarr_path}: {e}")
        return

    print(f"Global Datacube Dimensions: {dict(ds.sizes)}")
    print(f"Variables: {list(ds.data_vars.keys())}")
    time_series = pd.to_datetime(ds.time.values, utc=True)
    print(f"Full Temporal Range: {time_series.min().isoformat()} to {time_series.max().isoformat()} (T={len(time_series)})")
    print("-" * 80)

    precip = ds.precipitation_rate.values  # shape (T, H, W)
    valid = np.isfinite(precip)
    clean_precip = np.where(valid, precip, 0.0)

    splits_to_check = []

    # Test events
    for ev in settings.pipeline.splits.get("test_events", []):
        splits_to_check.append(("test", ev["id"], ev["name"], ev["start_utc"], ev["end_utc"]))

    # Validation events
    for ev in settings.pipeline.splits.get("validation_events", []):
        splits_to_check.append(("val", ev["id"], ev["name"], ev["start_utc"], ev["end_utc"]))

    # Train events
    for ev in settings.pipeline.splits.get("train_events", []):
        splits_to_check.append(("train", ev["id"], ev["name"], ev["start_utc"], ev["end_utc"]))

    print(f"{'Split / Event ID':<35} | {'Steps':<6} | {'Min (mm/h)':<10} | {'Mean (mm/h)':<11} | {'Max (mm/h)':<10} | {'Non-Zero %':<10}")
    print("-" * 92)

    total_split_stats = {}

    for partition, ev_id, ev_name, start_utc, end_utc in splits_to_check:
        t_start = pd.to_datetime(start_utc)
        t_end = pd.to_datetime(end_utc)
        mask = (time_series >= t_start) & (time_series <= t_end)
        n_steps = int(mask.sum())

        if n_steps == 0:
            print(f"{ev_id:<35} | {0:<6} | {'N/A':<10} | {'N/A':<11} | {'N/A':<10} | {'N/A':<10}")
            continue

        p_slice = clean_precip[mask, :, :]
        p_min = float(p_slice.min())
        p_mean = float(p_slice.mean())
        p_max = float(p_slice.max())
        nz_pct = float((p_slice > 0.01).mean() * 100.0)

        label = f"[{partition.upper()}] {ev_id}"
        print(f"{label:<35} | {n_steps:<6} | {p_min:<10.3f} | {p_mean:<11.4f} | {p_max:<10.2f} | {nz_pct:<9.2f}%")

        if partition not in total_split_stats:
            total_split_stats[partition] = []
        total_split_stats[partition].append(p_slice)

    print("-" * 92)
    print("AGGREGATED PARTITIONS:")
    for partition in ["train", "val", "test"]:
        if partition in total_split_stats:
            arr = np.concatenate(total_split_stats[partition], axis=0)
            n_steps = arr.shape[0]
            p_min = float(arr.min())
            p_mean = float(arr.mean())
            p_max = float(arr.max())
            nz_pct = float((arr > 0.01).mean() * 100.0)
            label = f"TOTAL {partition.upper()}"
            print(f"{label:<35} | {n_steps:<6} | {p_min:<10.3f} | {p_mean:<11.4f} | {p_max:<10.2f} | {nz_pct:<9.2f}%")

    print("=" * 80)


if __name__ == "__main__":
    z_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/datacube.zarr"
    audit_datacube(z_path)
