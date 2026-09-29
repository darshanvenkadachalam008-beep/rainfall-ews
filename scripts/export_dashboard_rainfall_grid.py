"""Exports compact rainfall and screening layers for the interactive dashboard."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from rews.config import get_settings


def export_dashboard_data():
    settings = get_settings()
    out_dir = Path("outputs") / "dashboard_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    datacube_path = Path(settings.processed_dir) / "datacube.zarr"
    ds = xr.open_zarr(datacube_path)
    time_series = pd.to_datetime(ds.time.values, utc=True)
    precip = ds.precipitation_rate.values
    lats = [round(float(v), 2) for v in ds.lat.values]
    lons = [round(float(v), 2) for v in ds.lon.values]

    # Load alert timeline
    timeline_path = Path("outputs/alerts/michaung_alert_timeline.json")
    with open(timeline_path, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)["timeline"]

    steps = []
    for item in timeline_data:
        dt = pd.to_datetime(item["timestamp_utc"])
        idx = np.where(time_series == dt)[0]
        if len(idx) > 0:
            frame = precip[idx[0]]
            # Crop to Greater Chennai region (12.0 - 14.5 N, 79.5 - 81.5 E)
            # Find indices
            lat_arr = np.array(lats)
            lon_arr = np.array(lons)
            lat_sub = np.where((lat_arr >= 12.0) & (lat_arr <= 14.5))[0]
            lon_sub = np.where((lon_arr >= 79.5) & (lon_arr <= 81.5))[0]
            
            sub_frame = frame[np.ix_(lat_sub, lon_sub)]
            # Round to 1 decimal place to keep JSON compact
            grid_vals = np.round(sub_frame, 1).tolist()
            steps.append({
                "timestamp": item["timestamp_utc"],
                "alert_level": item["alert_level"],
                "alert_label": item["alert_label"],
                "cma_peak_rate": item["cma_peak_rate_mm_h"],
                "acc_3h": item["acc_3h_forecast_mm"],
                "p_extremely_heavy": item["calibrated_probabilities"]["extremely_heavy_50mm_h"],
                "p_very_heavy": item["calibrated_probabilities"]["very_heavy_25mm_h"],
                "p_heavy": item["calibrated_probabilities"]["heavy_15_6mm_h"],
                "action": item["recommended_action"],
                "grid": grid_vals,
            })

    output_payload = {
        "bounds": {
            "min_lat": float(lat_arr[lat_sub].min()),
            "max_lat": float(lat_arr[lat_sub].max()),
            "min_lon": float(lon_arr[lon_sub].min()),
            "max_lon": float(lon_arr[lon_sub].max()),
        },
        "lats": [float(lat_arr[i]) for i in lat_sub],
        "lons": [float(lon_arr[j]) for j in lon_sub],
        "steps": steps,
    }

    out_file = out_dir / "michaung_rainfall_steps.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f)
    print(f"Exported {len(steps)} rainfall animation frames to {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    export_dashboard_data()
