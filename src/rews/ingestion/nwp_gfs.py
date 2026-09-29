"""NOAA GFS ingestion adapter using Herbie and AWS Open Data.

Includes robust GFS APCP precipitation bucket de-accumulation to convert
raw accumulation fields into discrete step rainfall rates.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie

from rews.config import REWSSettings, get_settings
from rews.ingestion.base import BaseIngestionAdapter, IngestionProvenance
from rews.utils.logging import get_logger

logger = get_logger("rews.ingestion.gfs")


def deaccumulate_gfs_apcp(
    forecast_hours: Sequence[int],
    cumulative_apcp: Union[Sequence[np.ndarray], np.ndarray],
) -> np.ndarray:
    """De-accumulates GFS APCP (Total Precipitation) buckets into discrete step increments.

    GFS v16 accumulates precipitation over rolling 6-hour windows:
      - f01: [0, 1] accumulation -> step rain = apcp(1)
      - f02: [0, 2] accumulation -> step rain = apcp(2) - apcp(1)
      ...
      - f06: [0, 6] accumulation -> step rain = apcp(6) - apcp(5)
      - f07: [6, 7] accumulation (bucket reset!) -> step rain = apcp(7)
      - f08: [6, 8] accumulation -> step rain = apcp(8) - apcp(7)

    This function detects bucket resets either from the forecast hour schedule
    or dynamically when apcp[i] < apcp[i-1], and prevents negative precipitation.

    Args:
        forecast_hours: Monotonically increasing sequence of forecast lead hours (e.g. [1, 2, 3, ...])
        cumulative_apcp: Array or sequence of arrays of shape (N_steps, ...) containing raw APCP.

    Returns:
        np.ndarray of same shape containing step precipitation (mm).
    """
    if len(forecast_hours) == 0:
        return np.array([])

    raw_arr = np.asarray(cumulative_apcp, dtype=np.float32)
    if raw_arr.shape[0] != len(forecast_hours):
        raise ValueError(
            f"Dimension mismatch: {len(forecast_hours)} forecast hours but {raw_arr.shape[0]} arrays"
        )

    deacc = np.zeros_like(raw_arr)
    # First step is always the accumulation from cycle start or bucket start
    deacc[0] = np.maximum(0.0, raw_arr[0])

    for i in range(1, len(forecast_hours)):
        prev_h = forecast_hours[i - 1]
        curr_h = forecast_hours[i]

        # GFS APCP accumulation buckets span (0, 6], (6, 12], (12, 18], etc.
        # Bucket index is ((hour - 1) // 6). A reset occurs whenever the bucket index changes,
        # or when the cumulative accumulation drops.
        prev_bucket = (prev_h - 1) // 6
        curr_bucket = (curr_h - 1) // 6
        is_bucket_reset = (curr_bucket != prev_bucket)

        diff = raw_arr[i] - raw_arr[i - 1]

        if is_bucket_reset:
            # New bucket: raw_arr[i] represents accumulation since the bucket reset hour
            step_val = np.maximum(0.0, raw_arr[i])
        else:
            # Within same bucket: difference, clamping minor numerical noise to 0
            step_val = np.maximum(0.0, diff)

        deacc[i] = step_val

    return deacc


class GFSAdapter(BaseIngestionAdapter):
    """Adapter for downloading and parsing NOAA GFS forecast cycles via Herbie."""

    def __init__(self, settings: Optional[REWSSettings] = None):
        super().__init__("nwp_gfs", settings)
        self.earliest_date = self.settings.pipeline.gfs.get("earliest_archive_date", "2021-03-22")

    def fetch_cycle_lead(
        self,
        date: Union[str, datetime],
        lead_hour: int,
        search_string: str = ":(APCP|PRATE|TMP:2 m|UGRD:10 m|VGRD:10 m|MSLET):",
    ) -> Path:
        """Fetches a specific GFS cycle and lead hour from AWS Open Data archive.

        Args:
            date: Cycle run date and hour, e.g. "2023-12-03 00:00"
            lead_hour: Forecast lead hour (e.g. 3, 6, 12)
            search_string: GRIB index regex filter for byte-range subsetting.
        """
        dest_file = self.cache_dir / f"gfs_{pd.to_datetime(date).strftime('%Y%m%d_%H%M')}_f{lead_hour:03d}.grib2"
        if dest_file.exists():
            logger.info("Found cached GFS file: %s", dest_file.name)
            return dest_file

        logger.info("Querying GFS via Herbie for %s f%03d...", date, lead_hour)
        H = Herbie(
            date,
            model="gfs",
            product="pgrb2.0p25",
            fxx=lead_hour,
            save_dir=self.cache_dir,
            verbose=False,
        )

        downloaded = H.download(search_string, save_dir=self.cache_dir)
        if downloaded is None or not Path(downloaded).exists():
            raise FileNotFoundError(f"Herbie could not download GFS {date} f{lead_hour}")

        Path(downloaded).rename(dest_file)

        # Record provenance
        bbox = self.settings.pilot.domains["rainfall"].bounds_tuple
        prov = IngestionProvenance(
            source="noaa_gfs_aws",
            target_path=dest_file,
            time_range=(str(date), f"lead_{lead_hour}h"),
            bbox=bbox,
            url_or_endpoint="aws_s3_noaa_gfs",
            extra_metadata={"lead_hour": lead_hour, "filter": search_string},
        )
        prov.finalize()
        return dest_file

    def fetch(self, date: Union[str, datetime], lead_hours: Sequence[int]) -> List[Path]:
        """Fetches an entire forecast cycle across requested lead hours."""
        paths = []
        for fxx in lead_hours:
            p = self.fetch_cycle_lead(date, fxx)
            paths.append(p)
        return paths
