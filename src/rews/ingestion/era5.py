"""ECMWF ERA5 reanalysis ingestion adapter.

Fetches hourly single-level parameters (total precipitation, 2m temp, 10m wind,
MSLP, total column water vapour) subsetted to the pilot domain via cdsapi.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import cdsapi

from rews.config import REWSSettings, get_settings
from rews.ingestion.base import BaseIngestionAdapter, IngestionProvenance
from rews.utils.logging import get_logger
from rews.utils.timeutils import to_utc

logger = get_logger("rews.ingestion.era5")


class ERA5Adapter(BaseIngestionAdapter):
    """Adapter for downloading ERA5 reanalysis single levels via cdsapi."""

    VARIABLES = [
        "total_precipitation",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "total_column_water_vapour",
    ]

    def __init__(self, settings: Optional[REWSSettings] = None):
        super().__init__("era5", settings)

    def fetch_day(
        self,
        date: Union[str, datetime],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Path:
        """Downloads all 24 hours of ERA5 single levels for a specified calendar date."""
        dt_utc = to_utc(date)
        date_str = dt_utc.strftime("%Y-%m-%d")
        dest_file = self.cache_dir / f"era5_single_levels_{dt_utc.strftime('%Y%m%d')}.nc"

        if dest_file.exists():
            logger.info("Found cached ERA5 file: %s", dest_file.name)
            return dest_file

        target_bbox = bbox or self.settings.pilot.domains["rainfall"].bounds_tuple
        west, south, east, north = target_bbox
        # CDS API expects area in [North, West, South, East]
        area = [north, west, south, east]

        key = self.settings.cdsapi_key
        url = self.settings.cdsapi_url or "https://cds.climate.copernicus.eu/api"

        if not key:
            # Check for manually placed NetCDF file
            candidates = list(self.cache_dir.glob(f"*{dt_utc.strftime('%Y%m%d')}*.nc"))
            if candidates:
                return candidates[0]
            raise PermissionError(
                f"Copernicus CDS API key (CDSAPI_KEY) not found in environment or .env, "
                f"and no local file for {date_str} in {self.cache_dir}."
            )

        client = cdsapi.Client(url=url, key=key, quiet=True)
        request_params = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": self.VARIABLES,
            "year": dt_utc.strftime("%Y"),
            "month": dt_utc.strftime("%m"),
            "day": dt_utc.strftime("%d"),
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
        }

        logger.info("Submitting CDS API request for ERA5 on %s...", date_str)
        client.retrieve("reanalysis-era5-single-levels", request_params, str(dest_file))

        prov = IngestionProvenance(
            source="ecmwf_era5_reanalysis",
            target_path=dest_file,
            time_range=(f"{date_str}T00:00:00Z", f"{date_str}T23:00:00Z"),
            bbox=target_bbox,
            url_or_endpoint=url,
            extra_metadata={"variables": self.VARIABLES, "area": area},
        )
        prov.finalize()
        return dest_file

    def fetch(self, date: Union[str, datetime]) -> Path:
        return self.fetch_day(date)
