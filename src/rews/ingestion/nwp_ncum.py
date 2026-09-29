"""NCMRWF NCUM (Unified Model) ingestion adapter.

Parses Indian National Centre for Medium Range Weather Forecasting (NCMRWF)
NCUM model outputs from local GRIB/NetCDF files placed in data/raw/nwp_ncum/.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import xarray as xr

from rews.config import REWSSettings, get_settings
from rews.ingestion.base import BaseIngestionAdapter, IngestionProvenance
from rews.utils.logging import get_logger
from rews.utils.timeutils import to_utc

logger = get_logger("rews.ingestion.ncum")


class NCUMAdapter(BaseIngestionAdapter):
    """Adapter for NCMRWF NCUM regional/global forecast files."""

    def __init__(self, settings: Optional[REWSSettings] = None):
        super().__init__("nwp_ncum", settings)

    def fetch(
        self, date: Union[str, datetime], lead_hour: int = 24
    ) -> Optional[Path]:
        """Looks for NCUM GRIB or NetCDF files in cache directory."""
        dt_utc = to_utc(date)
        date_tag = dt_utc.strftime("%Y%m%d")
        candidates = list(self.cache_dir.glob(f"*{date_tag}*f{lead_hour:02d}*.*")) + list(
            self.cache_dir.glob(f"*{date_tag}*.*")
        )

        if not candidates:
            logger.info(
                "No NCUM file found for %s in %s. NCUM is optional; pipeline will use GFS.",
                date_tag,
                self.cache_dir,
            )
            return None

        filepath = candidates[0]
        logger.info("Found NCUM file: %s", filepath.name)
        return filepath
