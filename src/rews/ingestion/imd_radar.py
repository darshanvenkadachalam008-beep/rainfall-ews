"""IMD Doppler Weather Radar (DWR) ingestion and parsing adapter.

Provides multi-format auto-detection (NetCDF, HDF5, GeoTIFF) for radar files
placed in data/raw/radar/. If no files are available, gracefully notifies
the pipeline and documents the limitation without failure.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import rasterio
import xarray as xr

from rews.config import REWSSettings, get_settings
from rews.ingestion.base import BaseIngestionAdapter, IngestionProvenance
from rews.utils.logging import get_logger
from rews.utils.timeutils import to_utc

logger = get_logger("rews.ingestion.radar")


class IMDRadarAdapter(BaseIngestionAdapter):
    """Adapter for IMD Chennai / Sriharikota Doppler Weather Radar."""

    def __init__(self, settings: Optional[REWSSettings] = None):
        super().__init__("radar", settings)

    def detect_format(self, filepath: Path) -> str:
        """Auto-detects radar file format from extension and header signature."""
        ext = filepath.suffix.lower()
        if ext in [".nc", ".nc4", ".netcdf"]:
            return "netcdf"
        elif ext in [".h5", ".hdf5", ".he5"]:
            return "hdf5"
        elif ext in [".tif", ".tiff"]:
            return "geotiff"

        # Inspect magic bytes
        with open(filepath, "rb") as f:
            header = f.read(8)
            if header.startswith(b"\x89HDF"):
                return "hdf5"
            elif header.startswith(b"CDF"):
                return "netcdf"
            elif header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
                return "geotiff"

        return "unknown"

    def read_radar_file(
        self, filepath: Path, bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> xr.Dataset:
        """Parses radar file into standardized (lat, lon) rain-rate dataset."""
        fmt = self.detect_format(filepath)
        target_bbox = bbox or self.settings.pilot.domains["rainfall"].bounds_tuple

        if fmt == "geotiff":
            with rasterio.open(filepath) as src:
                data = src.read(1)
                bounds = src.bounds
                lons = np.linspace(bounds.left, bounds.right, data.shape[1])
                lats = np.linspace(bounds.top, bounds.bottom, data.shape[0])

            # Convert dBZ to rain rate R (mm/h) using standard Marshall-Palmer: Z = 200 * R^1.6
            # R = (10^(dBZ / 10) / 200) ^ (1 / 1.6)
            dbz = np.where(data > 0, data, np.nan)
            z_linear = 10.0 ** (dbz / 10.0)
            rain_rate = np.maximum(0.0, (z_linear / 200.0) ** (1.0 / 1.6))

            ds = xr.Dataset(
                data_vars={"rain_rate": (("lat", "lon"), np.nan_to_num(rain_rate, nan=0.0))},
                coords={"lat": lats, "lon": lons},
                attrs={"units": "mm/h", "source": filepath.name},
            )
            return ds

        elif fmt == "netcdf":
            ds = xr.open_dataset(filepath)
            # Standardize variable name to rain_rate
            var_candidates = ["rain_rate", "precipitation_rate", "SRI", "PAC", "DBZ"]
            found_var = next((v for v in var_candidates if v in ds), None)
            if found_var is None:
                # Use first 2D variable
                found_var = list(ds.data_vars.keys())[0]

            if found_var == "DBZ":
                dbz = ds[found_var].values
                z_linear = 10.0 ** (dbz / 10.0)
                rr = (z_linear / 200.0) ** (1.0 / 1.6)
                ds["rain_rate"] = (ds[found_var].dims, np.nan_to_num(rr, nan=0.0))
            else:
                ds["rain_rate"] = ds[found_var]

            return ds

        else:
            raise NotImplementedError(f"Unsupported radar format: {fmt} for file {filepath}")

    def fetch(
        self, date: Union[str, datetime], bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Optional[Path]:
        """Scans radar cache for available acquisitions matching date."""
        dt_utc = to_utc(date)
        date_str = dt_utc.strftime("%Y%m%d")
        candidates = list(self.cache_dir.glob(f"*{date_str}*"))

        if not candidates:
            logger.info(
                "No IMD radar files located in %s for %s. "
                "Operational constraint: System will execute without DWR inputs.",
                self.cache_dir,
                date_str,
            )
            return None

        logger.info("Found %d radar file(s) for %s in %s", len(candidates), date_str, self.cache_dir)
        return candidates[0]
