"""NASA GPM IMERG ingestion adapter.

Supports half-hourly IMERG V07B (Early, Late, Final).
Nowcast input: IMERG Early / Late
Verification truth: IMERG Final
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import pandas as pd
import xarray as xr

from rews.config import REWSSettings, get_settings
from rews.ingestion.base import BaseIngestionAdapter, IngestionProvenance
from rews.utils.logging import get_logger
from rews.utils.timeutils import to_utc

logger = get_logger("rews.ingestion.imerg")
_NETCDF_LOCK = threading.Lock()


class IMERGAdapter(BaseIngestionAdapter):
    """Adapter for NASA GPM IMERG V07B (Early, Late, Final)."""

    def __init__(self, settings: Optional[REWSSettings] = None):
        super().__init__("imerg", settings)

    def get_remote_url(self, dt: datetime, run_type: str = "final") -> str:
        """Constructs GES DISC URL for IMERG half-hourly V07B."""
        dt_utc = to_utc(dt)
        year = dt_utc.strftime("%Y")
        day_of_year = dt_utc.strftime("%j")
        date_str = dt_utc.strftime("%Y%m%d")
        start_min = dt_utc.hour * 60 + (0 if dt_utc.minute < 30 else 30)
        end_min = start_min + 29
        start_str = f"S{dt_utc.strftime('%H')}{'00' if dt_utc.minute < 30 else '30'}00"
        end_str = f"E{dt_utc.strftime('%H')}{'29' if dt_utc.minute < 30 else '59'}59"
        min_of_day_str = f"{start_min:04d}"

        run_type_lower = run_type.lower()
        if run_type_lower == "final":
            subpath = "GPM_3IMERGHH.07"
            prefix = "3B-HHR.MS.MRG.3IMERG"
        elif run_type_lower == "late":
            subpath = "GPM_3IMERGHHL.07"
            prefix = "3B-HHR-L.MS.MRG.3IMERG"
        elif run_type_lower == "early":
            subpath = "GPM_3IMERGHHE.07"
            prefix = "3B-HHR-E.MS.MRG.3IMERG"
        else:
            raise ValueError(f"Unknown IMERG run type: {run_type}")

        filename = f"{prefix}.{date_str}-{start_str}-{end_str}.{min_of_day_str}.V07B.HDF5"
        url = f"https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/{subpath}/{year}/{day_of_year}/{filename}"
        return url

    def get_opendap_url(
        self,
        dt: datetime,
        bbox: Tuple[float, float, float, float],
        run_type: str = "final",
    ) -> str:
        """Constructs an OPeNDAP spatial subset URL for the IMERG granule."""
        dt_utc = to_utc(dt)
        year = dt_utc.strftime("%Y")
        day_of_year = dt_utc.strftime("%j")
        date_str = dt_utc.strftime("%Y%m%d")
        start_str = dt_utc.strftime("%H%M%S")
        end_dt = dt_utc + timedelta(minutes=29, seconds=59)
        end_str = end_dt.strftime("%H%M%S")
        total_minutes = dt_utc.hour * 60 + dt_utc.minute
        min_of_day_str = f"{total_minutes:04d}"

        run_type_lower = run_type.lower()
        if run_type_lower == "final":
            subpath = "GPM_3IMERGHH.07"
            prefix = "3B-HHR.MS.MRG.3IMERG"
        elif run_type_lower == "late":
            subpath = "GPM_3IMERGHL.07"
            prefix = "3B-HHR-L.MS.MRG.3IMERG"
        elif run_type_lower == "early":
            subpath = "GPM_3IMERGHHE.07"
            prefix = "3B-HHR-E.MS.MRG.3IMERG"
        else:
            raise ValueError(f"Unknown IMERG run type: {run_type}")

        west, south, east, north = bbox
        i_start = max(0, int(np.floor((west + 180.0) * 10)))
        i_end = min(3599, int(np.ceil((east + 180.0) * 10)) - 1)
        j_start = max(0, int(np.floor((south + 90.0) * 10)))
        j_end = min(1799, int(np.ceil((north + 90.0) * 10)) - 1)

        filename = f"{prefix}.{date_str}-S{start_str}-E{end_str}.{min_of_day_str}.V07B.HDF5"
        base_url = f"https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/{subpath}/{year}/{day_of_year}/{filename}.nc4"
        query = f"precipitation[0:0][{i_start}:{i_end}][{j_start}:{j_end}],lon[{i_start}:{i_end}],lat[{j_start}:{j_end}]"
        return f"{base_url}?{query}"

    def fetch_half_hourly(
        self,
        dt: Union[str, datetime],
        run_type: str = "final",
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Path:
        """Fetches and subsets an IMERG half-hourly HDF5 granule."""
        dt_utc = to_utc(dt)
        run_type_clean = run_type.lower()
        target_bbox = bbox or self.settings.pilot.domains["rainfall"].bounds_tuple

        date_tag = dt_utc.strftime("%Y%m%d_%H%M")
        dest_nc = self.cache_dir / f"imerg_{run_type_clean}_{date_tag}.nc"
        if dest_nc.exists() and dest_nc.stat().st_size > 1000:
            return dest_nc

        username = self.settings.earthdata_username
        password = self.settings.earthdata_password

        # Fast path: Try OPeNDAP spatial subsetting (downloads ~25 KB instead of 7.3 MB)
        if username and password:
            try:
                auth = (username, password)
                opendap_url = self.get_opendap_url(dt_utc, target_bbox, run_type=run_type_clean)
                resp = self.session.get(opendap_url, auth=auth, timeout=30)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    import netCDF4 as nc
                    with _NETCDF_LOCK:
                        with nc.Dataset(f"in_mem_{date_tag}", memory=resp.content) as ds_op:
                            lons = ds_op.variables["lon"][:].astype(np.float32)
                            lats = ds_op.variables["lat"][:].astype(np.float32)
                            precip = ds_op.variables["precipitation"][0, :, :].astype(np.float32)
                            precip = np.transpose(precip)
                            clean_precip = np.where(precip >= 0.0, precip, 0.0)
                            cropped_ds = xr.Dataset(
                                data_vars={"precipitationCal": (("lat", "lon"), clean_precip)},
                                coords={"lat": lats, "lon": lons},
                                attrs={"units": "mm/h", "source_file": f"imerg_{run_type_clean}_{date_tag}.HDF5"},
                            )
                            cropped_ds.to_netcdf(dest_nc)
                    prov = IngestionProvenance(
                        source=f"gpm_imerg_{run_type_clean}_v07b_opendap",
                        target_path=dest_nc,
                        time_range=(dt_utc.isoformat(), dt_utc.isoformat()),
                        bbox=target_bbox,
                        url_or_endpoint=opendap_url,
                        extra_metadata={"run_type": run_type_clean, "method": "opendap_subset"},
                    )
                    prov.finalize()
                    return dest_nc
            except Exception as e:
                logger.warning("OPeNDAP subsetting failed for %s, falling back to full HDF5: %s", date_tag, e)

        raw_hdf5 = self.cache_dir / f"imerg_raw_{run_type_clean}_{date_tag}.HDF5"
        if not raw_hdf5.exists():
            url = self.get_remote_url(dt_utc, run_type=run_type_clean)
            if not username or not password:
                # Check if file exists manually placed in data/raw/imerg
                manual_candidates = list(self.cache_dir.glob(f"*{date_tag}*.HDF5")) + list(
                    self.cache_dir.glob(f"*{date_tag}*.nc")
                )
                if manual_candidates:
                    return manual_candidates[0]
                raise PermissionError(
                    f"NASA Earthdata credentials not provided in .env (EARTHDATA_USERNAME / EARTHDATA_PASSWORD), "
                    f"and no local file matching {date_tag} found in {self.cache_dir}"
                )

            auth = (username, password)
            self.download_file_resumable(url, raw_hdf5, auth=auth)

        # Crop to bounding box to minimize storage and load times
        with _NETCDF_LOCK:
            cropped_ds = self.crop_imerg_hdf5(raw_hdf5, target_bbox)
            cropped_ds.to_netcdf(dest_nc)
        if raw_hdf5.exists():
            try:
                raw_hdf5.unlink()
            except Exception:
                pass

        prov = IngestionProvenance(
            source=f"gpm_imerg_{run_type_clean}_v07b",
            target_path=dest_nc,
            time_range=(dt_utc.isoformat(), dt_utc.isoformat()),
            bbox=target_bbox,
            url_or_endpoint=self.get_remote_url(dt_utc, run_type=run_type_clean),
            extra_metadata={"run_type": run_type_clean, "raw_hdf5": str(raw_hdf5.name)},
        )
        prov.finalize()
        return dest_nc

    @staticmethod
    def crop_imerg_hdf5(
        hdf5_path: Path, bbox: Tuple[float, float, float, float]
    ) -> xr.Dataset:
        """Extracts and crops IMERG precipitationCal array from raw HDF5."""
        west, south, east, north = bbox
        with h5py.File(hdf5_path, "r") as hf:
            lons = hf["Grid/lon"][:]
            lats = hf["Grid/lat"][:]
            if "Grid/precipitation" in hf:
                precip = hf["Grid/precipitation"][0, :, :]
            elif "Grid/precipitationCal" in hf:
                precip = hf["Grid/precipitationCal"][0, :, :]
            else:
                raise KeyError("No precipitation dataset found in IMERG HDF5")

        # Note: IMERG native array is shape (3600, 1800) corresponding to (lon, lat)
        lon_mask = (lons >= west) & (lons <= east)
        lat_mask = (lats >= south) & (lats <= north)

        sub_lons = lons[lon_mask]
        sub_lats = lats[lat_mask]
        # Transpose to (lat, lon) standard
        sub_precip = precip[np.ix_(lon_mask, lat_mask)].T
        # Mask out fill values (-9999.9)
        clean_precip = np.where(sub_precip >= 0.0, sub_precip, 0.0)

        ds = xr.Dataset(
            data_vars={"precipitationCal": (("lat", "lon"), clean_precip.astype(np.float32))},
            coords={"lat": sub_lats, "lon": sub_lons},
            attrs={"units": "mm/h", "source_file": hdf5_path.name},
        )
        return ds

    def fetch(
        self,
        dt: Union[str, datetime],
        run_type: str = "final",
        bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs: Any,
    ) -> Path:
        """Standard fetch interface implementing BaseIngestionAdapter."""
        return self.fetch_half_hourly(dt, run_type=run_type, bbox=bbox)

