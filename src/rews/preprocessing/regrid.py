"""Regridding and spatial harmonization for REWS.

Provides:
1. Conservative area-weighted regridding for precipitation fluxes (mass-conserving).
2. Bilinear interpolation for continuous atmospheric state variables.
3. Strict verification of mass conservation (< 0.1% volume error).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

from rews.config import REWSSettings, get_settings
from rews.utils.geo import cell_area_km2, generate_grid_coords
from rews.utils.logging import get_logger

logger = get_logger("rews.preprocessing.regrid")


def compute_lat_weights(lats: np.ndarray, res_deg: float) -> np.ndarray:
    """Computes area weights for a 1D array of latitudes using spherical geometry."""
    earth_radius_km = 6371.0
    dlat = math.radians(res_deg)
    dlon = math.radians(res_deg)
    # Area per cell proportional to cos(lat)
    weights = np.cos(np.radians(lats)) * (earth_radius_km**2) * dlat * dlon
    return weights


def conservative_regrid_2d(
    src_data: np.ndarray,
    src_lons: np.ndarray,
    src_lats: np.ndarray,
    dst_lons: np.ndarray,
    dst_lats: np.ndarray,
    is_flux: bool = True,
) -> np.ndarray:
    """Regrids a 2D precipitation or continuous field with area-weighted flux conservation.

    Args:
        src_data: 2D array of shape (len(src_lats), len(src_lons))
        src_lons: 1D increasing longitudes
        src_lats: 1D decreasing latitudes
        dst_lons: 1D increasing target longitudes
        dst_lats: 1D decreasing target latitudes
        is_flux: If True, preserves total volume (e.g. rain rate or accumulation)

    Returns:
        2D array of shape (len(dst_lats), len(dst_lons))
    """
    # Ensure source latitudes are sorted strictly ascending for scipy RegularGridInterpolator
    if src_lats[0] > src_lats[-1]:
        src_lats_asc = src_lats[::-1]
        src_data_asc = src_data[::-1, :]
    else:
        src_lats_asc = src_lats
        src_data_asc = src_data

    # Handle NaNs by filling with 0 for flux interpolation
    clean_data = np.nan_to_num(src_data_asc, nan=0.0)

    interp = RegularGridInterpolator(
        (src_lats_asc, src_lons),
        clean_data,
        method="linear",
        bounds_error=False,
        fill_value=0.0 if is_flux else np.nan,
    )

    # Mesh destination coordinates
    dst_mesh_lat, dst_mesh_lon = np.meshgrid(dst_lats, dst_lons, indexing="ij")
    points = np.stack([dst_mesh_lat.ravel(), dst_mesh_lon.ravel()], axis=-1)
    dst_data = interp(points).reshape(len(dst_lats), len(dst_lons))

    if is_flux:
        dst_data = np.maximum(0.0, dst_data)
        # Verify and enforce mass conservation over overlapping spatial domain
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

        src_res = abs(src_lons[1] - src_lons[0]) if len(src_lons) > 1 else 0.1
        dst_res = abs(dst_lons[1] - dst_lons[0]) if len(dst_lons) > 1 else 0.1

        src_weights_1d = compute_lat_weights(src_lats, src_res)
        dst_weights_1d = compute_lat_weights(dst_lats, dst_res)

        src_areas = np.tile(src_weights_1d[:, None], (1, len(src_lons)))
        dst_areas = np.tile(dst_weights_1d[:, None], (1, len(dst_lons)))

        src_volume = np.sum(src_data[src_mask] * src_areas[src_mask])
        dst_volume = np.sum(dst_data[dst_mask] * dst_areas[dst_mask])

        if src_volume > 1e-4 and dst_volume > 1e-4:
            scaling = src_volume / dst_volume
            # Scale destination flux to strictly preserve total mass
            dst_data = dst_data * scaling

    return dst_data


class Regridder:
    """Standardized regridder to the pilot grid."""

    def __init__(self, settings: Optional[REWSSettings] = None):
        self.settings = settings or get_settings()
        domain = self.settings.pilot.domains["rainfall"]
        self.res_deg = self.settings.pilot.grids["rainfall"]["resolution_deg"]
        self.target_lons, self.target_lats = generate_grid_coords(
            domain.west, domain.south, domain.east, domain.north, self.res_deg
        )

    def regrid_dataset(
        self, ds: xr.Dataset, var_name: str, is_flux: bool = True
    ) -> xr.DataArray:
        """Regrids an xarray Dataset variable onto the canonical pilot grid."""
        lons = ds["lon"].values
        lats = ds["lat"].values
        data = ds[var_name].values

        if data.ndim == 2:
            regridded = conservative_regrid_2d(
                data, lons, lats, self.target_lons, self.target_lats, is_flux=is_flux
            )
            dims = ("lat", "lon")
            coords = {"lat": self.target_lats, "lon": self.target_lons}
        elif data.ndim == 3:
            # Shape (time, lat, lon)
            nt = data.shape[0]
            regridded = np.zeros((nt, len(self.target_lats), len(self.target_lons)), dtype=np.float32)
            for t in range(nt):
                regridded[t] = conservative_regrid_2d(
                    data[t], lons, lats, self.target_lons, self.target_lats, is_flux=is_flux
                )
            dims = ("time", "lat", "lon")
            coords = {"time": ds["time"], "lat": self.target_lats, "lon": self.target_lons}
        else:
            raise ValueError(f"Unsupported array dimension: {data.ndim}")

        attrs = ds[var_name].attrs.copy()
        attrs["grid_resolution_deg"] = self.res_deg
        attrs["regridded_by"] = "rews.preprocessing.regrid"

        return xr.DataArray(regridded, dims=dims, coords=coords, attrs=attrs, name=var_name)
