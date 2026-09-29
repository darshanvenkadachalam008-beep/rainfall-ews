"""Geospatial helper utilities for REWS.

Handles bounding boxes, CRS conversions (WGS84 EPSG:4326 to UTM 44N EPSG:32644),
grid coordinate generation, and spatial bounds checking.
"""

from __future__ import annotations

import math
from typing import Tuple
import numpy as np
from pyproj import Transformer


_WGS84_TO_UTM44N = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
_UTM44N_TO_WGS84 = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)


def wgs84_to_utm44n(lon: float, lat: float) -> Tuple[float, float]:
    """Converts geographic coordinates (lon, lat in deg) to UTM 44N (easting, northing in m)."""
    return _WGS84_TO_UTM44N.transform(lon, lat)


def utm44n_to_wgs84(easting: float, northing: float) -> Tuple[float, float]:
    """Converts UTM 44N (easting, northing in m) to geographic coordinates (lon, lat in deg)."""
    return _UTM44N_TO_WGS84.transform(easting, northing)


def transform_bbox(
    west: float, south: float, east: float, north: float, src_crs: str, dst_crs: str
) -> Tuple[float, float, float, float]:
    """Transforms a bounding box from src_crs to dst_crs."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs = [west, east, west, east]
    ys = [south, south, north, north]
    tx, ty = transformer.transform(xs, ys)
    return min(tx), min(ty), max(tx), max(ty)


def generate_grid_coords(
    west: float, south: float, east: float, north: float, resolution: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates 1D lon and lat coordinate arrays centered on regular intervals.

    Returns:
        (lons, lats) where lons is increasing and lats is decreasing (standard raster convention).
    """
    lons = np.arange(west, east + resolution * 0.5, resolution)
    lats = np.arange(north, south - resolution * 0.5, -resolution)
    return lons, lats


def cell_area_km2(lat_deg: float, res_deg: float) -> float:
    """Computes approximate ground area of a geographic grid cell in km^2.

    Crucial for flux and mass conservation checks when regridding precipitation.
    """
    earth_radius_km = 6371.0
    dlat = math.radians(res_deg)
    dlon = math.radians(res_deg)
    lat_rad = math.radians(lat_deg)
    # Area = R^2 * cos(lat) * dlat * dlon
    return (earth_radius_km**2) * math.cos(lat_rad) * dlat * dlon
