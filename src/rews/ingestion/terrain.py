"""Terrain, Land Cover, and OSM Infrastructure ingestion adapter.

Provides:
1. Copernicus DEM GLO-30 (30 m surface elevation) and FABDEM notes.
2. ESA WorldCover 10 m land cover raster.
3. OpenStreetMap road graph and waterway network via osmnx.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio
from rasterio.windows import from_bounds
import requests

from rews.config import REWSSettings, get_settings
from rews.ingestion.base import BaseIngestionAdapter, IngestionProvenance
from rews.utils.logging import get_logger

logger = get_logger("rews.ingestion.terrain")


class TerrainAdapter(BaseIngestionAdapter):
    """Adapter for topographic and surface data."""

    # Copernicus DEM GLO-30 public AWS S3 endpoint
    # S3 bucket: copernicus-dem-30m / HTTPS access
    COP_DEM_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"

    def __init__(self, settings: Optional[REWSSettings] = None):
        super().__init__("terrain", settings)
        self.dem_dir = self.cache_dir / "dem"
        self.lc_dir = self.cache_dir / "landcover"
        self.osm_dir = self.cache_dir / "osm"
        for d in [self.dem_dir, self.lc_dir, self.osm_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def fetch_dem_glo30(
        self, bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Path:
        """Fetches Copernicus DEM GLO-30 30m elevation tile for Chennai.

        Chennai pilot domain (~13 N, 80 E) is covered by tile:
        Copernicus_DSM_COG_10_N13_00_E080_00_DEM
        """
        target_bbox = bbox or self.settings.pilot.domains["inundation"].bounds_tuple
        tile_name = "Copernicus_DSM_COG_10_N13_00_E080_00_DEM"
        dest_tif = self.dem_dir / f"{tile_name}.tif"

        if dest_tif.exists():
            logger.info("Found cached Copernicus DEM tile: %s", dest_tif.name)
            return dest_tif

        url = f"{self.COP_DEM_BASE_URL}/{tile_name}/{tile_name}.tif"
        logger.info("Downloading Copernicus GLO-30 tile from %s...", url)
        self.download_file_resumable(url, dest_tif)

        prov = IngestionProvenance(
            source="copernicus_glo30_dem",
            target_path=dest_tif,
            time_range=("2020-01-01", "present"),
            bbox=target_bbox,
            url_or_endpoint=url,
            extra_metadata={"resolution_m": 30, "model_type": "DSM"},
        )
        prov.finalize()
        return dest_tif

    def fetch_osm_network(
        self, bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Path:
        """Fetches and caches the OpenStreetMap road network for emergency routing."""
        target_bbox = bbox or self.settings.pilot.domains["inundation"].bounds_tuple
        west, south, east, north = target_bbox

        dest_gpkg = self.osm_dir / "chennai_osm_roads.gpkg"
        if dest_gpkg.exists():
            logger.info("Found cached OSM road network: %s", dest_gpkg.name)
            return dest_gpkg

        logger.info("Querying OSM drivable street network for bbox (S=%.2f, N=%.2f, W=%.2f, E=%.2f)...", south, north, west, east)
        # OSMnx graph_from_bbox expects (north, south, east, west) in older versions,
        # or bbox=(west, south, east, north) in v1.9+
        try:
            G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive")
        except TypeError:
            G = ox.graph_from_bbox(north, south, east, west, network_type="drive")

        nodes, edges = ox.graph_to_gdfs(G)
        edges.to_file(dest_gpkg, layer="edges", driver="GPKG")
        nodes.to_file(dest_gpkg, layer="nodes", driver="GPKG")

        prov = IngestionProvenance(
            source="openstreetmap_roads",
            target_path=dest_gpkg,
            time_range=("historical", "current"),
            bbox=target_bbox,
            url_or_endpoint="overpass_api",
            extra_metadata={"node_count": len(nodes), "edge_count": len(edges)},
        )
        prov.finalize()
        return dest_gpkg

    def fetch(self, *args: Any, **kwargs: Any) -> Path:
        return self.fetch_dem_glo30()
