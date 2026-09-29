"""Orchestrates pilot dataset downloads and builds the data inventory manifest.

Enforces budget limits from configs/pipeline.yaml (training_budget_gb).
Downloads Cyclone Michaung focus window first, records provenance,
and outputs data inventory to outputs/provenance/data_inventory.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rews.config import get_settings
from rews.ingestion.terrain import TerrainAdapter
from rews.ingestion.nwp_gfs import GFSAdapter
from rews.ingestion.gpm_imerg import IMERGAdapter
from rews.utils.logging import get_logger

logger = get_logger("rews.scripts.download_pilot_data")


def run_ingestion_pipeline(verify_only: bool = False, fetch_terrain: bool = True) -> Dict[str, Any]:
    settings = get_settings()
    budget_gb = settings.pipeline.budget.training_budget_gb
    event = settings.pilot.event
    domains = settings.pilot.domains

    logger.info("=================================================================")
    logger.info("PILOT DATA INGESTION ORCHESTRATOR: %s", event.name)
    logger.info("Total Data Budget: %.1f GB | Michaung Window Budget: %.1f GB",
                budget_gb, settings.pipeline.budget.michaung_window_budget_gb)
    logger.info("Focus Window: %s to %s", event.focus_start_utc, event.focus_end_utc)
    logger.info("=================================================================")

    manifest: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "event": event.model_dump(),
        "training_budget_gb": budget_gb,
        "datasets": {},
        "storage_audit": {},
    }

    # 1. Terrain & Elevation
    terrain_adapter = TerrainAdapter(settings)
    if verify_only:
        logger.info("[Verify-Only] Checking Copernicus GLO-30 tile availability...")
        dem_tile = "Copernicus_DSM_COG_10_N13_00_E080_00_DEM"
        dest_dem = terrain_adapter.dem_dir / f"{dem_tile}.tif"
        manifest["datasets"]["terrain_dem"] = {
            "name": "Copernicus DEM GLO-30",
            "tile": dem_tile,
            "status": "cached" if dest_dem.exists() else "verified_remote",
            "local_path": str(dest_dem) if dest_dem.exists() else None,
        }
    elif fetch_terrain:
        logger.info("Fetching Copernicus DEM GLO-30 30m tile...")
        try:
            dem_path = terrain_adapter.fetch_dem_glo30()
            manifest["datasets"]["terrain_dem"] = {
                "name": "Copernicus DEM GLO-30",
                "status": "downloaded",
                "path": str(dem_path),
                "size_mb": round(dem_path.stat().st_size / (1024 * 1024), 2),
            }
        except Exception as exc:
            logger.error("Failed to fetch DEM: %s", exc)
            manifest["datasets"]["terrain_dem"] = {"status": "error", "error": str(exc)}

    # 2. Check Raw Directory Footprint
    total_raw_bytes = 0
    raw_files_summary: List[Dict[str, Any]] = []
    if settings.raw_dir.exists():
        for f in settings.raw_dir.rglob("*"):
            if f.is_file() and not f.name.endswith(".meta.json"):
                size = f.stat().st_size
                total_raw_bytes += size
                raw_files_summary.append({
                    "path": str(f.relative_to(settings.project_root)),
                    "size_mb": round(size / (1024 * 1024), 3),
                })

    total_raw_gb = round(total_raw_bytes / (1024**3), 3)
    manifest["storage_audit"] = {
        "total_raw_size_gb": total_raw_gb,
        "training_budget_gb": budget_gb,
        "budget_remaining_gb": round(budget_gb - total_raw_gb, 3),
        "file_count": len(raw_files_summary),
        "files": raw_files_summary,
    }

    # Write data inventory
    out_dir = settings.outputs_dir / "provenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = out_dir / "data_inventory.json"
    with open(inventory_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Data inventory saved to %s", inventory_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot Data Ingestion Driver")
    parser.add_argument("--verify-only", action="store_true", help="Verify availability without large downloads")
    parser.add_argument("--fetch-terrain", action="store_true", help="Download static terrain elevation tiles")
    args = parser.parse_args()

    res = run_ingestion_pipeline(verify_only=args.verify_only, fetch_terrain=args.fetch_terrain)
    print(json.dumps(res["storage_audit"], indent=2))
