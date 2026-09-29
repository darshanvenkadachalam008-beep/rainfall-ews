"""Parallel downloader for Cyclone Michaung IMERG half-hourly window (2023-11-30 to 2023-12-08).

Downloads raw HDF5 granules from NASA GES DISC using Earthdata credentials,
immediately crops to the pilot domain bounding box, saves cropped NetCDFs,
and deletes the intermediate 7MB raw files to minimize disk storage.
"""

from __future__ import annotations

import concurrent.futures
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rews.config import get_settings
from rews.ingestion.gpm_imerg import IMERGAdapter
from rews.utils.logging import get_logger

logger = get_logger("rews.scripts.download_michaung")


def download_single_step(dt: datetime, adapter: IMERGAdapter) -> str:
    date_tag = dt.strftime("%Y%m%d_%H%M")
    dest_nc = adapter.cache_dir / f"imerg_final_{date_tag}.nc"
    if dest_nc.exists() and dest_nc.stat().st_size > 1000:
        return f"cached:{date_tag}"
    try:
        adapter.fetch_half_hourly(dt, run_type="final")
        return f"success:{date_tag}"
    except Exception as exc:
        logger.error("Failed to download %s: %s", date_tag, exc)
        return f"failed:{date_tag}:{exc}"


def main() -> None:
    settings = get_settings(reload=True)
    adapter = IMERGAdapter(settings)

    event = settings.pilot.event
    start_utc = pd.to_datetime(event.window_start_utc)
    end_utc = pd.to_datetime(event.window_end_utc)
    dates = pd.date_range(start_utc, end_utc, freq="30min", tz="UTC")

    logger.info("=================================================================")
    logger.info("DOWNLOADING CYCLONE MICHAUNG IMERG WINDOW")
    logger.info("Time Range: %s to %s (%d steps)", start_utc.isoformat(), end_utc.isoformat(), len(dates))
    logger.info("BBox: %s", settings.pilot.domains["rainfall"].bounds_tuple)
    logger.info("=================================================================")

    t0 = time.time()
    results = []
    max_workers = 8

    # Prioritize storm peak (Dec 03 - Dec 05) first, then remaining lead-up and recession
    peak_mask = (dates >= pd.Timestamp("2023-12-03T00:00:00Z")) & (dates <= pd.Timestamp("2023-12-05T23:59:59Z"))
    ordered_dates = list(dates[peak_mask]) + list(dates[~peak_mask])

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dt = {
            executor.submit(download_single_step, dt.to_pydatetime(), adapter): dt
            for dt in ordered_dates
        }
        done_count = 0
        for future in concurrent.futures.as_completed(future_to_dt):
            res = future.result()
            results.append(res)
            done_count += 1
            if done_count % 25 == 0 or done_count == len(dates):
                elapsed = time.time() - t0
                logger.info("Progress: %d / %d completed (%.1fs elapsed)", done_count, len(dates), elapsed)

    success_count = sum(1 for r in results if r.startswith("success") or r.startswith("cached"))
    failed_count = sum(1 for r in results if r.startswith("failed"))

    logger.info("=================================================================")
    logger.info("DOWNLOAD SUMMARY:")
    logger.info("Total steps: %d | Successful/Cached: %d | Failed: %d", len(dates), success_count, failed_count)
    logger.info("Total elapsed: %.1f seconds", time.time() - t0)
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
