"""Corpus downloader for SIH26071 REWS.

Downloads remaining multi-year train, validation, and secondary test events
from NASA GES DISC using Earthdata credentials. Automatically skips existing
NetCDFs, crops raw HDF5s to pilot bbox, and purges intermediate raw HDF5s.
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

logger = get_logger("rews.scripts.download_corpus")


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
    split_def = adapter.settings.pipeline.splits

    # Gather all event windows from config
    all_ranges = []
    
    # 1. Test events
    for ev in split_def.get("test_events", []):
        r = pd.date_range(ev["start_utc"], ev["end_utc"], freq="30min", tz="UTC")
        all_ranges.append((ev["id"], r))
        
    # 2. Validation events
    for ev in split_def.get("validation_events", []):
        r = pd.date_range(ev["start_utc"], ev["end_utc"], freq="30min", tz="UTC")
        all_ranges.append((ev["id"], r))

    # 3. Training events
    for ev in split_def.get("train_events", []):
        r = pd.date_range(ev["start_utc"], ev["end_utc"], freq="30min", tz="UTC")
        all_ranges.append((ev["id"], r))

    total_dates = []
    for ev_id, r in all_ranges:
        logger.info("Configured event %s: %d steps (%s to %s)", ev_id, len(r), r[0].isoformat(), r[-1].isoformat())
        total_dates.extend(r)

    total_dates = sorted(list(set(total_dates)))
    logger.info("Total unique corpus dates to check: %d steps", len(total_dates))

    t0 = time.time()
    results = []
    max_workers = 32

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dt = {
            executor.submit(download_single_step, dt.to_pydatetime(), adapter): dt
            for dt in total_dates
        }
        done_count = 0
        for future in concurrent.futures.as_completed(future_to_dt):
            res = future.result()
            results.append(res)
            done_count += 1
            if done_count % 50 == 0 or done_count == len(total_dates):
                elapsed = time.time() - t0
                logger.info("Progress: %d / %d processed (%.1fs elapsed)", done_count, len(total_dates), elapsed)

    success_count = sum(1 for r in results if r.startswith("success") or r.startswith("cached"))
    failed_count = sum(1 for r in results if r.startswith("failed"))

    logger.info("=================================================================")
    logger.info("CORPUS DOWNLOAD COMPLETE")
    logger.info("Total steps: %d | Success/Cached: %d | Failed: %d", len(total_dates), success_count, failed_count)
    logger.info("Total elapsed: %.1f seconds", time.time() - t0)
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
