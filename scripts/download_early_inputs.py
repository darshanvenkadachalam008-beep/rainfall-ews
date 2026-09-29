"""Download IMERG Early granules for both test events (Michaung and Nov 2021)."""

import concurrent.futures
import sys
import time
from datetime import datetime
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rews.config import get_settings
from rews.ingestion.gpm_imerg import IMERGAdapter
from rews.utils.logging import get_logger

logger = get_logger("rews.scripts.download_early")


def download_single_early(dt: datetime, adapter: IMERGAdapter) -> str:
    date_tag = dt.strftime("%Y%m%d_%H%M")
    dest_nc = adapter.cache_dir / f"imerg_early_{date_tag}.nc"
    if dest_nc.exists() and dest_nc.stat().st_size > 1000:
        return f"cached:{date_tag}"
    try:
        adapter.fetch_half_hourly(dt, run_type="early")
        return f"success:{date_tag}"
    except Exception as exc:
        logger.error("Failed to download early %s: %s", date_tag, exc)
        return f"failed:{date_tag}:{exc}"


def main() -> None:
    settings = get_settings(reload=True)
    adapter = IMERGAdapter(settings)
    split_def = adapter.settings.pipeline.splits

    test_dates = []
    for ev in split_def.get("test_events", []):
        r = pd.date_range(ev["start_utc"], ev["end_utc"], freq="30min", tz="UTC")
        logger.info("Event %s: %d steps", ev["id"], len(r))
        test_dates.extend(r)

    test_dates = sorted(list(set(test_dates)))
    logger.info("Total Early granules to fetch: %d", len(test_dates))

    t0 = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_dt = {
            executor.submit(download_single_early, dt.to_pydatetime(), adapter): dt
            for dt in test_dates
        }
        done = 0
        for f in concurrent.futures.as_completed(future_to_dt):
            results.append(f.result())
            done += 1
            if done % 50 == 0 or done == len(test_dates):
                logger.info("Progress: %d / %d early granules processed (%.1fs elapsed)", done, len(test_dates), time.time() - t0)

    success = sum(1 for r in results if r.startswith("success") or r.startswith("cached"))
    failed = sum(1 for r in results if r.startswith("failed"))
    logger.info("EARLY DOWNLOAD COMPLETE: %d total, %d success/cached, %d failed in %.1fs", len(test_dates), success, failed, time.time() - t0)


if __name__ == "__main__":
    main()
