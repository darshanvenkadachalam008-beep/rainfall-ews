"""ECMWF Open Data adapter.

Empirically tests retrieval from ECMWF Open Data service, documents
historical availability, and provides fallback to local GRIB files.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd
from ecmwf.opendata import Client

from rews.config import REWSSettings, get_settings
from rews.ingestion.base import BaseIngestionAdapter, IngestionProvenance
from rews.utils.logging import get_logger
from rews.utils.timeutils import to_utc

logger = get_logger("rews.ingestion.ecmwf")


class ECMWFOpenDataAdapter(BaseIngestionAdapter):
    """Adapter for ECMWF Open Data (IFS / AIFS)."""

    def __init__(self, source: str = "ecmwf", settings: Optional[REWSSettings] = None):
        super().__init__("nwp_ecmwf", settings)
        self.client = Client(source=source)

    def test_retrieval(self, date: Union[str, datetime], step: int = 0) -> Dict[str, Any]:
        """Empirically tests whether ECMWF Open Data serves the requested date and step."""
        dt_utc = to_utc(date)
        date_str = dt_utc.strftime("%Y%m%d")
        time_int = dt_utc.hour

        target_test_file = self.cache_dir / f"test_probe_{date_str}_{time_int:02d}z_step{step}.grib2"
        result: Dict[str, Any] = {
            "date": date_str,
            "time": time_int,
            "step": step,
            "available": False,
            "error": None,
            "file_size_bytes": 0,
        }

        try:
            logger.info("Empirically probing ECMWF Open Data for date=%s, time=%d, step=%d...", date_str, time_int, step)
            self.client.retrieve(
                date=date_str,
                time=time_int,
                step=step,
                stream="oper",
                type="fc",
                levtype="sfc",
                param=["tp", "2t", "10u", "10v", "msl"],
                target=str(target_test_file),
            )
            if target_test_file.exists() and target_test_file.stat().st_size > 0:
                result["available"] = True
                result["file_size_bytes"] = target_test_file.stat().st_size
                logger.info("ECMWF retrieval SUCCEEDED (size: %d bytes)", result["file_size_bytes"])
        except Exception as exc:
            result["available"] = False
            result["error"] = str(exc)
            logger.warning("ECMWF empirical retrieval failed for %s: %s", date_str, exc)
        finally:
            if target_test_file.exists():
                try:
                    target_test_file.unlink()
                except Exception:
                    pass

        return result

    def fetch(
        self, date: Union[str, datetime], lead_hours: Sequence[int]
    ) -> List[Path]:
        """Fetches forecast files or loads local manual GRIB files."""
        dt_utc = to_utc(date)
        date_tag = dt_utc.strftime("%Y%m%d_%H%M")
        paths: List[Path] = []

        for step in lead_hours:
            dest_file = self.cache_dir / f"ecmwf_{date_tag}_f{step:03d}.grib2"
            if dest_file.exists():
                paths.append(dest_file)
                continue

            # Try online retrieve
            try:
                self.client.retrieve(
                    date=dt_utc.strftime("%Y%m%d"),
                    time=dt_utc.hour,
                    step=step,
                    stream="oper",
                    type="fc",
                    levtype="sfc",
                    param=["tp", "2t", "10u", "10v", "msl"],
                    target=str(dest_file),
                )
                paths.append(dest_file)
            except Exception as exc:
                # Check for manual file
                manual = list(self.cache_dir.glob(f"*{date_tag}*f{step:03d}*.grib*"))
                if manual:
                    paths.append(manual[0])
                else:
                    logger.warning(
                        "ECMWF Open Data unavailable for %s step %d: %s. "
                        "Fallback: Place GRIB files in %s or rely on GFS.",
                        date_tag,
                        step,
                        exc,
                        self.cache_dir,
                    )
        return paths
