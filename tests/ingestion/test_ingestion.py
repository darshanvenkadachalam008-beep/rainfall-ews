"""Unit tests for REWS ingestion, GFS APCP de-accumulation, config loading, and utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest

from rews.config import get_settings
from rews.ingestion.base import IngestionProvenance
from rews.ingestion.nwp_gfs import deaccumulate_gfs_apcp
from rews.ingestion.imd_radar import IMDRadarAdapter
from rews.utils.geo import cell_area_km2, generate_grid_coords, utm44n_to_wgs84, wgs84_to_utm44n
from rews.utils.timeutils import get_imd_accumulation_window, get_imd_day_progress, ist_to_utc, to_utc, utc_to_ist


def test_config_loading() -> None:
    """Verifies that configs/ are parsed cleanly into typed Pydantic models."""
    settings = get_settings(reload=True)

    # Pilot config checks
    assert settings.pilot.region.id == "chennai_michaung_2023"
    assert settings.pilot.domains["rainfall"].west == 78.0
    assert settings.pilot.domains["rainfall"].east == 84.0
    assert settings.pilot.grids["rainfall"]["resolution_deg"] == 0.1

    # Thresholds config checks
    assert settings.thresholds.verification["fss_neighbourhood_km"] == [10, 20, 40, 80]
    assert len(settings.thresholds.alert_levels) == 4
    assert settings.thresholds.alert_levels[3].id == "red"

    # Pipeline budget checks
    assert settings.pipeline.budget.training_budget_gb == 25.0
    assert settings.pipeline.budget.michaung_window_budget_gb == 6.0


def test_gfs_apcp_deaccumulation() -> None:
    """Verifies de-accumulation of GFS APCP across 6-hour bucket reset boundaries (Amendment F)."""
    forecast_hours = [1, 2, 3, 4, 5, 6, 7, 8]
    # Bucket 1 (f01 to f06): cumulative accumulation from cycle start
    # Bucket 2 (f07 to f08): resets at f06, starts fresh accumulation from f06
    raw_apcp = np.array([2.0, 5.0, 9.0, 14.0, 20.0, 27.0, 4.0, 10.0], dtype=np.float32)

    step_rain = deaccumulate_gfs_apcp(forecast_hours, raw_apcp)

    expected = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 4.0, 6.0], dtype=np.float32)
    np.testing.assert_allclose(step_rain, expected, rtol=1e-5)


def test_gfs_apcp_deaccumulation_nonnegative() -> None:
    """Verifies that numerical truncation noise cannot create negative precipitation."""
    forecast_hours = [1, 2, 3]
    # Synthetic slight negative jitter at step 2 due to compression/rounding
    raw_apcp = np.array([5.0, 4.999, 8.0], dtype=np.float32)
    step_rain = deaccumulate_gfs_apcp(forecast_hours, raw_apcp)

    assert np.all(step_rain >= 0.0)


def test_time_conversions() -> None:
    """Verifies timezone transforms between UTC and IST."""
    utc_dt = datetime(2023, 12, 3, 12, 0, 0, tzinfo=timezone.utc)
    ist_dt = utc_to_ist(utc_dt)

    assert ist_dt.hour == 17
    assert ist_dt.minute == 30

    roundtrip_utc = ist_to_utc(ist_dt)
    assert roundtrip_utc == utc_dt


def test_imd_accumulation_window() -> None:
    """Verifies IMD 08:30-08:30 IST (03:00 UTC) accumulation day logic."""
    # Case A: 02:00 UTC (07:30 IST) -> belongs to previous day's window
    dt_early = datetime(2023, 12, 3, 2, 0, 0, tzinfo=timezone.utc)
    start_a, end_a = get_imd_accumulation_window(dt_early)
    assert start_a == datetime(2023, 12, 2, 3, 0, 0, tzinfo=timezone.utc)
    assert end_a == datetime(2023, 12, 3, 3, 0, 0, tzinfo=timezone.utc)

    # Case B: 04:00 UTC (09:30 IST) -> belongs to today's window
    dt_late = datetime(2023, 12, 3, 4, 0, 0, tzinfo=timezone.utc)
    start_b, end_b = get_imd_accumulation_window(dt_late)
    assert start_b == datetime(2023, 12, 3, 3, 0, 0, tzinfo=timezone.utc)
    assert end_b == datetime(2023, 12, 4, 3, 0, 0, tzinfo=timezone.utc)

    # Case C: Check day progress elapsed + remaining = 24 hours
    progress = get_imd_day_progress(dt_late)
    assert progress["elapsed_hours"] == pytest.approx(1.0)
    assert progress["remaining_hours"] == pytest.approx(23.0)
    assert progress["elapsed_hours"] + progress["remaining_hours"] == pytest.approx(24.0)


def test_geospatial_helpers() -> None:
    """Verifies CRS reprojections and grid calculations."""
    lon, lat = 80.27, 13.08  # Chennai
    easting, northing = wgs84_to_utm44n(lon, lat)

    assert 400000 < easting < 500000
    assert 1400000 < northing < 1500000

    r_lon, r_lat = utm44n_to_wgs84(easting, northing)
    assert r_lon == pytest.approx(lon, abs=1e-4)
    assert r_lat == pytest.approx(lat, abs=1e-4)

    # Grid coordinates
    lons, lats = generate_grid_coords(78.0, 10.0, 80.0, 12.0, 0.5)
    assert lons[0] == 78.0
    assert lons[-1] == 80.0
    assert lats[0] == 12.0
    assert lats[-1] == 10.0

    # Grid cell area
    area_chennai = cell_area_km2(13.08, 0.1)
    assert 115.0 < area_chennai < 125.0


def test_provenance_recording(tmp_path: Path) -> None:
    """Verifies provenance record generation and checksum validation."""
    test_file = tmp_path / "sample_data.bin"
    test_file.write_bytes(b"REWS_TEST_PAYLOAD_123456789")

    prov = IngestionProvenance(
        source="test_source",
        target_path=test_file,
        time_range=("2023-12-03T00:00:00Z", "2023-12-03T01:00:00Z"),
        bbox=(78.0, 11.0, 82.0, 15.0),
        url_or_endpoint="https://example.com/test",
    )
    record = prov.finalize()

    assert record["source"] == "test_source"
    assert record["size_bytes"] == len(b"REWS_TEST_PAYLOAD_123456789")
    assert record["sha256"] is not None

    meta_file = test_file.with_suffix(".bin.meta.json")
    assert meta_file.exists()
    loaded_meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert loaded_meta["sha256"] == record["sha256"]


def test_radar_format_detection(tmp_path: Path) -> None:
    """Verifies radar format auto-detection by extension and magic bytes."""
    radar_adapter = IMDRadarAdapter()

    nc_file = tmp_path / "radar_scan.nc"
    nc_file.write_bytes(b"CDF\x02" + b"\x00" * 32)
    assert radar_adapter.detect_format(nc_file) == "netcdf"

    h5_file = tmp_path / "radar_scan.h5"
    h5_file.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 32)
    assert radar_adapter.detect_format(h5_file) == "hdf5"

    tif_file = tmp_path / "radar_scan.tif"
    tif_file.write_bytes(b"II*\x00" + b"\x00" * 32)
    assert radar_adapter.detect_format(tif_file) == "geotiff"
