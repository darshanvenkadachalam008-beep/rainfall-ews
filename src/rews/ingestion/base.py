"""Base adapter specification for REWS data ingestion.

Provides standardized interface for fetching, caching, validating checksums,
and recording provenance metadata.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rews.config import REWSSettings, get_settings
from rews.utils.logging import get_logger

logger = get_logger("rews.ingestion.base")


class IngestionProvenance:
    """Provenance tracking for an ingested dataset."""

    def __init__(
        self,
        source: str,
        target_path: Path,
        time_range: Tuple[str, str],
        bbox: Tuple[float, float, float, float],
        url_or_endpoint: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.source = source
        self.target_path = target_path
        self.time_range = time_range
        self.bbox = bbox
        self.url_or_endpoint = url_or_endpoint
        self.extra_metadata = extra_metadata or {}
        self.download_timestamp_utc = datetime.now(timezone.utc).isoformat()
        self.sha256: Optional[str] = None
        self.size_bytes: int = 0

    def finalize(self) -> Dict[str, Any]:
        """Computes checksum and file size, then writes a sidecar .meta.json."""
        if self.target_path.exists() and self.target_path.is_file():
            self.size_bytes = self.target_path.stat().st_size
            hasher = hashlib.sha256()
            with open(self.target_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    hasher.update(chunk)
            self.sha256 = hasher.hexdigest()

        record = {
            "source": self.source,
            "target_file": str(self.target_path.name),
            "file_path": str(self.target_path),
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_bytes / (1024 * 1024), 3),
            "sha256": self.sha256,
            "download_timestamp_utc": self.download_timestamp_utc,
            "time_range": list(self.time_range),
            "bbox_wsen": list(self.bbox),
            "url_or_endpoint": self.url_or_endpoint,
            "extra": self.extra_metadata,
        }

        meta_path = self.target_path.with_suffix(self.target_path.suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return record


class EarthdataSession(requests.Session):
    """Preserves Authorization header across redirects to NASA hosts."""
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url
        if "Authorization" in headers:
            orig = requests.utils.urlparse(response.request.url)
            redirect = requests.utils.urlparse(url)
            if orig.hostname != redirect.hostname and redirect.hostname.endswith(".nasa.gov"):
                headers["Authorization"] = response.request.headers["Authorization"]
        return


class BaseIngestionAdapter(ABC):
    """Abstract base class for all data ingestion adapters."""

    def __init__(self, source_name: str, settings: Optional[REWSSettings] = None):
        self.source_name = source_name
        self.settings = settings or get_settings()
        self.cache_dir = self.settings.raw_dir / self.source_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = self._create_resilient_session()

    def _create_resilient_session(self) -> requests.Session:
        """Creates a requests Session with automatic exponential backoff retries."""
        session = EarthdataSession()
        retries = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def download_file_resumable(
        self,
        url: str,
        dest_path: Path,
        auth: Optional[Tuple[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Downloads a remote file with resume support and streaming chunks."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        req_headers = headers.copy() if headers else {}
        initial_bytes = 0

        if temp_path.exists():
            initial_bytes = temp_path.stat().st_size
            req_headers["Range"] = f"bytes={initial_bytes}-"
            logger.info("Resuming %s from byte %d", url, initial_bytes)

        t0 = time.time()
        with self.session.get(url, auth=auth, headers=req_headers, stream=True, timeout=60) as resp:
            if resp.status_code == 416:
                # Range not satisfiable (file already complete in temp)
                temp_path.rename(dest_path)
                return dest_path
            resp.raise_for_status()

            mode = "ab" if initial_bytes > 0 and resp.status_code == 206 else "wb"
            with open(temp_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        f.write(chunk)

        temp_path.rename(dest_path)
        elapsed = max(0.001, time.time() - t0)
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Downloaded %s (%.2f MB in %.1fs -> %.2f MB/s)",
            dest_path.name,
            size_mb,
            elapsed,
            size_mb / elapsed,
        )
        return dest_path

    @abstractmethod
    def fetch(self, *args: Any, **kwargs: Any) -> Path:
        """Fetches and caches the requested dataset, returning path to local file."""
        pass
