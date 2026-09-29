"""Structured logging module for REWS."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str = "rews", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Returns a standardized logger for REWS components."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(_FORMATTER)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setFormatter(_FORMATTER)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    return logger
