"""Time handling utilities for REWS.

Manages UTC/IST timezone conversions, IMD 08:30-08:30 IST accumulation day windows,
and rolling accumulation slices.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Tuple, Union
import pandas as pd


# IST is UTC + 5 hours 30 minutes
IST = timezone(timedelta(hours=5, minutes=30), name="IST")
UTC = timezone.utc


def to_utc(dt_like: Union[str, datetime, pd.Timestamp]) -> datetime:
    """Converts a string, naive datetime, or timezone-aware datetime to timezone-aware UTC."""
    if isinstance(dt_like, str):
        # Handle trailing Z or offsets
        ts = pd.to_datetime(dt_like)
        if ts.tz is None:
            return ts.tz_localize(UTC).to_pydatetime()
        return ts.tz_convert(UTC).to_pydatetime()
    elif isinstance(dt_like, pd.Timestamp):
        if dt_like.tz is None:
            return dt_like.tz_localize(UTC).to_pydatetime()
        return dt_like.tz_convert(UTC).to_pydatetime()
    elif isinstance(dt_like, datetime):
        if dt_like.tzinfo is None:
            return dt_like.replace(tzinfo=UTC)
        return dt_like.astimezone(UTC)
    else:
        raise TypeError(f"Unsupported datetime type: {type(dt_like)}")


def utc_to_ist(dt: datetime) -> datetime:
    """Converts UTC datetime to Indian Standard Time (IST)."""
    return to_utc(dt).astimezone(IST)


def ist_to_utc(dt: datetime) -> datetime:
    """Converts IST datetime to UTC datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.astimezone(UTC)


def get_imd_accumulation_window(dt: Union[str, datetime, pd.Timestamp]) -> Tuple[datetime, datetime]:
    """Returns the (start_utc, end_utc) for the official IMD 24-hour accumulation day.

    The IMD rainfall day runs from 08:30 IST on Day D to 08:30 IST on Day D+1.
    In UTC, 08:30 IST corresponds to 03:00 UTC on the same calendar day.

    If dt is at 02:00 UTC (07:30 IST), it belongs to the window starting at 03:00 UTC
    on the previous day and ending at 03:00 UTC today.
    """
    dt_utc = to_utc(dt)
    # Threshold in UTC is 03:00
    if (dt_utc.hour < 3) or (dt_utc.hour == 3 and dt_utc.minute == 0 and dt_utc.second == 0):
        # Belongs to previous day's 03:00 UTC start
        start_utc = datetime(dt_utc.year, dt_utc.month, dt_utc.day, 3, 0, 0, tzinfo=UTC) - timedelta(days=1)
    else:
        start_utc = datetime(dt_utc.year, dt_utc.month, dt_utc.day, 3, 0, 0, tzinfo=UTC)

    end_utc = start_utc + timedelta(hours=24)
    return start_utc, end_utc


def get_imd_day_progress(current_time: Union[str, datetime, pd.Timestamp]) -> dict[str, Any]:
    """Computes the elapsed and remaining portions of the active IMD 08:30-08:30 IST day.

    Crucial for Amendment G:
    The 24 h class probability is P(observed accumulation so far + forecast remainder >= threshold).
    """
    now_utc = to_utc(current_time)
    win_start, win_end = get_imd_accumulation_window(now_utc)
    elapsed_seconds = max(0.0, (now_utc - win_start).total_seconds())
    remaining_seconds = max(0.0, (win_end - now_utc).total_seconds())

    return {
        "window_start_utc": win_start.isoformat(),
        "window_end_utc": win_end.isoformat(),
        "elapsed_hours": elapsed_seconds / 3600.0,
        "remaining_hours": remaining_seconds / 3600.0,
        "elapsed_ratio": min(1.0, elapsed_seconds / (24.0 * 3600.0)),
    }
