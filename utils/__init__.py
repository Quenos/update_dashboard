"""Utility helpers shared across the update_dashboard project."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def read_timestamp(path: str, *, tz: timezone = timezone.utc) -> Optional[datetime]:
    """Return the timestamp stored at *path* or ``None`` if unavailable."""
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        content = file_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not content:
        return None
    try:
        ts = datetime.fromisoformat(content)
    except ValueError:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=tz)
    return ts


def write_timestamp(timestamp: datetime, path: str) -> None:
    """Persist *timestamp* to *path* using ISO 8601 representation."""
    file_path = Path(path)
    try:
        file_path.write_text(timestamp.isoformat(), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Unable to write timestamp to {path}") from exc
