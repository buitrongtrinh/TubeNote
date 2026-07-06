"""CSV performance log for load/dubbing runs."""

from __future__ import annotations

import csv
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import PROJECT_ROOT

LOG_PATH = PROJECT_ROOT / "data" / "logs" / "dubbing_runs.csv"
NAN = "NaN"
COLUMNS = [
    "run_id",
    "video_id",
    "run_index",
    "duration_min",
    "mode",
    "asr_engine",
    "asr_time_sec",
    "tts_engine",
    "tts_time_sec",
    "total_time_sec",
    "status",
    "error",
    "created_at",
]

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cell(value: Any) -> str:
    if value is None:
        return NAN
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return NAN
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    return text if text else NAN


def _read_rows() -> list[dict[str, str]]:
    if not LOG_PATH.exists():
        return []
    with LOG_PATH.open("r", encoding="utf-8", newline="") as file:
        rows = []
        for row in csv.DictReader(file):
            normalized = {column: row.get(column, NAN) or NAN for column in COLUMNS}
            if normalized["asr_time_sec"] == NAN and row.get("whisper_time_sec"):
                normalized["asr_time_sec"] = row.get("whisper_time_sec") or NAN
            rows.append(normalized)
        return rows


def _write_rows(rows: list[dict[str, str]]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOG_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, NAN) or NAN for column in COLUMNS})
    tmp.replace(LOG_PATH)


def _next_run_index(rows: list[dict[str, str]], video_id: str) -> int:
    current = 0
    for row in rows:
        if row.get("video_id") != video_id:
            continue
        try:
            current = max(current, int(row.get("run_index") or 0))
        except ValueError:
            continue
    return current + 1


def has_run(run_id: str | None) -> bool:
    if not run_id:
        return False
    with _LOCK:
        return any(row.get("run_id") == run_id for row in _read_rows())


def create_run(
    *,
    video_id: str,
    duration_min: float | None,
    mode: str,
    asr_engine: str | None = None,
    asr_time_sec: float | None = None,
    total_time_sec: float | None = None,
    status: str = "loaded",
    error: str | None = None,
) -> str:
    with _LOCK:
        rows = _read_rows()
        run_index = _next_run_index(rows, video_id)
        run_id = f"{video_id}_{run_index}"
        rows.append({
            "run_id": run_id,
            "video_id": video_id,
            "run_index": _cell(run_index),
            "duration_min": _cell(duration_min),
            "mode": _cell(mode),
            "asr_engine": _cell(asr_engine),
            "asr_time_sec": _cell(asr_time_sec),
            "tts_engine": NAN,
            "tts_time_sec": NAN,
            "total_time_sec": _cell(total_time_sec),
            "status": _cell(status),
            "error": _cell(error),
            "created_at": _now(),
        })
        _write_rows(rows)
        return run_id


def update_run(run_id: str, **fields: Any) -> bool:
    if not run_id:
        return False
    with _LOCK:
        rows = _read_rows()
        for row in rows:
            if row.get("run_id") != run_id:
                continue
            for key, value in fields.items():
                if key in COLUMNS:
                    row[key] = _cell(value)
            _write_rows(rows)
            return True
    return False


def get_run(run_id: str | None) -> dict[str, str] | None:
    if not run_id:
        return None
    with _LOCK:
        for row in _read_rows():
            if row.get("run_id") == run_id:
                return dict(row)
    return None


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", NAN):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
