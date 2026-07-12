"""Normalize chapter metadata supplied by YouTube/yt-dlp.

Chapters are metadata, not subtitles: their time ranges must stay owned by the
application and never be copied through an LLM translation response.
"""
from __future__ import annotations

import re
from typing import Iterable


_DESCRIPTION_CHAPTER_RE = re.compile(
    r"^\s*(?P<time>(?:\d{1,2}:)?\d{1,2}:\d{2})"
    r"(?:\s*[-–—|]\s*|\s{2,}|\s+)(?P<title>.+?)\s*$"
)


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _timestamp_seconds(value: str) -> float | None:
    parts = value.strip().split(":")
    if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts):
        return None
    numbers = [int(part) for part in parts]
    if len(numbers) == 2:
        minutes, seconds = numbers
        hours = 0
    else:
        hours, minutes, seconds = numbers
    if seconds >= 60 or minutes >= 60:
        return None
    return float(hours * 3600 + minutes * 60 + seconds)


def _clean_title(value: object) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def _finalize(items: Iterable[tuple[float, float | None, str]], duration: object) -> list[dict]:
    """Validate starts, infer ends, and return the metadata shape used by UI."""
    total_duration = _number(duration)
    ordered: list[tuple[float, float | None, str]] = []
    previous_start = -1.0
    for start, end, title in items:
        if not title or start <= previous_start:
            continue
        if total_duration is not None and start >= total_duration:
            continue
        previous_start = start
        ordered.append((start, end, title))

    if len(ordered) < 2:
        return []

    chapters: list[dict] = []
    for index, (start, supplied_end, title) in enumerate(ordered):
        if index + 1 < len(ordered):
            # A chapter remains active until the next chapter starts. yt-dlp's
            # supplied end can include a small gap, which would create an
            # unlabelled hole on Vidstack's seek bar.
            end = ordered[index + 1][0]
        else:
            end = total_duration if total_duration and total_duration > start else supplied_end
        if end is None or end <= start:
            return []
        chapters.append({
            "index": index + 1,
            "start": round(start, 3),
            "end": round(end, 3),
            "title": title,
            "title_vi": None,
        })
    return chapters


def normalize_ytdlp_chapters(raw_chapters: object, duration: object) -> list[dict]:
    """Normalize yt-dlp's ``chapters`` list without trusting malformed rows."""
    if not isinstance(raw_chapters, list):
        return []
    items: list[tuple[float, float | None, str]] = []
    for raw in raw_chapters:
        if not isinstance(raw, dict):
            continue
        start = _number(raw.get("start_time", raw.get("start")))
        title = _clean_title(raw.get("title"))
        if start is None or not title:
            continue
        items.append((start, _number(raw.get("end_time", raw.get("end"))), title))
    return _finalize(items, duration)


def parse_description_chapters(description: object, duration: object) -> list[dict]:
    """Extract genuine YouTube-style timestamp chapters from a description.

    Requiring the first marker at 00:00 and at least two entries avoids turning
    incidental timestamp references in prose into player chapters.
    """
    if not isinstance(description, str):
        return []
    items: list[tuple[float, float | None, str]] = []
    for line in description.splitlines():
        match = _DESCRIPTION_CHAPTER_RE.match(line)
        if not match:
            continue
        start = _timestamp_seconds(match.group("time"))
        title = _clean_title(match.group("title"))
        if start is None or not title:
            continue
        items.append((start, None, title))
    if not items or items[0][0] != 0:
        return []
    return _finalize(items, duration)


def chapters_from_video_info(info: dict) -> tuple[list[dict], str | None]:
    """Prefer yt-dlp chapters and only fall back to timestamped description."""
    chapters = normalize_ytdlp_chapters(info.get("chapters"), info.get("duration"))
    if chapters:
        return chapters, "yt-dlp"
    chapters = parse_description_chapters(info.get("description"), info.get("duration"))
    if chapters:
        return chapters, "description"
    return [], None
