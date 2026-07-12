"""Build WebVTT subtitles from local transcript/dubbing segments."""
from __future__ import annotations

from backend.services.video.timing import display_range


def _ts(t: float) -> str:
    """giây -> 'HH:MM:SS.mmm' (định dạng WebVTT)."""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def build_vtt(segments: list[dict], field: str, prefer_tts_timing: bool = False) -> str:
    """Dựng nội dung .vtt từ ``field`` của mỗi segment.

    Cue hiện từ start của segment hiện tại đến start của segment tiếp theo.
    Segment cuối dùng start + duration.
    """
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments):
        text = (seg.get(field) or "").replace("\n", " ").strip()
        if not text:
            continue
        start, end = display_range(segments, i, prefer_tts_timing=prefer_tts_timing)
        if end <= start:
            end = start + max(seg.get("duration", 0), 0.5)
        lines.append(f"{_ts(start)} --> {_ts(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def build_chapter_vtt(chapters: list[dict]) -> str:
    """Build a WebVTT chapters track from normalized metadata."""
    lines = ["WEBVTT", ""]
    for chapter in chapters:
        title = " ".join(str(chapter.get("title_vi") or "").split())
        try:
            start = float(chapter.get("start"))
            end = float(chapter.get("end"))
        except (TypeError, ValueError):
            continue
        if not title or end <= start:
            continue
        lines.append(f"{_ts(start)} --> {_ts(end)}")
        lines.append(title)
        lines.append("")
    return "\n".join(lines)
