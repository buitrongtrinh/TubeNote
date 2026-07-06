"""Orchestrator: thử yt-dlp manual subtitle trước, fallback Whisper STT."""
from __future__ import annotations

import os
from typing import Callable, Optional

from ...config import CFG
from .transcript_whisper import fetch_transcript as _fetch_whisper
from .transcript_yt_dlp import fetch_transcript as _fetch_ytdlp
from .types import Transcript
from .utils import extract_video_id


ProgressCallback = Callable[[str], None]
SUBTITLES_DIR = str(CFG.paths.subtitles_dir)


def _noop(_: str) -> None:
    pass


def fetch_transcript(
    url: str,
    languages: list[str],
    on_progress: Optional[ProgressCallback] = None,
    whisper_preset: str | None = None,
) -> Transcript | None:
    """Chạy yt-dlp trước, fallback Whisper; trả về Transcript hoặc None.

    Báo qua ``on_progress``: cache hit, hoặc fetcher nào thắng.
    """
    progress = on_progress or _noop

    video_id = extract_video_id(url)
    cached_path = os.path.join(SUBTITLES_DIR, f"{video_id}.json")
    if os.path.exists(cached_path):
        progress("📦 Dùng subtitle trong cache")
        return Transcript.load_from_json(cached_path)

    progress("🔍 Thử lấy manual subtitle (yt-dlp)…")
    path = _fetch_ytdlp(url=url, languages=languages, output_dir=SUBTITLES_DIR)
    if path is not None:
        progress("✅ Có manual subtitle — dùng yt-dlp")
        return Transcript.load_from_json(path)

    progress("⏭️ Không có manual subtitle → fallback Whisper STT")
    progress("🎙️ Chuẩn bị Whisper STT…")
    path = _fetch_whisper(
        url=url,
        output_dir=SUBTITLES_DIR,
        preset=whisper_preset,
        on_progress=progress,
    )
    if path is None:
        return None
    progress("✅ Whisper STT thành công")
    return Transcript.load_from_json(path)
