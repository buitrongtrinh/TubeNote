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
    sentence_pause_alpha: float | None = None,
    caption_pause_alpha: float | None = None,
) -> tuple[Transcript | None, str | None]:
    """Chạy yt-dlp trước, fallback Whisper; trả về (Transcript | None, source).

    ``source``: "manual_sub" (phụ đề người đăng tự làm) | "whisper" (STT) |
    "cache" (đọc file đã có — nguồn gốc do lần fetch trước quyết định, caller
    tra lại metadata nếu cần). Báo qua ``on_progress``: cache hit, hoặc
    fetcher nào thắng.

    ``sentence_pause_alpha``/``caption_pause_alpha``: override ngưỡng cắt câu
    theo mode người dùng chọn (None = dùng mặc định config.yaml). Chỉ có tác
    dụng khi transcript CHƯA có trong cache — cache lưu câu đã dựng sẵn, đổi
    mode cho video đã load cần xoá cache để dựng lại.
    """
    progress = on_progress or _noop

    video_id = extract_video_id(url)
    cached_path = os.path.join(SUBTITLES_DIR, f"{video_id}.json")
    if os.path.exists(cached_path):
        progress("Dùng subtitle trong cache")
        return Transcript.load_from_json(cached_path), "cache"

    progress("Thử lấy manual subtitle (yt-dlp)…")
    path = _fetch_ytdlp(
        url=url, languages=languages, output_dir=SUBTITLES_DIR,
        pause_alpha=caption_pause_alpha,
    )
    if path is not None:
        progress("Có manual subtitle — dùng yt-dlp")
        return Transcript.load_from_json(path), "manual_sub"

    progress("Không có manual subtitle → fallback Whisper STT")
    progress("Chuẩn bị Whisper STT…")
    path = _fetch_whisper(
        url=url,
        output_dir=SUBTITLES_DIR,
        preset=whisper_preset,
        on_progress=progress,
        sentence_pause_alpha=sentence_pause_alpha,
    )
    if path is None:
        return None, None
    progress("Whisper STT thành công")
    return Transcript.load_from_json(path), "whisper"
