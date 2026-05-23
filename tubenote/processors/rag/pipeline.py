"""RAG pipeline — fetch transcript → chunk → embed → store. Idempotent."""
from __future__ import annotations

from typing import Callable, Optional

from ...config import CFG
from ..youtube.transcript import fetch_transcript
from ..youtube.utils import extract_video_id
from .chunker import chunk_text
from .store import get_vector_store, is_indexed

ProgressCallback = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def ingest(url: str, on_progress: Optional[ProgressCallback] = None) -> dict:
    """Fetch transcript → chunk → embed → store. Idempotent.

    Caching ở 2 tầng:
        - File system (mỗi fetcher có ``@skip_if_exists``)
        - Vector store (``is_indexed`` check)

    Returns:
        ``{"video_id": str, "indexed": bool, "chunks": int}``
    """
    progress = on_progress or _noop
    languages = CFG.transcript.default_languages

    video_id = extract_video_id(url)

    if is_indexed(video_id):
        progress(f"✅ Video `{video_id}` đã indexed sẵn — dùng cache")
        return {"video_id": video_id, "indexed": False, "chunks": 0}

    trans = fetch_transcript(url, languages=languages, on_progress=progress)
    if trans is None:
        raise RuntimeError("Không lấy được transcript bằng cả yt-dlp lẫn Whisper.")
    progress(f"📝 Transcript OK ({len(trans.segments)} segments)")

    progress("✂️ Đang split text thành chunks…")
    docs = chunk_text(trans.full_text)
    progress(f"📊 Đang embed + store {len(docs)} chunks vào Chroma…")
    store = get_vector_store(video_id)
    store.add_documents(docs)
    progress(f"✅ Indexed {len(docs)} chunks cho video `{video_id}`")

    return {"video_id": video_id, "indexed": True, "chunks": len(docs)}
