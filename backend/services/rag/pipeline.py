"""RAG pipeline — local subtitles/transcript → chunk → embed → store."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from ...config import CFG
from ..video.timing import display_range
from ..youtube.transcript import fetch_transcript
from ..youtube.utils import extract_video_id
from .chunker import chunk_subtitle_segments, chunk_text
from .store import get_vector_store, has_timestamp_index, is_indexed, reset_vector_store

ProgressCallback = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def _load_json(path: Path) -> list[dict]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise RuntimeError(f"Subtitle không đúng định dạng list: {path}")
    return data


def _subtitle_candidates(video_id: str) -> list[tuple[str, Path]]:
    return [
        ("subtitles", CFG.paths.subtitles_dir / f"{video_id}.json"),
    ]


def _segment_text(seg: dict, source: str) -> str:
    if seg.get("text_vi"):
        return str(seg["text_vi"]).strip()
    if seg.get("text_tts"):
        return str(seg["text_tts"]).strip()
    return str(seg.get("text") or "").replace("\n", " ").strip()


def load_local_subtitle_segments(video_id: str) -> tuple[str, list[dict]]:
    """Load the best local subtitle file and normalize it for RAG chunking."""
    for source, path in _subtitle_candidates(video_id):
        if not path.exists():
            continue
        raw_segments = _load_json(path)
        normalized: list[dict] = []
        for index, seg in enumerate(raw_segments):
            if not isinstance(seg, dict):
                continue
            text = _segment_text(seg, source)
            if not text:
                continue
            start, end = display_range(raw_segments, index, prefer_tts_timing=True)
            if end <= start:
                end = start + max(float(seg.get("duration", 0.0) or 0.0), 0.5)
            normalized.append({
                "index": index,
                "start": start,
                "end": end,
                "text": text,
            })
        if normalized:
            return source, normalized
    raise RuntimeError("Không tìm thấy subtitle local để index RAG.")


def ingest_video_id(video_id: str, on_progress: Optional[ProgressCallback] = None) -> dict:
    """Index local subtitles for a video. Reindex legacy plain-text collections."""
    progress = on_progress or _noop
    video_id = extract_video_id(video_id)

    if has_timestamp_index(video_id):
        progress(f"Video `{video_id}` đã indexed RAG sẵn")
        return {"video_id": video_id, "indexed": False, "chunks": 0}
    if is_indexed(video_id):
        progress("Collection RAG cũ thiếu timestamp, đang reindex")
        reset_vector_store(video_id)

    source, segments = load_local_subtitle_segments(video_id)
    progress(f"Đang chunk subtitle từ {source} ({len(segments)} segments)")
    docs = chunk_subtitle_segments(segments, video_id=video_id, source=source)
    if not docs:
        raise RuntimeError("Không tạo được chunk RAG từ subtitle.")
    for doc in docs:
        doc.metadata["embedding_provider"] = CFG.embedding.provider
        doc.metadata["embedding_model"] = CFG.embedding.model

    progress(f"Đang embed + lưu {len(docs)} chunks vào Chroma")
    store = get_vector_store(video_id)
    store.add_documents(docs)
    progress(f"Đã index RAG cho video `{video_id}`")
    return {"video_id": video_id, "indexed": True, "chunks": len(docs), "source": source}


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
        progress(f"Video `{video_id}` đã indexed sẵn — dùng cache")
        return {"video_id": video_id, "indexed": False, "chunks": 0}

    trans = fetch_transcript(url, languages=languages, on_progress=progress)
    if trans is None:
        raise RuntimeError("Không lấy được transcript bằng cả yt-dlp lẫn Whisper.")
    progress(f"Transcript OK ({len(trans.segments)} segments)")

    progress("Đang split text thành chunks…")
    docs = chunk_text(trans.full_text)
    progress(f"Đang embed + store {len(docs)} chunks vào Chroma…")
    store = get_vector_store(video_id)
    store.add_documents(docs)
    progress(f"Indexed {len(docs)} chunks cho video `{video_id}`")

    return {"video_id": video_id, "indexed": True, "chunks": len(docs)}
