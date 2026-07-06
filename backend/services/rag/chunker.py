"""Text splitter — port nguyên từ rag.ipynb."""
from __future__ import annotations


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def chunk_text(full_text: str, chunk_size: int = 500, chunk_overlap: int = 100):
    """Split → cleanup → Document objects."""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_text(full_text)
    chunks = [c.lstrip(". ").strip() for c in chunks if c.strip()]
    return [Document(page_content=c) for c in chunks]


def chunk_subtitle_segments(
    segments: list[dict],
    *,
    video_id: str,
    source: str,
    target_chunk_chars: int = 900,
    max_chunk_chars: int = 1200,
    overlap_segments: int = 2,
):
    """Build timestamp-aware chunks from subtitle segments.

    ``segments`` items must already contain normalized ``text``, ``start``,
    ``end`` and ``index`` keys. Overlap is segment-based so boundary questions
    still have enough neighboring context for retrieval.
    """
    from langchain_core.documents import Document

    clean_segments = [
        {
            "index": int(seg["index"]),
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": str(seg.get("text") or "").strip(),
        }
        for seg in segments
        if str(seg.get("text") or "").strip()
    ]
    if not clean_segments:
        return []

    docs: list[Document] = []

    def current_len(items: list[dict]) -> int:
        return sum(len(item["text"]) for item in items) + max(0, len(items) - 1)

    def emit(items: list[dict]) -> None:
        if not items:
            return
        start = items[0]["start"]
        end = max(item["end"] for item in items)
        text = " ".join(item["text"] for item in items).strip()
        if not text:
            return
        docs.append(Document(
            page_content=f"[{_fmt_time(start)}-{_fmt_time(end)}] {text}",
            metadata={
                "video_id": video_id,
                "source": source,
                "start": start,
                "end": end,
                "segment_start": items[0]["index"],
                "segment_end": items[-1]["index"],
                "rag_version": "subtitle-v1",
            },
        ))

    current: list[dict] = []
    last_emitted_end = -1
    for seg in clean_segments:
        current.append(seg)
        if current_len(current) >= target_chunk_chars or current_len(current) >= max_chunk_chars:
            emit(current)
            last_emitted_end = current[-1]["index"]
            current = current[-overlap_segments:] if overlap_segments > 0 else []
    if current and (not docs or current[-1]["index"] > last_emitted_end):
        emit(current)
    return docs
