"""LangChain Chroma store với cosine distance + normalized relevance score."""
from __future__ import annotations

from langchain_core.documents import Document

from ...config import PROJECT_ROOT
from .embedder import get_embedding


_PERSIST_DIR = PROJECT_ROOT / "output" / "chroma"


def _cosine_to_relevance(distance: float) -> float:
    """Chroma cosine distance ∈ [0, 2] → relevance ∈ [0, 1].

    distance=0 (identical) → 1.0
    distance=1 (orthogonal) → 0.5
    distance=2 (opposite) → 0.0
    """
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


def get_vector_store(video_id: str):
    from langchain_chroma import Chroma

    _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=f"video_{video_id}",
        embedding_function=get_embedding(),
        persist_directory=str(_PERSIST_DIR),
        collection_metadata={"hnsw:space": "cosine"},
        relevance_score_fn=_cosine_to_relevance,
    )


def is_indexed(video_id: str) -> bool:
    return get_vector_store(video_id)._collection.count() > 0


def get_all_docs(store) -> list[Document]:
    """Lấy toàn bộ chunks từ Chroma store — dùng để build BM25 in-memory."""
    raw = store.get()
    texts = raw.get("documents") or []
    metas = raw.get("metadatas") or [{}] * len(texts)
    return [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metas)]
