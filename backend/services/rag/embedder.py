"""RAG embedding factory.

Project lock vào ``CFG.embedding.model`` (xem config.yaml). Đổi model →
bắt buộc xóa ``data/chroma/*`` vì dim thay đổi.
"""
from __future__ import annotations

from ...config import CFG

_EMBEDDING = None


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_embedding():
    global _EMBEDDING
    if _EMBEDDING is None:
        provider = (CFG.embedding.provider or "huggingface").lower()
        if provider not in {"huggingface", "sentence_transformers", "sentence-transformers"}:
            raise RuntimeError(f"Embedding provider không hỗ trợ: {CFG.embedding.provider!r}")

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:
            raise RuntimeError(
                "Thiếu dependency embedding local. Chạy: "
                "python -m pip install -r requirements.txt"
            ) from e

        _EMBEDDING = HuggingFaceEmbeddings(
            model=CFG.embedding.model,
            model_kwargs={
                "device": _resolve_device(CFG.embedding.device),
                "local_files_only": CFG.embedding.local_files_only,
            },
            encode_kwargs={"normalize_embeddings": CFG.embedding.normalize},
        )
    return _EMBEDDING
