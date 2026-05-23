"""Ollama embedding — tránh torch/transformers DLL conflict trên Windows.

Project lock vào ``CFG.embedding.model`` (xem config.yaml). Đổi model →
bắt buộc xóa ``output/chroma/*`` vì dim thay đổi.
"""
from __future__ import annotations

from ...config import CFG

_EMBEDDING = None


def get_embedding():
    global _EMBEDDING
    if _EMBEDDING is None:
        from langchain_ollama import OllamaEmbeddings

        _EMBEDDING = OllamaEmbeddings(
            model=CFG.embedding.model,
            base_url=CFG.embedding.base_url,
        )
    return _EMBEDDING
