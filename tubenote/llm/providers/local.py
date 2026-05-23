"""Local LLM provider — Ollama (đã host sẵn local hoặc remote)."""
from __future__ import annotations

import os
from typing import Optional


def build(
    model: str = "llama3.2:latest",
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    num_ctx: int = 8192,
    **_: object,
):
    from langchain_ollama import ChatOllama

    url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(
        model=model,
        base_url=url,
        temperature=temperature,
        num_ctx=num_ctx,
    )
