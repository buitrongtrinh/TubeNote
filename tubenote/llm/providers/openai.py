"""OpenAI provider."""
from __future__ import annotations

import os
from typing import Optional


def build(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.3,
    base_url: Optional[str] = None,
    **_: object,
):
    from langchain_openai import ChatOpenAI

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Thiếu OPENAI_API_KEY. Set trong .env hoặc truyền api_key trong UI."
        )
    kwargs = {"model": model, "api_key": key, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url  # support OpenAI-compatible endpoints
    return ChatOpenAI(**kwargs)
