"""Anthropic Claude provider."""
from __future__ import annotations

import os
from typing import Optional


def build(
    model: str = "claude-haiku-4-5",
    api_key: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    timeout: Optional[float] = None,
    max_retries: int = 2,
    **_: object,
):
    from langchain_anthropic import ChatAnthropic

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "Thiếu ANTHROPIC_API_KEY. Set trong .env hoặc truyền api_key trong UI."
        )
    kwargs = {
        "model": model,
        "api_key": key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    return ChatAnthropic(**kwargs)
