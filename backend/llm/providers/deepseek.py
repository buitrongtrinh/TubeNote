"""DeepSeek provider via OpenAI-compatible API."""
from __future__ import annotations

import os
from typing import Optional


def build(
    model: str = "deepseek-v4-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    **_: object,
):
    from langchain_openai import ChatOpenAI

    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "Thiếu DEEPSEEK_API_KEY. Set trong .env hoặc truyền api_key trong UI."
        )
    kwargs = {
        "model": model,
        "api_key": key,
        "base_url": base_url or "https://api.deepseek.com",
        "temperature": temperature,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return ChatOpenAI(**kwargs)
