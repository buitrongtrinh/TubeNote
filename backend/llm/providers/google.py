"""Google Gemini provider."""
from __future__ import annotations

import os
from typing import Optional


def build(
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.3,
    request_timeout: Optional[float] = None,
    timeout: Optional[float] = None,
    retries: Optional[int] = None,
    max_retries: Optional[int] = None,
    **_: object,
):
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Thiếu GOOGLE_API_KEY. Set trong .env hoặc truyền api_key trong UI."
        )
    kwargs = {
        "model": model,
        "google_api_key": key,
        "temperature": temperature,
    }
    effective_timeout = request_timeout if request_timeout is not None else timeout
    if effective_timeout is not None:
        kwargs["request_timeout"] = effective_timeout
    effective_retries = retries if retries is not None else max_retries
    if effective_retries is not None:
        kwargs["retries"] = effective_retries
    return ChatGoogleGenerativeAI(**kwargs)
