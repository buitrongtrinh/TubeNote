"""Google Gemini provider."""
from __future__ import annotations

import os
from typing import Optional


def build(
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.3,
    **_: object,
):
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Thiếu GOOGLE_API_KEY. Set trong .env hoặc truyền api_key trong UI."
        )
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=temperature,
    )
