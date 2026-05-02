"""Liệt kê / filter voices của TTS provider."""
from __future__ import annotations

from typing import List, Optional


# Cache list voice (gọi async API sẽ chậm) — có thể fill khi khởi động
_VOICE_CACHE: Optional[List[dict]] = None


def list_voices(language: Optional[str] = None) -> List[dict]:
    """Trả về list voice. Format mỗi voice:
        {"name": "vi-VN-HoaiMyNeural", "gender": "Female", "language": "vi-VN", ...}

    Args:
        language: lọc theo locale (vd 'vi', 'en'). None = trả tất cả.

    Returns:
        List voice metadata.
    """
    # TODO: dùng edge_tts.list_voices() (async) — bọc qua asyncio.run
    # import asyncio
    # import edge_tts
    # voices = asyncio.run(edge_tts.list_voices())
    # if language:
    #     voices = [v for v in voices if v["Locale"].startswith(language)]
    # return voices
    raise NotImplementedError("TODO: implement với edge-tts list_voices().")
