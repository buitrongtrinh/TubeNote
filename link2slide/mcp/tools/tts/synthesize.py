"""MCP tool: synthesize text to speech, trả về đường dẫn file MP3."""
from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from ....processors.tts.synthesize import synthesize


def synthesize_text_to_speech(
    text: str,
    voice: Optional[str] = None,
) -> str:
    """Convert text to MP3 audio file. Returns the file path.

    Args:
        text: Văn bản cần đọc thành audio.
        voice: Tên voice TTS (mặc định 'vi-VN-HoaiMyNeural').
               Có thể list voices qua `list_voices`.

    Returns:
        Absolute path tới file MP3 đã tạo.
    """
    path = synthesize(text=text, voice=voice)
    return str(path)


def register(mcp: FastMCP) -> None:
    mcp.tool()(synthesize_text_to_speech)
