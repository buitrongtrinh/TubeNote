"""MCP tool: liệt kê voice có sẵn."""
from __future__ import annotations

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from ....tts.voices import list_voices as _list_voices


def list_tts_voices(language: Optional[str] = None) -> str:
    """List available TTS voices, optionally filtered by language.

    Args:
        language: Locale filter (vd 'vi', 'en'). Bỏ trống = lấy tất cả.

    Returns:
        JSON string danh sách voice (name, gender, language).
    """
    voices = _list_voices(language)
    return json.dumps(voices, ensure_ascii=False, indent=2)


def register(mcp: FastMCP) -> None:
    mcp.tool()(list_tts_voices)
