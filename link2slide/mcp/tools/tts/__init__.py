"""TTS MCP tools — đăng ký mọi tool con của domain TTS."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import list_voices, synthesize


def register(mcp: FastMCP) -> None:
    synthesize.register(mcp)
    list_voices.register(mcp)
