"""YouTube MCP tools — đăng ký mọi tool con của domain youtube."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import languages, metadata, transcript


def register(mcp: FastMCP) -> None:
    transcript.register(mcp)
    metadata.register(mcp)
    languages.register(mcp)
