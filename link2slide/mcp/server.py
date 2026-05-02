"""MCP stdio server. Auto-register mọi tool trong package `tools/`."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .tools import register_all

mcp = FastMCP("link2slide")
register_all(mcp)


if __name__ == "__main__":
    mcp.run()
