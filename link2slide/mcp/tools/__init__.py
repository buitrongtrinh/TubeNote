"""MCP tool registry.

Mỗi tool sống trong 1 file module .py trong folder này. Mỗi file phải
expose hàm `register(mcp: FastMCP)` để attach tool vào MCP server.
`register_all()` sẽ tự scan và gọi chúng.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def register_all(mcp: "FastMCP") -> list[str]:
    """Import mọi module con và gọi register(mcp). Trả về list tên đã load."""
    loaded = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        register = getattr(module, "register", None)
        if register is None:
            continue
        register(mcp)
        loaded.append(info.name)
    return loaded
