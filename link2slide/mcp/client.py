"""Explicit MCP stdio client — spawn MCP server as subprocess, talk via stdio."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..config import PROJECT_ROOT


class MCPClient:
    """Persistent stdio connection to an MCP server subprocess."""

    def __init__(
        self,
        command: str,
        args: List[str],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self._params = StdioServerParameters(
            command=command,
            args=args,
            cwd=str(cwd or PROJECT_ROOT),
            env=env,
        )
        self._stdio_cm = None
        self._session_cm = None
        self._session: Optional[ClientSession] = None

    @classmethod
    def default(cls) -> "MCPClient":
        """Spawn the bundled MCP server (apps/mcp_server.py)."""
        return cls(command=sys.executable, args=["-m", "apps.mcp_server"])

    async def __aenter__(self) -> "MCPClient":
        self._stdio_cm = stdio_client(self._params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._session_cm is not None:
                await self._session_cm.__aexit__(exc_type, exc, tb)
        finally:
            if self._stdio_cm is not None:
                await self._stdio_cm.__aexit__(exc_type, exc, tb)
            self._session = None
            self._session_cm = None
            self._stdio_cm = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCPClient not connected — use `async with MCPClient(...)`.")
        return self._session

    async def list_tools(self) -> List[Any]:
        resp = await self.session.list_tools()
        return list(resp.tools)

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> str:
        resp = await self.session.call_tool(name, arguments or {})
        parts: List[str] = []
        for c in resp.content:
            text = getattr(c, "text", None)
            if text is not None:
                parts.append(text)
        if resp.isError:
            raise RuntimeError(f"Tool {name!r} failed: {''.join(parts) or '<no detail>'}")
        return "\n".join(parts)

    async def langchain_tools(self) -> list:
        from langchain_mcp_adapters.tools import load_mcp_tools

        return await load_mcp_tools(self.session)
