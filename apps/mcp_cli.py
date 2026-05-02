"""Debug CLI — gọi MCP tool trực tiếp (không qua LLM agent).

Usage:
    python -m apps.mcp_cli list
    python -m apps.mcp_cli call <tool_name> [key=value ...]

Ví dụ:
    python -m apps.mcp_cli list
    python -m apps.mcp_cli call get_video_metadata url="https://youtu.be/aircAruvnKk"
    python -m apps.mcp_cli call list_available_languages url="https://youtu.be/aircAruvnKk"
    python -m apps.mcp_cli call get_youtube_transcript url="https://youtu.be/aircAruvnKk" end_seconds=60
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from link2slide.mcp.client import MCPClient


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _parse_kwargs(argv: list[str]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    for token in argv:
        if "=" not in token:
            raise SystemExit(f"Argument phải dạng key=value, gặp: {token!r}")
        k, v = token.split("=", 1)
        kwargs[k.strip()] = _coerce(v.strip())
    return kwargs


async def _list() -> None:
    async with MCPClient.default() as client:
        tools = await client.list_tools()
        for t in tools:
            print(f"\n=== {t.name} ===")
            if t.description:
                print(t.description.strip())
            print("schema:", json.dumps(t.inputSchema, indent=2, ensure_ascii=False))


async def _call(name: str, args: Dict[str, Any]) -> None:
    async with MCPClient.default() as client:
        out = await client.call_tool(name, args)
        print(out)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "list":
        asyncio.run(_list())
    elif cmd == "call":
        if len(sys.argv) < 3:
            print("Usage: python -m apps.mcp_cli call <tool_name> [key=value ...]")
            sys.exit(1)
        tool_name = sys.argv[2]
        kwargs = _parse_kwargs(sys.argv[3:])
        asyncio.run(_call(tool_name, kwargs))
    else:
        print(f"Lệnh không hợp lệ: {cmd!r}. Dùng 'list' hoặc 'call'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
