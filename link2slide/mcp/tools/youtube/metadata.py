"""MCP tool: fetch YouTube video metadata via oembed (no API key)."""
from __future__ import annotations

import json
import urllib.request

from mcp.server.fastmcp import FastMCP

from ....youtube.transcripts import extract_video_id


def get_video_metadata(url: str) -> str:
    """Fetch basic YouTube video metadata (title, channel, thumbnail) via oembed.

    Does not require API key. Returns JSON string.
    """
    video_id = extract_video_id(url)
    oembed = (
        "https://www.youtube.com/oembed?"
        f"url=https://youtu.be/{video_id}&format=json"
    )
    req = urllib.request.Request(oembed, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.load(r)
    return json.dumps(
        {
            "video_id": video_id,
            "title": data.get("title"),
            "channel": data.get("author_name"),
            "channel_url": data.get("author_url"),
            "thumbnail": data.get("thumbnail_url"),
        },
        ensure_ascii=False,
        indent=2,
    )


def register(mcp: FastMCP) -> None:
    mcp.tool()(get_video_metadata)
