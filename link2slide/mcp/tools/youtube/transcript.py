"""MCP tool: fetch YouTube transcript with language fallback + time-range."""
from __future__ import annotations

from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from ....youtube.transcripts import fetch_transcript


def get_youtube_transcript(
    url: str,
    languages: Optional[List[str]] = None,
    start_seconds: float = 0,
    end_seconds: float = 0,
) -> str:
    """Fetch the transcript/captions of a YouTube video as plain text.

    Use start_seconds / end_seconds to extract only a portion of the video
    (first 5 minutes -> end_seconds=300). end_seconds=0 means to the end.

    Args:
        url: Full YouTube URL or 11-character video ID.
        languages: Preferred caption languages (default from config.yaml).
        start_seconds: Start offset in seconds.
        end_seconds: End offset in seconds (0 = until end).
    """
    return fetch_transcript(
        url=url,
        languages=languages,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def register(mcp: FastMCP) -> None:
    mcp.tool()(get_youtube_transcript)
