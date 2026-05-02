"""MCP tool: list caption languages available on a YouTube video."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from youtube_transcript_api import YouTubeTranscriptApi

from ....processors.youtube.cookies import build_session, list_cookie_files
from ....processors.youtube.transcripts import extract_video_id


def list_available_languages(url: str) -> str:
    """List all caption languages available for a YouTube video.

    Returns one language per line as:
        code (name) [manual|generated] [translatable]
    """
    video_id = extract_video_id(url)
    files = list_cookie_files()
    session = build_session(files[0]) if files else None
    api = YouTubeTranscriptApi(http_client=session) if session else YouTubeTranscriptApi()
    lines = []
    for t in api.list(video_id):
        kind = "generated" if t.is_generated else "manual"
        tr = " translatable" if t.is_translatable else ""
        lines.append(f"{t.language_code} ({t.language}) [{kind}]{tr}")
    return "\n".join(lines) if lines else "(no transcripts available)"


def register(mcp: FastMCP) -> None:
    mcp.tool()(list_available_languages)
