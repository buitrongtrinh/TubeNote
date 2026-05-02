"""Lấy metadata YouTube video qua oembed (public, không cần API key)."""
from __future__ import annotations

import json
import urllib.request
from typing import Optional, TypedDict

from .transcripts import extract_video_id


class VideoMetadata(TypedDict, total=False):
    video_id: str
    title: Optional[str]
    channel: Optional[str]
    channel_url: Optional[str]
    thumbnail: Optional[str]


def fetch_metadata(url: str) -> VideoMetadata:
    """Trả về dict metadata: video_id, title, channel, channel_url, thumbnail."""
    video_id = extract_video_id(url)
    oembed = (
        "https://www.youtube.com/oembed?"
        f"url=https://youtu.be/{video_id}&format=json"
    )
    req = urllib.request.Request(oembed, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.load(r)
    return {
        "video_id": video_id,
        "title": data.get("title"),
        "channel": data.get("author_name"),
        "channel_url": data.get("author_url"),
        "thumbnail": data.get("thumbnail_url"),
    }
