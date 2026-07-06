"""Fetch metadata + manual subtitle qua yt-dlp (không cần cookies.txt).

Public surface:
    - ``fetch_metadata(url, ...)`` — side-effect, save ``data/metadata/{vid}.json``
    - ``fetch_transcript(url, languages, ...)`` — side-effect, save txt + json

Cả 2 fn đều decorated với ``@skip_if_exists``: nếu file đích đã tồn tại thì
bỏ qua, không hit YouTube.
"""
from __future__ import annotations

import json
import os
from typing import TypedDict

import requests
import yt_dlp

from ...config import CFG
from .cookies import get_ytdlp_cookie_opts
from .types import Transcript, TranscriptEntry
from .utils import extract_video_id, skip_if_exists


METADATA_DIR = str(CFG.paths.metadata_dir)
SUBTITLES_DIR = str(CFG.paths.subtitles_dir)


class VideoMetadata(TypedDict):
    video_id: str
    title: str | None
    channel: str | None
    channel_url: str | None
    thumbnail: str | None
    description: str | None
    duration: int | None
    view_count: int | None
    like_count: int | None
    comment_count: int | None
    channel_follower_count: int | None
    upload_date: str | None
    categories: list[str] | None
    tags: list[str] | None
    webpage_url: str | None


def _build_ydl_opts(languages: list[str]) -> dict:
    return {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": False,  # manual only
        "subtitleslangs": languages,
        "subtitlesformat": "json3",
        **get_ytdlp_cookie_opts(),
    }


def _build_metadata_ydl_opts() -> dict:
    return {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        **get_ytdlp_cookie_opts(),
    }


def _extract_info(url: str, languages: list[str]) -> dict:
    with yt_dlp.YoutubeDL(_build_ydl_opts(languages)) as ydl:
        return ydl.extract_info(url, download=False)


def _extract_metadata_info(url: str) -> dict:
    with yt_dlp.YoutubeDL(_build_metadata_ydl_opts()) as ydl:
        return ydl.extract_info(url, download=False)


def _clean_transcript(events: list[dict]) -> Transcript | None:
    entries: list[TranscriptEntry] = []
    for event in events:
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(seg.get("utf8", "") for seg in segs).strip()
        if not text:
            continue
        entries.append(TranscriptEntry(
            text=text,
            start=event.get("tStartMs", 0) / 1000,
            duration=event.get("dDurationMs", 0) / 1000,
        ))
    return Transcript(entries) if entries else None


def _get_transcript(info: dict, lang: str) -> Transcript | None:
    tracks = (info.get("subtitles") or {}).get(lang)
    if not tracks:
        return None

    json3_track = next((t for t in tracks if t.get("ext") == "json3"), None)
    if not json3_track:
        return None

    res = requests.get(json3_track["url"], timeout=15)
    res.raise_for_status()
    return _clean_transcript(res.json().get("events", []))


def _build_metadata(info: dict) -> VideoMetadata:
    return {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("uploader"),
        "channel_url": info.get("uploader_url"),
        "thumbnail": info.get("thumbnail"),
        "description": info.get("description"),
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "comment_count": info.get("comment_count"),
        "channel_follower_count": info.get("channel_follower_count"),
        "upload_date": info.get("upload_date"),
        "categories": info.get("categories"),
        "tags": info.get("tags"),
        "webpage_url": info.get("webpage_url"),
    }


def _save_metadata(metadata: VideoMetadata, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{metadata['video_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# public side-effect functions (used by orchestrator)
# ---------------------------------------------------------------------------

@skip_if_exists
def fetch_metadata(
    url: str,
    output_dir: str = METADATA_DIR,
    ext: str = "json",
) -> str:
    """Fetch + save metadata JSON. Trả về path file đã ghi."""
    info = _extract_metadata_info(url)
    metadata = _build_metadata(info)
    return _save_metadata(metadata, output_dir=output_dir)


@skip_if_exists
def fetch_transcript(
    url: str,
    languages: list[str] =["en"],
    output_dir: str = SUBTITLES_DIR,
    ext: str = "json",
) -> str | None:
    """Fetch manual subtitle, lặp qua ``languages`` theo priority.

    Side-effect: lưu ``{output_dir}/{vid}.json``.
    Trả về path json nếu OK, ``None`` nếu không có manual subtitle.
    """
    video_id = extract_video_id(url)
    info = _extract_info(url, languages)
    for lang in languages:
        trans = _get_transcript(info, lang)
        if trans:
            return trans.save_json(video_id=video_id, folder=output_dir)
    return None
