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
from ..video.chapters import chapters_from_video_info
from .cookies import get_ytdlp_cookie_opts
from .types import Transcript
from .utils import extract_video_id, skip_if_exists


METADATA_DIR = str(CFG.paths.metadata_dir)
SUBTITLES_DIR = str(CFG.paths.subtitles_dir)


class VideoMetadata(TypedDict):
    video_id: str
    title: str | None
    channel: str | None
    channel_id: str | None
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
    chapters: list[dict]
    chapters_source: str | None


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


# Avatar theo channel_id — nhiều video cùng kênh chỉ tra 1 lần mỗi phiên.
_CHANNEL_AVATAR_CACHE: dict[str, str] = {}


def _pick_avatar_url(thumbnails: list[dict] | None) -> str:
    """Chọn avatar trong thumbnails của TRANG KÊNH: ưu tiên id chứa 'avatar',
    không có thì lấy ảnh VUÔNG lớn nhất — banner kênh luôn là ảnh ngang dẹt
    (vd 1060x175), avatar luôn vuông (vd 900x900)."""
    best, best_w = "", 0
    for t in thumbnails or []:
        url = t.get("url") or ""
        if not url:
            continue
        if "avatar" in str(t.get("id") or "").lower():
            return url
        w, h = t.get("width"), t.get("height")
        if w and h and w == h and w > best_w:
            best, best_w = url, w
    return best


def _extract_channel_avatar(channel_id: str | None) -> str:
    """Avatar kênh KHÔNG nằm trong info của video — phải trích thêm trang
    kênh (flat + playlist_items=0 nên không đụng tới video nào, chỉ ~1-2s).
    Đây là dữ liệu trang trí: mọi lỗi đều nuốt và trả chuỗi rỗng để UI
    fallback về chữ cái đầu tên kênh — không được làm hỏng bước metadata."""
    if not channel_id:
        return ""
    if channel_id in _CHANNEL_AVATAR_CACHE:
        return _CHANNEL_AVATAR_CACHE[channel_id]
    avatar = ""
    try:
        opts = {
            **_build_metadata_ydl_opts(),
            "extract_flat": True,
            "playlist_items": "0",
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            cinfo = ydl.extract_info(
                f"https://www.youtube.com/channel/{channel_id}", download=False,
            )
        avatar = _pick_avatar_url(cinfo.get("thumbnails"))
    except Exception:
        avatar = ""
    _CHANNEL_AVATAR_CACHE[channel_id] = avatar
    return avatar


def _cues_to_pseudo_words(cues: list[dict]) -> list[dict]:
    """Chẻ cue caption thành pseudo word-timestamps để đi qua CÙNG bộ máy
    dựng câu với đường Whisper (``_split_into_entries``).

    Manual sub chỉ có timing theo cue (khối hiển thị), không theo từ — rải
    từ của mỗi cue lên khoảng thời gian của cue theo tỉ lệ độ dài ký tự.
    Từ trong cùng cue xếp nối đuôi (gap = 0) nên không bao giờ sinh điểm cắt
    pause giả; gap THẬT chỉ xuất hiện giữa 2 cue (từ timing caption). Dấu câu
    dính theo từ nên quy tắc cắt tại dấu kết câu hoạt động y hệt Whisper.
    Mốc nội suy chỉ lệch khi câu kết thúc GIỮA cue (vài trăm ms, trong sức
    hấp thụ của audio_fit).
    """
    words: list[dict] = []
    for cue in cues:
        tokens = str(cue["text"]).split()
        if not tokens:
            continue
        start = float(cue["start"])
        duration = max(float(cue["duration"]), 0.05)
        total_chars = sum(len(token) for token in tokens) + len(tokens) - 1
        cursor = start
        for token in tokens:
            share = (len(token) + 1) / max(1, total_chars)
            end = min(start + duration, cursor + duration * share)
            words.append({"word": token, "start": round(cursor, 3), "end": round(end, 3)})
            cursor = end
    return words


def _clean_transcript(events: list[dict], pause_alpha: float | None = None) -> Transcript | None:
    """json3 events -> câu hoàn chỉnh.

    Cue caption chia theo DÒNG HIỂN THỊ (vừa màn hình), không theo câu — 1 câu
    nói thường bị bổ thành 3-4 cue, ngược lại 1 cue có thể chứa cuối câu này +
    đầu câu kia. Dùng thẳng cue làm đơn vị dịch/TTS cho ra câu cụt vụn vặt —
    nên dựng lại câu bằng cùng thuật toán với đường Whisper (qua pseudo-words,
    xem ``_cues_to_pseudo_words``).
    """
    from backend.services.youtube.transcript_whisper import _split_into_entries
    from backend.config import CFG

    cues: list[dict] = []
    for event in events:
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(seg.get("utf8", "") for seg in segs).strip()
        if not text:
            continue
        cues.append({
            "text": text,
            "start": event.get("tStartMs", 0) / 1000,
            "duration": event.get("dDurationMs", 0) / 1000,
        })
    if not cues:
        return None
    entries = _split_into_entries(
        _cues_to_pseudo_words(cues),
        max_words=CFG.whisper.sentence_max_words,
        pause_alpha=pause_alpha if pause_alpha is not None else CFG.whisper.caption_sentence_pause_alpha,
        min_words=CFG.whisper.sentence_min_words,
    )
    return Transcript(entries) if entries else None


def _get_transcript(info: dict, lang: str, pause_alpha: float | None = None) -> Transcript | None:
    tracks = (info.get("subtitles") or {}).get(lang)
    if not tracks:
        return None

    json3_track = next((t for t in tracks if t.get("ext") == "json3"), None)
    if not json3_track:
        return None

    res = requests.get(json3_track["url"], timeout=15)
    res.raise_for_status()
    return _clean_transcript(res.json().get("events", []), pause_alpha)


def _build_metadata(info: dict) -> VideoMetadata:
    chapters, chapters_source = chapters_from_video_info(info)
    return {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("uploader"),
        "channel_id": info.get("channel_id"),
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
        "chapters": chapters,
        "chapters_source": chapters_source,
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
    metadata["channel_avatar"] = _extract_channel_avatar(info.get("channel_id"))
    return _save_metadata(metadata, output_dir=output_dir)


def ensure_metadata_chapters(
    url: str,
    output_dir: str = METADATA_DIR,
    ext: str = "json",
) -> str:
    """Backfill chapter metadata once for cached videos from older app versions.

    ``fetch_metadata`` is intentionally cache-first. Existing metadata therefore
    needs this narrow refresh path when it predates the ``chapters`` field.
    """
    video_id = extract_video_id(url)
    path = os.path.join(output_dir, f"{video_id}.{ext}")
    metadata: dict = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as file:
            metadata = json.load(file)
        if "chapters" in metadata:
            return path

    info = _extract_metadata_info(url)
    if not metadata:
        metadata = _build_metadata(info)
        metadata["channel_avatar"] = _extract_channel_avatar(info.get("channel_id"))
    else:
        chapters, chapters_source = chapters_from_video_info(info)
        metadata["chapters"] = chapters
        metadata["chapters_source"] = chapters_source
    return _save_metadata(metadata, output_dir=output_dir)


@skip_if_exists
def fetch_transcript(
    url: str,
    languages: list[str] =["en"],
    output_dir: str = SUBTITLES_DIR,
    ext: str = "json",
    pause_alpha: float | None = None,
) -> str | None:
    """Fetch manual subtitle, lặp qua ``languages`` theo priority.

    Side-effect: lưu ``{output_dir}/{vid}.json``.
    Trả về path json nếu OK, ``None`` nếu không có manual subtitle.
    """
    video_id = extract_video_id(url)
    info = _extract_info(url, languages)
    for lang in languages:
        trans = _get_transcript(info, lang, pause_alpha)
        if trans:
            return trans.save_json(video_id=video_id, folder=output_dir)
    return None
