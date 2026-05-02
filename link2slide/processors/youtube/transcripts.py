"""YouTube transcript fetching with language fallback + cookie rotation."""
from __future__ import annotations

import re
import time
from typing import List, Optional

from youtube_transcript_api import FetchedTranscript, YouTubeTranscriptApi

from ...config import CFG
from .cookies import build_session, list_cookie_files

_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/|youtube\.com/live/)"
    r"([A-Za-z0-9_-]{11})"
)


def extract_video_id(url_or_id: str) -> str:
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    m = _VIDEO_ID_RE.search(s)
    if m:
        return m.group(1)
    raise ValueError(f"Không lấy được video ID từ: {url_or_id!r}")

def check_languages(languages: Optional[List[str]]) -> List[str]:
    if not languages:
        return CFG.transcript.default_languages
    for lang in languages:
        if not re.fullmatch(r"[a-z]{2}(-[A-Z]{2})?", lang):
            raise ValueError(f"Invalid language code: {lang!r}")
    return languages

def _fetch_once(video_id: str, languages: List[str], session)-> Optional[FetchedTranscript]:
    """
    Try fetching manual transcript for the given languages (in order) because auto-generated ones are often low-quality. 
    If no manual transcript is found, return None to let ai agent decide call other tools.
    """
    api = YouTubeTranscriptApi(http_client=session) if session else YouTubeTranscriptApi()
    manual_transcripts = api.list(video_id)._manually_created_transcripts
    if manual_transcripts and languages:
        for lang in languages:
            if lang in manual_transcripts:
                tr = manual_transcripts[lang]
                return tr.fetch()
        return api.fetch(video_id)
    return None


def fetch_transcript_raw(
    url: str,
    languages: Optional[List[str]] = None,
) -> FetchedTranscript:
    video_id = extract_video_id(url)
    langs = languages or CFG.transcript.default_languages
    cookie_files = list_cookie_files()
    attempts = cookie_files or [None]
    last_err: Optional[Exception] = None

    fetched: FetchedTranscript = None
    fetch_attempted = False
    for idx, cookie_file in enumerate(attempts):
        session = build_session(cookie_file) if cookie_file else None
        try:
            fetched = _fetch_once(video_id, langs, session)
            fetch_attempted = True
            break
        except Exception as e:
            last_err = e
            if idx < len(attempts) - 1:
                time.sleep(0.5)
            continue

    if not fetch_attempted:
        raise RuntimeError(
            f"Không fetch được transcript (đã thử {len(attempts)} cookie). Lỗi cuối: {last_err}"
        )

    return fetched

def fetch_transcript(
    url: str,
    languages: Optional[List[str]] = None,
    start_seconds: float = 0,
    end_seconds: float = 0,
) -> Optional[str]:
    raw_transcript = fetch_transcript_raw(url, languages)
    if raw_transcript is None:
        return None  # không có manual transcript, để agent xử lý

    raw = raw_transcript.to_raw_data()
    end = end_seconds if end_seconds > 0 else float("inf")

    filtered = [
        s for s in raw
        if start_seconds <= s["start"] < end
    ]

    return "\n".join(s["text"] for s in filtered)