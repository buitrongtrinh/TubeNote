"""YouTube downloader using yt-dlp."""
from __future__ import annotations

import yt_dlp
import os
from .cookies import get_ytdlp_cookie_opts
from .utils import extract_video_id, skip_if_exists

@skip_if_exists
def download_audio(
    url: str,
    output_dir: str = "output/audio",
    ext: str = "m4a",
)->str:
    """Download audio (m4a) từ URL. Save thành ``{output_dir}/{video_id}.m4a``."""
    
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": f"{output_dir}/%(id)s.%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        **get_ytdlp_cookie_opts(),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    video_id = extract_video_id(url)
    return os.path.join(output_dir, f"{video_id}.{ext}")

@skip_if_exists
def download_video(
    url: str,
    output_dir: str = "output/video",
    ext: str = "mp4",
):
    """Download video-only (không audio) từ URL. Save thành ``{output_dir}/{video_id}.mp4``."""

    ydl_opts = {
        "format": "bestvideo[ext=mp4]/bestvideo",
        "outtmpl": f"{output_dir}/%(id)s.%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        **get_ytdlp_cookie_opts(),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    video_id = extract_video_id(url)
    return os.path.join(output_dir, f"{video_id}.{ext}")