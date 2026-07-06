"""YouTube downloader using yt-dlp."""
from __future__ import annotations

import os
from pathlib import Path

import imageio_ffmpeg
import yt_dlp

from ...config import CFG
from .cookies import get_ytdlp_cookie_opts
from .utils import extract_video_id, skip_if_exists


# Dùng ffmpeg bundle từ imageio-ffmpeg → yt-dlp không cần ffmpeg system.
# imageio đặt tên binary là ``ffmpeg-win-x86_64-v7.1.exe`` (không phải ffmpeg.exe),
# nên phải pass FULL PATH tới executable — không phải directory.
_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

_AUDIO_DIR = str(CFG.paths.audio_dir)
_VIDEO_DIR = str(CFG.paths.video_dir)


@skip_if_exists
def download_audio(
    url: str,
    output_dir: str = _AUDIO_DIR,
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
    output_dir: str = _VIDEO_DIR,
    ext: str = "mp4",
) -> str:
    """Download video-only (KHÔNG kèm audio). Dùng làm intermediate cho cả
    vietsub (combine với audio + sub) và dubbing (combine với TTS audio).
    """
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