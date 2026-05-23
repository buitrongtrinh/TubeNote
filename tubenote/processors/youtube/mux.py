"""Mux video, audio, subtitle using ffmpeg."""
from __future__ import annotations

import os
import subprocess


def mux(
    video_path: str,
    audio_path: str | None = None,
    sub_path: str | None = None,
    output_path: str = "output/final/output.mp4",
) -> str:
    """Merge video + audio + sub thành 1 file mp4.
    
    Args:
        video_path: file video-only
        audio_path: file audio-only (optional)
        sub_path:   file subtitle .srt/.ass (optional, burn cứng vào video)
        output_path: đường dẫn output
    
    Returns:
        output_path
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = ["ffmpeg", "-i", video_path]

    if audio_path:
        cmd += ["-i", audio_path]

    if sub_path:
        cmd += ["-vf", f"subtitles={sub_path}"]
        cmd += ["-c:v", "libx264"]   # phải re-encode khi burn sub
    else:
        cmd += ["-c:v", "copy"]

    cmd += ["-c:a", "copy", "-y", output_path]

    subprocess.run(cmd, check=True, capture_output=True)

    return output_path