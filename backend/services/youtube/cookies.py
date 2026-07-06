"""Cookie loading helpers for yt-dlp."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from ...config import CFG, PROJECT_ROOT


def list_cookie_files() -> List[Path]:
    files: List[Path] = []
    if CFG.cookies.single_file and CFG.cookies.single_file.is_file():
        files.append(CFG.cookies.single_file)

    # Resolve cookies dir: ưu tiên CFG, fallback về ./cookies/ nếu folder tồn tại + có .txt
    cookies_dir = CFG.cookies.dir
    if cookies_dir is None:
        fallback = PROJECT_ROOT / "cookies"
        if fallback.is_dir() and any(fallback.glob("*.txt")):
            cookies_dir = fallback

    if cookies_dir and cookies_dir.is_dir():
        files.extend(sorted(p for p in cookies_dir.glob("*.txt") if p.is_file()))

    seen = set()
    out: List[Path] = []
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out

def get_ytdlp_cookie_opts() -> dict:
    """Return yt-dlp options dict cho cookies (để bypass bot detection).

    Priority:
        1. ``YT_COOKIES_BROWSER`` env (vd: ``chrome``/``firefox``/``edge``) →
           ``cookiesfrombrowser`` (đọc trực tiếp từ browser đang chạy, tiện nhất).
        2. ``CFG.cookies.single_file`` → ``cookiefile``.
        3. File ``.txt`` đầu tiên trong ``CFG.cookies.dir`` → ``cookiefile``.
        4. Dict rỗng (best-effort, không cookies).
    """
    browser = os.getenv("YT_COOKIES_BROWSER", "").strip().lower()
    if browser:
        # yt-dlp: ("chrome",) hoặc ("chrome", "Default") nếu cần profile cụ thể
        return {"cookiesfrombrowser": (browser,)}

    files = list_cookie_files()
    if files:
        return {"cookiefile": str(files[0])}

    return {}
