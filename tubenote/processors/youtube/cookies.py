"""Cookie loading + rotation cho youtube-transcript-api + helper cho yt-dlp."""
from __future__ import annotations

import os
import random
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from typing import List, Optional

import requests

from ...config import CFG, PROJECT_ROOT


def _load_jar(path: Path) -> MozillaCookieJar:
    jar = MozillaCookieJar(str(path))
    jar.load(ignore_discard=True, ignore_expires=True)
    return jar


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


def list_available_browsers() -> List[str]:
    """Detect browsers có profile data trên Windows (yt-dlp-compatible names).

    Trả về list theo thứ tự phổ biến — caller dùng để show dropdown.
    """
    candidates: List[tuple[str, Path]] = [
        ("chrome",  Path(os.getenv("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"),
        ("edge",    Path(os.getenv("LOCALAPPDATA", "")) / "Microsoft" / "Edge" / "User Data"),
        ("brave",   Path(os.getenv("LOCALAPPDATA", "")) / "BraveSoftware" / "Brave-Browser" / "User Data"),
        ("vivaldi", Path(os.getenv("LOCALAPPDATA", "")) / "Vivaldi" / "User Data"),
        ("firefox", Path(os.getenv("APPDATA", "")) / "Mozilla" / "Firefox" / "Profiles"),
        ("opera",   Path(os.getenv("APPDATA", "")) / "Opera Software" / "Opera Stable"),
    ]
    return [name for name, path in candidates if path.exists()]


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


def build_session(cookie_file: Optional[Path]) -> Optional[requests.Session]:
    if not cookie_file:
        return None
    session = requests.Session()
    session.cookies = _load_jar(cookie_file)
    ua_pool = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    ]
    session.headers["User-Agent"] = random.choice(ua_pool)
    return session
