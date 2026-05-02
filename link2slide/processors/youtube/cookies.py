"""Cookie loading + rotation for youtube-transcript-api."""
from __future__ import annotations

import random
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from typing import List, Optional

import requests

from ...config import CFG


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
        fallback = CFG.PROJECT_ROOT / "cookies"
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
