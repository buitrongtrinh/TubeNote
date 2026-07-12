"""Cookie loading helpers for yt-dlp."""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import List, Optional

from yt_dlp.cookies import extract_cookies_from_browser

from ...config import CFG, PROJECT_ROOT

# Thứ tự dò trình duyệt khi YT_COOKIES_BROWSER không được set — ưu tiên các
# trình duyệt phổ biến nhất trước. "safari" bỏ qua trên non-macOS ở dưới.
_BROWSER_PROBE_ORDER = (
    "chrome", "edge", "firefox", "brave", "chromium", "opera", "vivaldi", "whale",
)
# Cookie này chỉ xuất hiện trên domain youtube.com khi trình duyệt đang đăng
# nhập Google/YouTube thật — dùng làm tín hiệu "trình duyệt này dùng được".
_LOGIN_INDICATOR_COOKIE = "LOGIN_INFO"


@functools.lru_cache(maxsize=1)
def _detect_logged_in_browser() -> Optional[str]:
    """Dò các trình duyệt cài trên máy, trả về tên trình duyệt đầu tiên có
    cookie đăng nhập YouTube hợp lệ. Chỉ đọc cookie DB local (rẻ, không gọi
    mạng) nên an toàn để thử nhiều trình duyệt. Cache theo process — trình
    duyệt dùng được không đổi giữa các lần gọi trong cùng 1 lần chạy backend.
    """
    for browser in _BROWSER_PROBE_ORDER:
        try:
            jar = extract_cookies_from_browser(browser)
        except Exception:
            continue
        if any(c.name == _LOGIN_INDICATOR_COOKIE and "youtube.com" in c.domain for c in jar):
            return browser
    return None


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

# Khi có cookie đăng nhập hợp lệ, yt-dlp chọn client "tv" (cần đăng nhập, đổi
# lại lấy được format đầy đủ) — client này bắt buộc giải "n challenge" bằng
# JS, nếu không giải được thì mọi format cần giải sẽ bị loại bỏ (chỉ còn
# storyboard). Mặc định yt-dlp chỉ thử runtime "deno" (thường không có sẵn)
# và không tự tải script giải mã. Bật thêm "node" (phổ biến, hay có sẵn) +
# cho phép tải remote component chính thức của yt-dlp (ejs:github) để giải
# quyết việc này khi cookie thật sự được dùng.
_JS_CHALLENGE_OPTS = {
    "js_runtimes": {"deno": {}, "node": {}},
    "remote_components": ["ejs:github"],
}


def get_ytdlp_cookie_opts() -> dict:
    """Return yt-dlp options dict cho cookies (để bypass bot detection).

    Priority:
        1. ``YT_COOKIES_BROWSER`` env (vd: ``chrome``/``firefox``/``edge``) →
           ``cookiesfrombrowser`` (đọc trực tiếp từ browser đang chạy, tiện nhất).
        2. Không set ``YT_COOKIES_BROWSER`` → tự dò các trình duyệt phổ biến
           cài trên máy, dùng trình duyệt đầu tiên có cookie đăng nhập YouTube
           hợp lệ (xem ``_detect_logged_in_browser``).
        3. ``CFG.cookies.single_file`` → ``cookiefile``.
        4. File ``.txt`` đầu tiên trong ``CFG.cookies.dir`` → ``cookiefile``.
        5. Dict rỗng (best-effort, không cookies).

    Khi có cookie (case 1-4), kèm theo cấu hình giải n-challenge (xem
    ``_JS_CHALLENGE_OPTS``) vì cookie hợp lệ khiến yt-dlp chọn client cần
    giải challenge đó.
    """
    browser = os.getenv("YT_COOKIES_BROWSER", "").strip().lower()
    if not browser:
        browser = _detect_logged_in_browser() or ""
    if browser:
        # yt-dlp: ("chrome",) hoặc ("chrome", "Default") nếu cần profile cụ thể
        return {"cookiesfrombrowser": (browser,), **_JS_CHALLENGE_OPTS}

    files = list_cookie_files()
    if files:
        return {"cookiefile": str(files[0]), **_JS_CHALLENGE_OPTS}

    return {}
