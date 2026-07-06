"""SearXNG self-hosted search client.

Container đã được orchestrate bằng ``docker-compose.yml`` ở project root:

    docker compose up -d        # khởi động lần đầu / sau reboot
    docker compose down         # tắt hẳn

Code này chỉ làm 2 việc:
    1. ``is_reachable()`` — health check
    2. ``web_search()`` — gọi ``/search?format=json``

Setup SearXNG (host JSON + tắt limiter) đã có sẵn trong
``searxng_config/settings.yml`` — không cần edit thủ công.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse

import requests

from ...config import CFG


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str

    @property
    def domain(self) -> str:
        try:
            return urlparse(self.url).netloc or self.url
        except Exception:
            return self.url


class SearXNGError(RuntimeError):
    """Daemon không reachable, JSON format chưa bật, hoặc trả format lạ."""


def is_reachable(base_url: str | None = None, timeout: float = 1.5) -> bool:
    base = base_url or CFG.web_search.base_url
    try:
        r = requests.get(f"{base}/healthz", timeout=timeout)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    # fallback nếu /healthz không có
    try:
        r = requests.get(base, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


def web_search(query: str, n: int | None = None, base_url: str | None = None) -> List[WebResult]:
    """Trả top-N kết quả. Raise ``SearXNGError`` nếu container chưa chạy."""
    base = base_url or CFG.web_search.base_url
    max_n = n if n is not None else CFG.web_search.max_results

    if not is_reachable(base, timeout=1.0):
        raise SearXNGError(
            f"SearXNG không reachable tại {base}. "
            f"Khởi động bằng: `docker compose up -d` (từ project root)."
        )

    try:
        r = requests.get(
            f"{base}/search",
            params={"q": query, "format": "json", "categories": "general"},
            timeout=10,
        )
    except requests.RequestException as e:
        raise SearXNGError(f"Lỗi gọi SearXNG: {e}") from e

    if r.status_code != 200:
        raise SearXNGError(
            f"SearXNG trả HTTP {r.status_code}. Có thể settings.yml lỗi — "
            f"check `docker compose logs searxng`."
        )

    try:
        data = r.json()
    except ValueError as e:
        raise SearXNGError("SearXNG trả non-JSON — settings.yml có thể thiếu `formats: [json]`") from e

    return [
        WebResult(
            title=x.get("title", "").strip(),
            url=x.get("url", "").strip(),
            snippet=(x.get("content", "") or "").strip()[:300],
        )
        for x in data.get("results", [])[:max_n]
        if x.get("url")
    ]


def format_for_llm(results: List[WebResult]) -> str:
    """Format ``[WEB-N]`` markers để LLM cite được."""
    if not results:
        return ""
    return "\n\n".join(
        f"[WEB-{i+1}] {r.title}\n"
        f"Domain: {r.domain}\n"
        f"URL: {r.url}\n"
        f"Snippet: {r.snippet}"
        for i, r in enumerate(results)
    )
