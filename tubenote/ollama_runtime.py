"""Auto-start Ollama daemon nếu chưa chạy.

Project lúc nào cũng cần Ollama (embedding lock vào ``qwen3-embedding:0.6b``),
nên check + start sớm ở entry point (Streamlit / CLI) tốt hơn là để user tự mò
``WinError 10061``.

Public API:
    - ``is_running(base_url)`` — health check ``/api/tags``
    - ``find_ollama_exe()`` — tìm ``ollama.exe`` trên Windows
    - ``ensure_running(base_url, wait_seconds)`` — block đến khi ready hoặc fail
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests


DEFAULT_BASE_URL = "http://localhost:11434"


def is_running(base_url: str = DEFAULT_BASE_URL, timeout: float = 1.0) -> bool:
    """Ping ``/api/tags``. ``True`` nếu daemon đang nghe port."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def find_ollama_exe() -> Optional[str]:
    """Tìm ``ollama.exe``. ``None`` nếu không cài."""
    # 1. PATH (nếu user thêm Ollama vào PATH)
    if (on_path := shutil.which("ollama")):
        return on_path

    # 2. Windows default install: %LOCALAPPDATA%\Programs\Ollama\ollama.exe
    local_app = os.getenv("LOCALAPPDATA")
    if local_app:
        candidate = Path(local_app) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.is_file():
            return str(candidate)

    # 3. macOS/Linux fallback (rare cho project này)
    for p in ("/usr/local/bin/ollama", "/opt/homebrew/bin/ollama"):
        if Path(p).is_file():
            return p

    return None


def _start_detached(exe_path: str) -> None:
    """Spawn ``ollama serve`` detached (Python exit không kill daemon)."""
    kwargs: dict = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        # DETACHED_PROCESS + CREATE_NO_WINDOW: không có console popup, không bị Streamlit kéo theo khi exit
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
            | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        )
    else:
        kwargs["start_new_session"] = True

    subprocess.Popen([exe_path, "serve"], **kwargs)


def ensure_running(
    base_url: str = DEFAULT_BASE_URL,
    wait_seconds: float = 15.0,
    poll_interval: float = 0.5,
) -> bool:
    """Đảm bảo daemon ready. Auto-start nếu chưa chạy.

    Returns:
        ``True`` nếu daemon ready trong ``wait_seconds``,
        ``False`` nếu chưa cài / start fail / timeout.
    """
    if is_running(base_url):
        return True

    exe = find_ollama_exe()
    if exe is None:
        return False  # chưa cài Ollama

    _start_detached(exe)

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if is_running(base_url):
            return True
        time.sleep(poll_interval)
    return False
