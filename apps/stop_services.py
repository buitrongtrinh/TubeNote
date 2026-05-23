"""Stop background services.

Usage:
    python -m apps.stop_services           # stop SearXNG container (docker compose stop)
    python -m apps.stop_services --down    # remove containers (docker compose down)
    python -m apps.stop_services --ollama  # cũng kill Ollama processes (cẩn thận!)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def stop_searxng(down: bool = False) -> bool:
    """Stop container (giữ) hoặc down (remove)."""
    cmd = ["docker", "compose", "down"] if down else ["docker", "compose", "stop"]
    try:
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            print(f"✅ {' '.join(cmd)} OK")
            return True
        print(f"❌ {r.stderr.strip() or r.stdout.strip()}")
        return False
    except FileNotFoundError:
        print("❌ Docker chưa cài / chưa có trong PATH")
        return False


def stop_ollama() -> bool:
    """Kill Ollama processes (cẩn thận: ảnh hưởng app khác dùng Ollama)."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True, check=False)
            subprocess.run(["taskkill", "/F", "/IM", "ollama app.exe"], capture_output=True, check=False)
        else:
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True, check=False)
        print("✅ Killed Ollama processes")
        return True
    except Exception as e:
        print(f"❌ Stop Ollama fail: {e}")
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Stop TubeNote background services.")
    p.add_argument("--down", action="store_true",
                   help="Remove containers (docker compose down). Mặc định chỉ stop.")
    p.add_argument("--ollama", action="store_true",
                   help="Cũng kill Ollama (cẩn thận: ảnh hưởng app khác dùng Ollama).")
    args = p.parse_args()

    stop_searxng(down=args.down)
    if args.ollama:
        stop_ollama()


if __name__ == "__main__":
    main()
