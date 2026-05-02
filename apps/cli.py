"""CLI entrypoint.

Usage:
    python -m apps.cli <youtube_url> [instruction...]
    python -m apps.cli <url> --provider google --model gemini-2.5-flash
    python -m apps.cli <url> --quiet                # tắt log progress
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from link2slide.agents.summarizer import summarize


# ANSI color helpers — auto-disable nếu stderr không phải TTY
def _supports_color() -> bool:
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


_USE_COLOR = _supports_color()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


DIM = lambda t: _c("2", t)
BOLD = lambda t: _c("1", t)
CYAN = lambda t: _c("36", t)
GREEN = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
MAGENTA = lambda t: _c("35", t)


def _truncate(s: str, n: int = 120) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def _print_event(event: dict) -> None:
    """In event ra stderr theo style giống Claude Code."""
    t = event.get("type")
    if t == "agent_start":
        print(DIM("● Agent khởi động..."), file=sys.stderr)
    elif t == "tool_call":
        args_str = json.dumps(event["args"], ensure_ascii=False)
        print(
            f"{CYAN('⚙')}  {BOLD(event['name'])}{DIM('(')}{DIM(_truncate(args_str, 100))}{DIM(')')}",
            file=sys.stderr,
        )
    elif t == "tool_result":
        name = event["name"]
        chars = len(event["content"])
        preview = _truncate(event["content"], 80)
        print(
            f"   {GREEN('└')} {DIM(f'{name} → {chars} chars')}  {DIM(preview)}",
            file=sys.stderr,
        )
    elif t == "repair":
        print(YELLOW("⚠  Phát hiện chữ Hán → đang dịch lại sang tiếng Việt..."), file=sys.stderr)
    elif t == "done":
        print(MAGENTA("● Hoàn tất.\n"), file=sys.stderr)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tóm tắt video YouTube bằng LLM.")
    p.add_argument("url", help="Link YouTube hoặc 11-char video ID")
    p.add_argument(
        "instruction",
        nargs="*",
        help="Chỉ dẫn bổ sung, vd: 'Chỉ tóm tắt 5 phút đầu'",
    )
    p.add_argument("--provider", help="google | openai | anthropic")
    p.add_argument("--model", help="Override model name")
    p.add_argument("--api-key", dest="api_key", help="Override API key")
    p.add_argument("--quiet", "-q", action="store_true", help="Tắt log progress")
    return p.parse_args()


def main() -> None:
    args = _parse()
    instruction = " ".join(args.instruction)
    on_event = None if args.quiet else _print_event

    summary = asyncio.run(
        summarize(
            args.url,
            instruction,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            on_event=on_event,
        )
    )
    print("\n===== TÓM TẮT =====\n")
    print(summary)


if __name__ == "__main__":
    main()
