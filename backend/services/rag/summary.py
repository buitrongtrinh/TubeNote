"""Cached video summaries for RAG chat."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import PROJECT_ROOT
from backend.services.rag.pipeline import load_local_subtitle_segments
from backend.services.youtube.utils import extract_video_id

SUMMARY_DIR = PROJECT_ROOT / "data" / "rag_summary"
MAX_TRANSCRIPT_CHARS = 28000

SUMMARY_SYSTEM_PROMPT = """\
Bạn tạo tóm tắt ngắn để dùng làm context nền cho chức năng hỏi đáp video.

Yêu cầu:
- Trả lời bằng tiếng Việt.
- Chỉ dựa trên transcript được cung cấp.
- Không bịa chi tiết ngoài video.
- UI render Markdown, được dùng **bold**; KHÔNG dùng heading ('#'), bảng, code fence (```).
- Viết 1 đoạn tổng quan ngắn, sau đó 3-5 ý chính.
- Mỗi ý chính bắt đầu bằng "- ". Không dùng bullet "*".
- Giữ nguyên tên riêng, tên sản phẩm, tên dự án tiếng Anh như transcript nếu không chắc cách viết.
- Không dịch thô các cụm như "swarm of agents"; nếu cần, dùng "nhiều agent" hoặc "nhóm agent".
- Nếu video có các phần rõ ràng, nhắc theo thứ tự xuất hiện.
"""


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _clean_summary(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)^\s*[\*\u2022]\s+", "- ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def summary_path(video_id: str) -> Path:
    return SUMMARY_DIR / f"{extract_video_id(video_id)}.json"


def load_summary(video_id: str) -> dict | None:
    path = summary_path(video_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not data.get("summary"):
        return None
    data["cached"] = True
    return data


def _format_transcript(segments: list[dict]) -> str:
    lines: list[str] = []
    total = 0
    for segment in segments:
        start = max(0, float(segment.get("start") or 0.0))
        end = max(start, float(segment.get("end") or start))
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        line = f"[{_fmt_time(start)}-{_fmt_time(end)}] {text}"
        if total + len(line) > MAX_TRANSCRIPT_CHARS:
            lines.append("[Transcript đã được cắt ngắn để vừa context.]")
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    value = max(0, int(seconds))
    minutes = value // 60
    secs = value % 60
    return f"{minutes:02d}:{secs:02d}"


def generate_summary(
    video_id: str,
    llm: BaseChatModel,
    force: bool = False,
    provider: str = "",
    model: str = "",
) -> dict:
    vid = extract_video_id(video_id)
    if not force:
        cached = load_summary(vid)
        if cached:
            return cached

    source, segments = load_local_subtitle_segments(vid)
    transcript = _format_transcript(segments)
    if not transcript:
        raise RuntimeError("Không có transcript để tóm tắt.")

    response = llm.invoke([
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Video id: {vid}\n"
            f"Nguồn subtitle: {source}\n\n"
            f"Transcript:\n{transcript}"
        )),
    ])
    summary = _clean_summary(_extract_text(response))
    if not summary:
        raise RuntimeError("LLM không trả về tóm tắt.")

    payload = {
        "video_id": vid,
        "summary": summary,
        "source": source,
        "segments": len(segments),
        "provider": provider,
        "model": model,
        "created_at": time.time(),
        "cached": False,
    }
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    with summary_path(vid).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload
