"""Whisper STT (Speech-To-Text) fallback — audio → Transcript với fine-grained segments.

Route theo language detected từ metadata video:
  - 'en' → ``base.en`` (English-only, WER ~6%)
  - 'vi' → ``base`` multilingual (handle code-switching VN+EN, giữ "LLM"
           không transliterate như PhoWhisper)

Dùng openai-whisper library (không phải HF transformers) vì:
  - Native long-form chunking → timestamps fine-grained cho vietsub
  - ``initial_prompt`` support đơn giản
  - Cache nhẹ ở ``%USERPROFILE%\\.cache\\whisper\\`` (.pt single file)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from .dowload import download_audio
from .types import Transcript, TranscriptEntry
from .utils import extract_video_id, skip_if_exists


SUBTITLES_RAW_DIR = "output/subtitles_raw"
METADATA_DIR = "output/metadata"
AUDIO_DIR = "output/audio"

EN_MODEL = "small.en"
VI_MODEL = "base"

INITIAL_PROMPT = (
    "Video kỹ thuật về AI, machine learning, prompt engineering. "
    "Các thuật ngữ thường gặp: LLM, MCP, API, RAG, embedding, prompt, "
    "vector database, fine-tune, transformer, Claude, Gemini, ChatGPT, "
    "GPT-4, Llama, OpenAI, Anthropic, Hugging Face, npm, Python, Docker."
)

# VN-specific characters (đ + diacritics, marker tiếng Việt)
_VN_CHARS = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯ]"
    r"|[àáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]"
    r"|[ÀÁẠẢÃẰẮẶẲẴẦẤẬẨẪÈÉẸẺẼỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕỒỐỘỔỖỜỚỢỞỠÙÚỤỦŨỪỨỰỬỮỲÝỴỶỸ]"
)

_MODELS: dict[str, object] = {}


def _detect_language(title: str = "", channel: str = "") -> str:
    """VN nếu title HOẶC channel có diacritic VN; mặc định 'en'."""
    blob = f"{title} {channel}"
    return "vi" if _VN_CHARS.search(blob) else "en"


def _read_cached_metadata(video_id: str) -> dict:
    path = Path(METADATA_DIR) / f"{video_id}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_model(language: str):
    """Lazy-load Whisper model. Cache theo language. GPU auto-detect."""
    if language in _MODELS:
        return _MODELS[language]

    import torch
    import whisper

    name = VI_MODEL if language == "vi" else EN_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[whisper] Loading {name} on {device}…", flush=True)
    _MODELS[language] = whisper.load_model(name, device=device)
    return _MODELS[language]


def _segments_to_transcript(segments: list[dict]) -> Optional[Transcript]:
    entries: list[TranscriptEntry] = []
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        start = float(s.get("start") or 0.0)
        end = float(s.get("end") or start)
        entries.append(TranscriptEntry(
            text=text,
            start=start,
            duration=max(0.0, end - start),
        ))
    return Transcript(entries) if entries else None


@skip_if_exists
def fetch_transcript(
    url: str,
    output_dir: str = SUBTITLES_RAW_DIR,
    ext: str = "json",
    language: Optional[str] = None,
) -> Optional[str]:
    """Download audio (skip nếu có) → Whisper transcribe → save json.

    Routing model:
      - ``language='en'`` → ``base.en``
      - ``language='vi'`` → ``base`` multilingual
      - ``language=None`` → auto-detect từ metadata cache.

    Output: list segments với start/duration fine-grained (suitable cho vietsub).
    """
    video_id = extract_video_id(url)
    download_audio(url=url)

    audio_path = None
    for audio_ext in ["mp3", "m4a"]:
        candidate = os.path.join(AUDIO_DIR, f"{video_id}.{audio_ext}")
        if os.path.exists(candidate):
            audio_path = candidate
            break
    if audio_path is None:
        return None

    if language is None:
        meta = _read_cached_metadata(video_id)
        language = _detect_language(meta.get("title", ""), meta.get("channel", ""))
        print(f"[whisper] Auto-detected language: {language!r}", flush=True)

    kwargs = {
        "initial_prompt": INITIAL_PROMPT,
        "condition_on_previous_text": False,
        "verbose": False,
    }
    if language == "vi":
        kwargs["language"] = "vi"  # base.en tự biết English, không cần truyền

    result = _get_model(language).transcribe(audio_path, **kwargs)
    trans = _segments_to_transcript(result.get("segments", []))
    if trans is None:
        return None
    return trans.save_json(video_id=video_id, folder=output_dir)
