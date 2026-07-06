"""User glossary: dạng viết → dạng đọc cho TTS.

Ưu tiên cao nhất trong cascade chuẩn hoá — dùng cho các ngoại lệ mà rule tự
động không đoán được (acronym đọc thành từ như RAM/NASA, thuật ngữ khó đọc).
File ``data/glossary.json`` do người dùng chỉnh, đè lên BUILTIN_GLOSSARY.
"""
from __future__ import annotations

import json
from pathlib import Path

# Acronym viết in nhưng đọc thành TỪ (không đánh vần từng chữ). Nếu không có
# entry ở đây, rule tách-chữ-in sẽ đánh vần chúng sai kiểu "ar ei em".
BUILTIN_GLOSSARY: dict[str, str] = {
    "RAM": "ram",
    "ROM": "rôm",
    "VRAM": "vi ram",
    "LAN": "lan",
    "SIM": "sim",
    "PIN": "pin",
    "NASA": "na sa",
}

_CACHE: dict[str, object] = {"key": None, "value": None}


def glossary_path() -> Path:
    try:
        from backend.config import CFG
        return CFG.paths.glossary_file
    except ModuleNotFoundError as exc:  # môi trường test không có pyyaml
        if exc.name != "yaml":
            raise
    return Path("data/glossary.json")


def load_glossary() -> dict[str, str]:
    """Builtin + file người dùng (file thắng khi trùng key). Cache theo mtime."""
    path = glossary_path()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = None
    key = (str(path), mtime)
    if _CACHE["key"] == key:
        return _CACHE["value"]  # type: ignore[return-value]

    merged = dict(BUILTIN_GLOSSARY)
    if mtime is not None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                merged.update({
                    str(source).strip(): str(spoken).strip()
                    for source, spoken in raw.items()
                    if str(source).strip() and str(spoken).strip()
                })
        except (json.JSONDecodeError, OSError):
            pass  # file hỏng → dùng builtin, không chặn pipeline

    _CACHE["key"] = key
    _CACHE["value"] = merged
    return merged
