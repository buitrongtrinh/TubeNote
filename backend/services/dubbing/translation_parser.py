"""Parser for numbered LLM translation batches."""
from __future__ import annotations

import re


def parse_translation_result(text: str) -> tuple[str, list[tuple[int, str]]]:
    text = text.strip()
    batch_match = re.search(r"\[(.*?)\]", text)
    if not batch_match:
        raise ValueError("Không tìm thấy batch")

    translations = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)\s*[\.\)\:\-]\s*(.*)$", line)
        if match:
            translations.append((int(match.group(1)), match.group(2).strip()))
    return batch_match.group(1), translations
