from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List
import os
import json


@dataclass
class TranscriptEntry:
    text: str
    start: float
    duration: float


@dataclass
class Transcript:
    segments: List[TranscriptEntry]

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)

    def save_json(self, video_id: str, folder: str = "data/subtitles") -> str:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{video_id}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in self.segments], f, ensure_ascii=False, indent=2)

        return path

    @classmethod
    def load_from_json(cls, path: str) -> "Transcript":
        with open(path, encoding="utf-8") as f:
            segments = json.load(f)
        return cls([
            TranscriptEntry(text=s["text"], start=s["start"], duration=s["duration"])
            for s in segments
        ])
