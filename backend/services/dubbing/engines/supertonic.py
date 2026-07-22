"""Supertonic TTS adapter for video dubbing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.config import CFG
from backend.services.dubbing.audio_fit import active_range_samples, fit_to_slot

if TYPE_CHECKING:
    from supertonic import TTS

# Dải speed KHUYẾN NGHỊ của nhà phát triển Supertonic (py/README.md: "Recommended
# speed range is between 0.9 and 1.5 for natural-sounding results"). API vẫn nhận
# 0.7-2.0, nhưng ngoài dải này mô hình dễ mất ổn định khi đọc — maintainer xác
# nhận lỗi nuốt/lặp từ phụ thuộc tổ hợp "text / voice / seed / speed", và người
# dùng báo nuốt từ ở speed chậm 0.7-0.8.
#
# Ép sát trần không mất gì: phần nén còn lại đã có WSOLA/atempo trong
# fit_to_slot gánh, đúng như ý đồ ghi trong adaptive_speed ("nhường cho
# WSOLA/atempo... tránh ép TTS sát nút ngay từ lúc tổng hợp").
TTS_SPEED_MIN = 0.9
TTS_SPEED_MAX = 1.5


def _speech_slot(data: list, idx: int, sample_rate: int) -> int:
    """Sample count reserved for speech, matching the source segment duration."""
    item = data[idx]
    slot = item["duration"]
    return int(max(0.1, slot) * sample_rate)


def adaptive_speed(text: str, slot_seconds: float, alpha: float) -> float:
    """Tốc độ nói cho 1 segment, thích ứng theo độ dài text thật thay vì hằng số.

    Câu vừa/thưa slot (không dư) → speed = 1.0, không ép nhanh vô cớ. Câu vượt
    ngân sách → speed tăng đúng theo tỉ lệ dư thực sự (ước tính bằng
    ``natural_duration_seconds``), giảm nhẹ bởi ``alpha`` (>1, khuyến nghị
    ~1.2): chỉ yêu cầu Supertonic nói nhanh gần đủ mức cần, phần dư còn lại
    nhường cho WSOLA/atempo trong ``fit_to_slot`` (đã xác nhận đủ sức nén bất
    kỳ tỉ lệ nào) — tránh ép TTS sát nút ngay từ lúc tổng hợp.
    """
    from backend.services.dubbing.duration_budget import natural_duration_seconds

    slot_seconds = max(0.1, float(slot_seconds or 0.0))
    ratio_needed = natural_duration_seconds(text) / slot_seconds if text else 1.0
    return 1.0 + max(0.0, ratio_needed - 1.0) / max(1.0, float(alpha or 1.0))


def text_to_speech(
    data: list,
    output_path: str,
    tts: "TTS",
    style: str,
    speed_alpha: float = CFG.tts.supertonic.speed_alpha,
    total_steps: int = CFG.tts.supertonic.num_step,
    wsola_limit: float = CFG.tts.supertonic.wsola_limit,
    on_progress=None,
):
    """Generate Supertonic speech and place each segment at its source timing."""
    import numpy as np

    if not data:
        return []

    sample_rate = tts.sample_rate
    last = data[-1]
    total_samples = int((last["start"] + last["duration"]) * sample_rate)
    full_audio = np.zeros(total_samples, dtype=np.float32)
    total = len(data)

    total_steps = max(1, min(100, int(total_steps or 8)))

    def synth(text: str, spd: float):
        spd = max(TTS_SPEED_MIN, min(TTS_SPEED_MAX, spd))
        wav, _ = tts.synthesize(
            text=text,
            voice_style=style,
            lang="vi",
            total_steps=total_steps,
            speed=spd,
            max_chunk_length=1000,
            silence_duration=0.05,
        )
        return wav.flatten()

    timings = []

    for idx, item in enumerate(data):
        print(f"Processing {idx + 1}/{total}", end="\r")
        if on_progress:
            on_progress(idx + 1, total)
        text = item.get("text_tts", "")
        if not text:
            continue

        slot = _speech_slot(data, idx, sample_rate)
        spd = adaptive_speed(text, item.get("duration", 0.0), speed_alpha)
        wav, speech_len, fit_meta = fit_to_slot(
            synth(text, spd),
            slot,
            sample_rate,
            wsola_limit=wsola_limit,
        )
        active_start, active_end = active_range_samples(wav, sample_rate)
        speech_len = max(0, active_end - active_start)
        fit_meta["active_start"] = round(active_start / sample_rate, 3)
        fit_meta["active_end"] = round(active_end / sample_rate, 3)
        fit_meta["speed"] = round(spd, 3)
        timings.append({
            "index": idx,
            "source_indices": item.get("source_indices", [idx]),
            "start": float(item["start"]) + active_start / sample_rate,
            "speech_duration": round(speech_len / sample_rate, 3),
            "fit": fit_meta,
        })

        start_sample = int(item["start"] * sample_rate)
        required_len = start_sample + len(wav)
        if required_len > len(full_audio):
            full_audio = np.pad(full_audio, (0, required_len - len(full_audio)), mode="constant")
        end = min(start_sample + len(wav), len(full_audio))
        full_audio[start_sample:end] += wav[:end - start_sample]

    peak = float(np.max(np.abs(full_audio))) if full_audio.size else 0.0
    if peak > 0.97:
        full_audio *= 0.97 / peak

    tts.save_audio(full_audio, output_path)
    return timings
