"""OmniVoice TTS adapter for video dubbing.

The import is intentionally lazy in ``init_omnivoice`` so the default
Supertonic flow keeps working without OmniVoice/Torch installed.
"""
from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Callable

from backend.services.dubbing.audio_fit import active_range_samples, fit_to_slot

OMNIVOICE_SAMPLE_RATE = 24000
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def release_omnivoice_models() -> None:
    """Release cached OmniVoice weights before another GPU model is loaded."""
    import gc

    _MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
        raise RuntimeError("Cần cài imageio-ffmpeg hoặc ffmpeg để trích giọng từ video.")


def _segment_text(item: dict) -> str:
    """Text khớp với NGÔN NGỮ GỐC của audio (``text``) — reference audio được
    cắt trực tiếp từ audio nguồn nên transcript phải cùng ngôn ngữ với nó, ưu
    tiên hơn các field tiếng Việt (``text_tts``/``text_vi``/``source_texts``)
    chỉ dùng làm fallback khi entry không có ``text`` (vd data đã merge)."""
    text = item.get("text") or item.get("text_tts") or item.get("text_vi")
    if not text and isinstance(item.get("source_texts"), list):
        text = " ".join(str(t) for t in item["source_texts"] if t)
    return str(text or "").strip()


def pick_reference_window(data: list[dict]) -> tuple[float, float, str]:
    """Chọn 1 cửa sổ [start, start+duration] 3-10s có đủ chữ để làm giọng mẫu.

    ``data`` có thể là subtitle gốc (key ``text``) hoặc chunk đã merge cho TTS
    (key ``text_tts``/``source_texts``) — ``_segment_text`` đọc đúng cả 2.
    """
    candidates: list[tuple[float, float, float, str]] = []
    for start_idx, first in enumerate(data):
        start = float(first.get("start", 0.0))
        texts: list[str] = []
        previous_end = start
        total_gap = 0.0
        for item in data[start_idx:start_idx + 8]:
            item_start = float(item.get("start", previous_end))
            item_end = item_start + max(0.0, float(item.get("duration", 0.0)))
            if texts and item_start - previous_end > 1.25:
                break
            if texts:
                total_gap += max(0.0, item_start - previous_end)
            text = _segment_text(item)
            if text:
                texts.append(text)
            previous_end = max(previous_end, item_end)
            duration = previous_end - start
            if duration > 10.0:
                break
            transcript = " ".join(texts)
            letter_count = sum(char.isalpha() for char in transcript)
            if 3.0 <= duration <= 10.0 and letter_count >= 24:
                score = abs(duration - 7.0) + total_gap * 2.0
                candidates.append((score, start, duration, transcript))

    if not candidates:
        # Không có tổ hợp nào gọn trong [3,10]s — có thể vì mọi segment đều dài
        # hơn 10s (vd sentence_max_words đang tắt/lớn, không còn cắt cứng theo
        # số từ). Lấy segment đầu tiên đủ chữ, cắt bớt còn ~7s thay vì bó tay;
        # transcript ước lượng theo tỉ lệ ký tự (không có word-timestamp ở đây
        # để cắt chính xác hơn, nhưng ref_text chỉ là hint cho model clone).
        for item in data:
            duration = max(0.0, float(item.get("duration", 0.0)))
            text = _segment_text(item)
            if duration >= 3.0 and sum(char.isalpha() for char in text) >= 24:
                target = min(duration, 7.0)
                if text and target < duration:
                    keep = max(1, int(len(text) * (target / duration)))
                    text = text[:keep].rsplit(" ", 1)[0] or text[:keep]
                candidates.append((0.0, float(item.get("start", 0.0)), target, text))
                break

    if not candidates:
        raise ValueError("Không tìm được đoạn thoại 3–10 giây để clone giọng từ video.")

    _, start, duration, transcript = min(candidates, key=lambda item: item[0])
    return start, duration, transcript


def prepare_source_voice_reference(
    data: list[dict],
    source_audio: str,
    output_path: str,
) -> tuple[str, str]:
    """Extract a clean-sized reference window using source subtitle timings."""
    start, duration, transcript = pick_reference_window(data)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _ffmpeg_executable(),
        "-y", "-v", "error",
        "-ss", f"{start:.3f}",
        "-i", str(source_audio),
        "-t", f"{duration:.3f}",
        "-vn", "-ac", "1", "-ar", str(OMNIVOICE_SAMPLE_RATE),
        "-c:a", "pcm_s16le", str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return str(destination), transcript


def _best_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_device(device: str | None) -> str:
    if not device or device == "auto":
        return _best_device()
    if device == "gpu":
        return "cuda"
    return device


def init_omnivoice(model_name: str, device: str | None = None):
    """Load and cache an OmniVoice model."""
    resolved_device = _resolve_device(device)
    key = (model_name, resolved_device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], resolved_device

    try:
        import torch
        from omnivoice import OmniVoice
    except ImportError as exc:
        raise RuntimeError(
            "OmniVoice chưa được cài. Cài dependency trước khi chọn engine này: "
            "python -m pip install -r requirements.txt"
        ) from exc

    dtype = torch.float32 if resolved_device == "cpu" else torch.float16
    model = OmniVoice.from_pretrained(model_name, device_map=resolved_device, dtype=dtype)
    _MODEL_CACHE[key] = model
    return model, resolved_device


def _slot_seconds(data: list[dict], index: int) -> float:
    """Return the fixed TTS slot for an already-split/merged segment."""
    item = data[index]
    return max(0.2, float(item.get("duration", 0.0)))


def _generation_slot_seconds(
    data: list[dict],
    index: int,
    *,
    alpha: float,
    min_delta: float,
) -> float:
    """Return the duration passed to OmniVoice generate().

    Thay vì cộng cứng 1 hằng số cho mọi câu (câu ngắn bị cấp dư quá nhiều →
    model tự giãn ra nói chậm/nhiều khoảng nghỉ, phải cắt lặng rất mạnh mới về
    đúng slot), chỉ cộng thêm đúng phần "dư" thực sự cần — ước tính từ độ dài
    thật của ``text_tts`` so với slot, theo tốc độ nói tự nhiên
    (``duration_budget.natural_duration_seconds``). ``alpha`` là hệ số an toàn
    (>1, khuyến nghị ~1.2) vì bản thân ước tính đã có sai số cộng dồn (đếm âm
    tiết + tốc độ nói trung bình); dư giây được ``fit_to_slot`` cắt lặng êm,
    còn thiếu giây dễ ép model nói nhanh/mất chữ — nên thiên về hướng dư nhẹ.
    ``min_delta`` là sàn cho câu đã đủ/thưa ngân sách (không có "dư" để cộng).
    """
    from backend.services.dubbing.duration_budget import natural_duration_seconds

    slot = _slot_seconds(data, index)
    text_tts = (data[index].get("text_tts") or "").strip()
    natural = natural_duration_seconds(text_tts) if text_tts else 0.0
    excess = max(0.0, natural - slot)
    delta = max(float(min_delta or 0.0), excess * float(alpha or 1.0))
    return max(0.2, slot + delta)


def _chunks(items: list[tuple[int, dict]], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def synthesize_omnivoice(
    data: list[dict],
    output_path: str,
    config: dict,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Generate a full dubbed WAV aligned to source segment starts."""
    import numpy as np
    import soundfile as sf

    if not data:
        return []

    model_name = str(config.get("model") or "k2-fsa/OmniVoice")
    model, resolved_device = init_omnivoice(model_name, config.get("device"))
    sample_rate = int(getattr(model, "sampling_rate", OMNIVOICE_SAMPLE_RATE))

    last = data[-1]
    total_samples = int((float(last["start"]) + float(last["duration"])) * sample_rate)
    full_audio = np.zeros(total_samples, dtype=np.float32)
    total = len(data)
    timings: list[dict] = []

    voice_mode = config.get("voice_mode") or "default"
    ref_audio = config.get("reference_audio_id") or config.get("ref_audio") or None
    ref_text = config.get("reference_text") or config.get("ref_text") or None
    instruct = (config.get("instruction") or "").strip() or None
    language = config.get("language") or "vi"
    num_step = int(config.get("num_step") or 32)
    postprocess_output = bool(config.get("postprocess_output", False))
    batch_size = max(1, int(config.get("batch_size") or 1))
    raw_wsola_limit = config.get("wsola_limit")
    wsola_limit = float(1.00 if raw_wsola_limit is None else raw_wsola_limit)
    fit_audio = bool(config.get("fit_audio", True))
    raw_delta_alpha = config.get("generation_delta_alpha")
    generation_delta_alpha = float(1.2 if raw_delta_alpha is None else raw_delta_alpha)
    raw_delta_min = config.get("generation_delta_min")
    generation_delta_min = float(0.3 if raw_delta_min is None else raw_delta_min)

    if voice_mode == "clone":
        if not ref_audio:
            raise ValueError("OmniVoice clone cần reference audio.")
        if not Path(ref_audio).exists():
            raise ValueError(f"Không tìm thấy reference audio cho OmniVoice: {ref_audio}")
    clone_prompt = None
    if voice_mode == "clone" and ref_audio:
        clone_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            preprocess_prompt=bool(config.get("preprocess_prompt", True)),
        )

    items = [(idx, item) for idx, item in enumerate(data) if (item.get("text_tts") or "").strip()]
    done = 0
    for batch in _chunks(items, batch_size):
        texts = [(item.get("text_tts") or "").strip() for _, item in batch]
        target_durations = [_slot_seconds(data, idx) for idx, _ in batch]
        generation_durations = [
            _generation_slot_seconds(
                data, idx, alpha=generation_delta_alpha, min_delta=generation_delta_min,
            )
            for idx, _ in batch
        ]
        kwargs = {
            "text": texts,
            "language": [language] * len(batch),
            "duration": generation_durations,
            "num_step": num_step,
            "postprocess_output": postprocess_output,
        }
        if voice_mode == "clone" and clone_prompt is not None:
            kwargs["voice_clone_prompt"] = [clone_prompt] * len(batch)
            if instruct:
                kwargs["instruct"] = [instruct] * len(batch)
        elif voice_mode == "design" and instruct:
            kwargs["instruct"] = [instruct] * len(batch)
        elif instruct:
            kwargs["instruct"] = [instruct] * len(batch)

        audios = model.generate(**kwargs)

        for (idx, item), target_duration, generation_duration, audio in zip(
            batch,
            target_durations,
            generation_durations,
            audios,
        ):
            audio = audio.astype(np.float32)
            target_len = int(target_duration * sample_rate)
            if fit_audio:
                audio, speech_len, fit_meta = fit_to_slot(
                    audio,
                    target_len,
                    sample_rate,
                    wsola_limit=wsola_limit,
                )
            else:
                fit_meta = {
                    "enabled": False,
                    "raw_duration": round(len(audio) / sample_rate, 3),
                    "target_duration": round(target_len / sample_rate, 3),
                    "silence_cut_duration": 0.0,
                    "wsola_ratio": 1.0,
                    "atempo_ratio": 1.0,
                    "fit_ratio": round(len(audio) / max(1, target_len), 3),
                    "trimmed_duration": 0.0,
                    "warnings": ["Fit audio đang tắt; audio raw có thể lệch slot."],
                }
            fit_meta["generation_duration"] = round(generation_duration, 3)
            fit_meta["generation_duration_delta"] = round(generation_duration - target_duration, 3)
            active_start, active_end = active_range_samples(audio, sample_rate)
            speech_len = max(0, active_end - active_start)
            fit_meta["active_start"] = round(active_start / sample_rate, 3)
            fit_meta["active_end"] = round(active_end / sample_rate, 3)

            start_sample = int(float(item["start"]) * sample_rate)
            required_len = start_sample + len(audio)
            if required_len > len(full_audio):
                full_audio = np.pad(full_audio, (0, required_len - len(full_audio)), mode="constant")
            end = min(start_sample + len(audio), len(full_audio))
            full_audio[start_sample:end] += audio[:end - start_sample]

            timings.append({
                "index": idx,
                "source_indices": item.get("source_indices", [idx]),
                "start": float(item["start"]) + active_start / sample_rate,
                "speech_duration": round(speech_len / sample_rate, 3),
                "fit": fit_meta,
            })

            done += 1
            if on_progress:
                on_progress(done, total)

    peak = float(np.max(np.abs(full_audio))) if full_audio.size else 0.0
    if peak > 0.97:
        full_audio *= 0.97 / peak

    sf.write(output_path, full_audio, sample_rate)
    config["device"] = resolved_device
    config["postprocess_output"] = postprocess_output
    config["fit_audio"] = fit_audio
    return timings
