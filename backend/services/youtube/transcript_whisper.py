"""Whisper STT fallback — audio → Transcript với fine-grained segments.

Mặc định dùng faster-whisper/CTranslate2 theo preset trong config để chạy nhanh
hơn trên GPU/CPU. OpenAI Whisper vẫn được giữ làm fallback cấu hình.
"""
from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path
from typing import Callable, Optional

from ...config import CFG
from ..gpu_runtime import GPU_MODEL_LOCK
from .download import download_audio
from .types import Transcript, TranscriptEntry
from .utils import extract_video_id, skip_if_exists


SUBTITLES_DIR = str(CFG.paths.subtitles_dir)
METADATA_DIR = str(CFG.paths.metadata_dir)
AUDIO_DIR = str(CFG.paths.audio_dir)

EN_MODEL = CFG.whisper.en_model
VI_MODEL = CFG.whisper.vi_model
INITIAL_PROMPT = CFG.whisper.initial_prompt

# VN-specific characters (đ + diacritics, marker tiếng Việt)
_VN_CHARS = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯ]"
    r"|[àáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]"
    r"|[ÀÁẠẢÃẰẮẶẲẴẦẤẬẨẪÈÉẸẺẼỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕỒỐỘỔỖỜỚỢỞỠÙÚỤỦŨỪỨỰỬỮỲÝỴỶỸ]"
)

_MODELS: dict[str, object] = {}
_MODEL_ALIASES = {
    "openai/whisper-large-v3-turbo": "turbo",
    "whisper-large-v3-turbo": "turbo",
    "large-v3-turbo": "turbo",
}
_FASTER_MODEL_ALIASES = {
    **_MODEL_ALIASES,
    "turbo": "turbo",
    "small.en": "small.en",
    "medium.en": "medium.en",
}


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


def _preset_config(preset: str | None, language: str | None = None) -> dict:
    preset_id = preset or CFG.whisper.default_preset or "gpu"
    presets = CFG.whisper.presets or {}
    cfg = dict(presets.get(preset_id) or {})
    if not cfg:
        cfg = {
            "engine": CFG.whisper.engine,
            "model": EN_MODEL if language == "en" else VI_MODEL,
            "device": "cuda",
            "compute_type": "float16",
            "batch_size": 1,
            "beam_size": 5,
            "language": language,
        }
    cfg["id"] = preset_id
    cfg["engine"] = str(cfg.get("engine") or CFG.whisper.engine or "faster")
    if language and not cfg.get("language"):
        cfg["language"] = language
    return cfg


def _get_openai_model(language: str):
    """Load Whisper Turbo on CUDA after releasing any cached TTS model."""
    key = f"openai:{language}"
    if key in _MODELS:
        return _MODELS[key]

    import torch
    import whisper

    if not torch.cuda.is_available():
        raise RuntimeError("Whisper large-v3-turbo cần CUDA nhưng không tìm thấy GPU khả dụng.")

    from backend.services.dubbing.engines.omnivoice import release_omnivoice_models
    release_omnivoice_models()

    configured_name = VI_MODEL if language == "vi" else EN_MODEL
    name = _MODEL_ALIASES.get(configured_name.lower(), configured_name)
    device = "cuda"
    print(f"[whisper] Loading {name} on {device}…", flush=True)
    _MODELS[key] = whisper.load_model(name, device=device)
    return _MODELS[key]


def _resolve_cpu_threads(raw: object) -> int:
    """0 = auto theo số core máy (clamp ``hardware.max_auto_threads``)."""
    threads = int(raw or 0)
    if threads <= 0:
        threads = min(os.cpu_count() or 4, CFG.hardware.max_auto_threads)
    return threads


def _get_faster_model(cfg: dict):
    cpu_threads = _resolve_cpu_threads(cfg.get("cpu_threads"))
    key = "faster:{model}:{device}:{compute}:{cpu_threads}:{num_workers}".format(
        model=cfg.get("model"),
        device=cfg.get("device"),
        compute=cfg.get("compute_type"),
        cpu_threads=cpu_threads,
        num_workers=cfg.get("num_workers") or 1,
    )
    if key in _MODELS:
        return _MODELS[key]

    from backend.services.dubbing.engines.omnivoice import release_omnivoice_models
    release_omnivoice_models()

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper chưa được cài. Chạy: python -m pip install -r requirements.txt"
        ) from exc

    model_name = str(cfg.get("model") or "turbo")
    model_name = _FASTER_MODEL_ALIASES.get(model_name.lower(), model_name)
    device = str(cfg.get("device") or "cuda")
    compute_type = str(cfg.get("compute_type") or ("float16" if device == "cuda" else "int8"))
    num_workers = int(cfg.get("num_workers") or 1)
    print(
        f"[whisper] Loading faster-whisper {model_name} on {device} ({compute_type})…",
        flush=True,
    )
    _MODELS[key] = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
    )
    return _MODELS[key]


def _release_models() -> None:
    """Drop model references and return Whisper's cached CUDA memory."""
    _MODELS.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# --- Dựng lại "câu" từ word_timestamps ------------------------------------
#
# Whisper tự cắt segment theo timestamp-token nội bộ, KHÔNG neo theo pause
# thật, nên có thể dính 2 câu làm 1 (nói nhanh, ít nghỉ) hoặc lệch mốc
# đầu/cuối so với từ thật. Khi word_timestamps=True, ta bỏ hẳn ranh giới
# segment thô của Whisper, phẳng hoá TOÀN BỘ từ của audio thành 1 dòng thời
# gian duy nhất rồi tự dựng lại câu từ các tín hiệu: gap giữa 2 từ, dấu kết
# câu; câu vượt ngưỡng max_words được cắt mềm tại dấu phẩy/liên từ — không
# bao giờ cắt cứng theo đếm từ (mảnh câu cụt dịch + đọc TTS đều tệ).

_SOFT_PAUSE_SUFFIXES = (",", ";", "-")
_SENTENCE_END_SUFFIXES = (".", "!", "?", "…")
_SOFT_CONJUNCTIONS = {"and", "but", "because", "so", "which", "that"}


def _is_soft_pause(word: dict) -> bool:
    return word["word"].endswith(_SOFT_PAUSE_SUFFIXES)


def _is_sentence_end(word: dict) -> bool:
    return word["word"].endswith(_SENTENCE_END_SUFFIXES)


def _is_conjunction(word: dict) -> bool:
    return word["word"].strip(".,;-").lower() in _SOFT_CONJUNCTIONS


def _words_to_entry(words: list[dict], i: int, j: int) -> TranscriptEntry:
    text = " ".join(w["word"] for w in words[i:j])
    start = words[i]["start"]
    end = words[j - 1]["end"]
    return TranscriptEntry(text=text, start=start, duration=max(0.0, end - start))


def _find_balanced_cut(candidates: list[int], i: int, j: int) -> int:
    """Trong các vị trí cắt ứng viên (cắt ngay sau index k), chọn điểm chia
    đôi [i, j) cân bằng nhất (gần giữa nhất). Trả -1 nếu không có ứng viên."""
    best_k, best_diff = -1, j - i
    for k in candidates:
        diff = abs((k - i + 1) - (j - k - 1))
        if diff < best_diff:
            best_diff, best_k = diff, k
    return best_k


def _split_long_run(words: list[dict], i: int, j: int, max_words: int) -> list[tuple[int, int]]:
    """Chia đệ quy [i, j) khi vượt max_words — CHỈ tại điểm cắt tự nhiên.

    Ưu tiên dấu phẩy/chấm phẩy/gạch ngang gần giữa đoạn nhất; không có thì
    liên từ (and/but/because/...) — cắt TRƯỚC liên từ để nó mở đầu vế sau
    (vế trước kết thúc bằng "because" lơ lửng thì dịch lẫn TTS đều hụt).
    KHÔNG còn cắt cứng theo đếm từ: cắt giữa cụm từ đang dang dở cho ra hai
    mảnh câu cụt — bản dịch ngang phè và giọng đọc sai nhịp; một câu dài
    trọn vẹn đọc lên vẫn tự nhiên (câu quá tải đã có cảnh báo mật độ ở UI).
    ``max_words`` vì vậy là NGƯỠNG KÍCH HOẠT cắt mềm, không phải trần cứng —
    đoạn không có điểm bấu víu nào được giữ nguyên.
    """
    if j - i <= max_words:
        return [(i, j)]

    soft_candidates = [m for m in range(i + 1, j - 1) if _is_soft_pause(words[m])]
    k = _find_balanced_cut(soft_candidates, i, j)
    if k == -1:
        # Ứng viên là m-1 (từ ngay TRƯỚC liên từ) để vế sau mở đầu bằng liên
        # từ; range từ i+2 giữ vế đầu không rỗng.
        conj_candidates = [m - 1 for m in range(i + 2, j - 1) if _is_conjunction(words[m])]
        k = _find_balanced_cut(conj_candidates, i, j)

    if k == -1:
        return [(i, j)]

    return (
        _split_long_run(words, i, k + 1, max_words)
        + _split_long_run(words, k + 1, j, max_words)
    )


def _merge_short_ranges(
    words: list[dict], ranges: list[tuple[int, int]], min_words: int,
) -> list[tuple[int, int]]:
    """Gộp các dải quá ngắn (< min_words từ, vd 1 từ đứng riêng) vào dải liền
    kề GẦN HƠN về thời gian (gap nhỏ hơn), thay vì để đứng tách 1 mình — dịch
    và lồng tiếng 1 câu chỉ 1-2 từ nghe cụt lủn, thiếu ngữ cảnh.
    """
    if len(ranges) <= 1 or min_words <= 1:
        return ranges

    merged = [list(r) for r in ranges]
    changed = True
    while changed:
        changed = False
        for idx, (i, j) in enumerate(merged):
            if j - i >= min_words:
                continue
            has_prev = idx > 0
            has_next = idx < len(merged) - 1
            if not has_prev and not has_next:
                break
            if has_prev and has_next:
                gap_prev = words[i]["start"] - words[merged[idx - 1][1] - 1]["end"]
                gap_next = words[merged[idx + 1][0]]["start"] - words[j - 1]["end"]
                merge_into_prev = gap_prev <= gap_next
            else:
                merge_into_prev = has_prev
            if merge_into_prev:
                merged[idx - 1][1] = j
            else:
                merged[idx + 1][0] = i
            merged.pop(idx)
            changed = True
            break
    return [tuple(r) for r in merged]


def _split_into_entries(
    words: list[dict], max_words: int, pause_alpha: float, min_words: int = 1,
) -> list[TranscriptEntry]:
    """Dựng lại 'câu' từ dòng từ phẳng: cắt khi gap thời gian giữa 2 từ liên
    tiếp >= pause_alpha, khi gặp từ kết câu (.!?…); câu vượt max_words được
    cắt mềm tại dấu phẩy/liên từ nếu có (xem ``_split_long_run`` — không có
    thì giữ nguyên, không cắt cứng); rồi gộp lại các dải < min_words từ
    vào dải liền kề gần hơn (xem ``_merge_short_ranges``).

    ``max_words <= 0`` tắt hẳn tầng cắt mềm theo độ dài (kể cả tại dấu
    phẩy/liên từ) — chỉ còn cắt theo gap/dấu kết câu.
    """
    ranges: list[tuple[int, int]] = []
    i = 0
    n = len(words)
    while i < n:
        j = i + 1
        while j < n and words[j]["start"] - words[j - 1]["end"] < pause_alpha:
            if _is_sentence_end(words[j]):
                j += 1
                break
            j += 1
        if max_words <= 0 or j - i <= max_words:
            ranges.append((i, j))
        else:
            ranges.extend(_split_long_run(words, i, j, max_words))
        i = j

    ranges = _merge_short_ranges(words, ranges, min_words)
    return [_words_to_entry(words, start, end) for start, end in ranges]


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


ProgressCallback = Callable[[str], None]


def _transcribe_openai(
    audio_path: str,
    language: str,
    on_progress: ProgressCallback | None = None,
) -> Optional[Transcript]:
    kwargs = {
        "initial_prompt": INITIAL_PROMPT,
        "condition_on_previous_text": False,
        "verbose": False,
    }
    model = _get_openai_model(language)
    if getattr(model, "is_multilingual", False):
        kwargs["language"] = language
    result = model.transcribe(audio_path, **kwargs)
    if on_progress:
        on_progress("Đang nhận diện giọng nói bằng Whisper 100%")
    return _segments_to_transcript(result.get("segments", []))


def _faster_segments_to_transcript_with_progress(
    segments: object,
    duration: float | None,
    on_progress: ProgressCallback | None = None,
    *,
    sentence_max_words: int | None = None,
    sentence_pause_alpha: float | None = None,
    sentence_min_words: int | None = None,
) -> Optional[Transcript]:
    """Chuyển segments của faster-whisper → Transcript.

    Nếu segment có ``.words`` (word_timestamps=True), gom từ của các segment
    liên tiếp thành 1 dòng phẳng rồi dựng lại câu bằng ``_split_into_entries``
    (mốc đầu/cuối được siết về đúng từ thật, không còn dùng biên segment thô).
    Segment nào thiếu ``.words`` (word_timestamps tắt, hoặc alignment lỗi cho
    đúng đoạn đó) sẽ fallback dùng nguyên segment.text/start/end — không bao
    giờ làm mất lời thoại.
    """
    max_words = int(sentence_max_words or CFG.whisper.sentence_max_words)
    pause_alpha = float(sentence_pause_alpha if sentence_pause_alpha is not None else CFG.whisper.sentence_pause_alpha)
    min_words = int(sentence_min_words if sentence_min_words is not None else CFG.whisper.sentence_min_words)

    entries: list[TranscriptEntry] = []
    word_buffer: list[dict] = []

    def flush_words() -> None:
        if word_buffer:
            entries.extend(_split_into_entries(word_buffer, max_words, pause_alpha, min_words))
            word_buffer.clear()

    last_pct = -1
    total_duration = max(0.1, float(duration or 0.0))
    for segment in segments:
        text = (getattr(segment, "text", "") or "").strip()
        start = float(getattr(segment, "start", 0.0) or 0.0)
        end = float(getattr(segment, "end", start) or start)
        if on_progress and total_duration > 0:
            pct = max(0, min(99, int(end * 100 / total_duration)))
            if pct >= last_pct + 2:
                on_progress(f"Đang nhận diện giọng nói bằng Whisper {pct}%")
                last_pct = pct
        if not text:
            continue

        words = getattr(segment, "words", None) or []
        if words:
            for w in words:
                word_text = (w.word or "").strip()
                if word_text:
                    word_buffer.append({
                        "word": word_text,
                        "start": float(w.start),
                        "end": float(w.end),
                    })
        else:
            flush_words()
            entries.append(TranscriptEntry(
                text=text,
                start=start,
                duration=max(0.0, end - start),
            ))
    flush_words()

    if on_progress:
        on_progress("Đang nhận diện giọng nói bằng Whisper 100%")
    return Transcript(entries) if entries else None


def _transcribe_faster(
    audio_path: str,
    cfg: dict,
    language: str,
    on_progress: ProgressCallback | None = None,
) -> Optional[Transcript]:
    if on_progress:
        on_progress("Đang tải model Whisper…")
    model = _get_faster_model(cfg)
    if on_progress:
        on_progress("Đang chuẩn bị nhận diện giọng nói bằng Whisper 0%")
    beam_size = int(cfg.get("beam_size") or 5)
    batch_size = int(cfg.get("batch_size") or 1)
    progressive = bool(cfg.get("progressive", True))
    transcribe_kwargs = {
        "beam_size": beam_size,
        "language": cfg.get("language") or language,
        "initial_prompt": INITIAL_PROMPT or None,
        "condition_on_previous_text": False,
        "without_timestamps": False,
        "log_progress": False,
        "word_timestamps": bool(cfg.get("word_timestamps", True)),
    }
    if batch_size > 1 and not progressive:
        from faster_whisper import BatchedInferencePipeline
        batched_model = BatchedInferencePipeline(model=model)
        segments, _info = batched_model.transcribe(
            audio_path,
            batch_size=batch_size,
            **transcribe_kwargs,
        )
    else:
        if batch_size > 1 and on_progress:
            on_progress("Chạy Whisper progressive để cập nhật tiến độ đều hơn 0%")
        segments, _info = model.transcribe(audio_path, **transcribe_kwargs)
    return _faster_segments_to_transcript_with_progress(
        segments,
        getattr(_info, "duration", None),
        on_progress,
    )


@skip_if_exists
def fetch_transcript(
    url: str,
    output_dir: str = SUBTITLES_DIR,
    ext: str = "json",
    language: Optional[str] = None,
    preset: Optional[str] = None,
    on_progress: ProgressCallback | None = None,
) -> Optional[str]:
    """Download audio (skip nếu có) → Whisper transcribe → save json.

    Model routing:
      - default path uses faster-whisper presets from ``backend/config.yaml``.
      - ``preset='gpu'`` currently maps to medium.en on CUDA float16.
      - ``preset='cpu'`` currently maps to small.en on CPU int8.
      - legacy openai-whisper fallback is still supported via config.

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

    cfg = _preset_config(preset, language)
    language = str(cfg.get("language") or language or "en")
    with GPU_MODEL_LOCK:
        try:
            if cfg["engine"] == "openai":
                trans = _transcribe_openai(audio_path, language, on_progress)
            else:
                trans = _transcribe_faster(audio_path, cfg, language, on_progress)
        finally:
            _release_models()
    if trans is None:
        return None
    return trans.save_json(video_id=video_id, folder=output_dir)
