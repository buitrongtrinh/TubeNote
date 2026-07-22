"""Orchestration cho luồng auto-dubbing — framework-agnostic.

Đây là lõi dùng chung: backend FastAPI và các client khác đều gọi các hàm ở đây.
Không phụ thuộc framework UI. Mọi đường dẫn neo theo
``PROJECT_ROOT`` (qua ``CFG.paths``) nên chạy đúng bất kể cwd.
"""
from __future__ import annotations

import os
import json
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

from backend.config import CFG, PROJECT_ROOT
from backend.services.youtube.utils import extract_video_id, load_json

# ── Paths (tuyệt đối, neo theo repo root) ────────────────────────────────────────
METADATA_DIR = CFG.paths.metadata_dir
SUBTITLES_DIR = CFG.paths.subtitles_dir
AUDIO_DUB_DIR = PROJECT_ROOT / "data" / "audio_dub"
VIDEO_DUB_DIR = PROJECT_ROOT / "data" / "video_dub"
BACKGROUND_DIR = PROJECT_ROOT / "data" / "background"
# Tham số tuning từng engine sống trong config.yaml (tts.supertonic /
# tts.omnivoice) — 1 nơi duy nhất để chỉnh, không cần sửa code.
TTS_POLICIES = {
    "supertonic": asdict(CFG.tts.supertonic),
    "omnivoice": asdict(CFG.tts.omnivoice),
}
OMNIVOICE_SOURCE_VOICE_ID = "source_video"


def metadata_path(vid: str) -> Path:
    return METADATA_DIR / f"{vid}.json"


def raw_subtitles_path(vid: str) -> Path:
    return SUBTITLES_DIR / f"{vid}.json"


def engine_subtitles_path(vid: str, engine: str) -> Path:
    if engine not in TTS_POLICIES:
        raise ValueError(f"TTS engine không hợp lệ: {engine!r}")
    return SUBTITLES_DIR / f"{vid}.json"


def subtitles_path(vid: str, engine: str | None = None) -> Path:
    """Return subtitle file for playback/API.

    - engine provided: validate engine, then return the shared subtitle file.
    - no engine: return the shared subtitle file.
    """
    if engine:
        return engine_subtitles_path(vid, engine)
    return raw_subtitles_path(vid)


def audio_dub_path(vid: str) -> Path:
    return AUDIO_DUB_DIR / f"{vid}.wav"


def video_dub_path(vid: str) -> Path:
    return VIDEO_DUB_DIR / f"{vid}.mp4"


def _background_meta(metadata: dict) -> dict:
    dubbing = metadata.get("dubbing") if isinstance(metadata.get("dubbing"), dict) else {}
    background = dubbing.get("background") if isinstance(dubbing.get("background"), dict) else None
    if isinstance(background, dict):
        return background
    return {}


def _stored_voice_percent(background_meta: dict) -> float:
    """Thanh giọng gốc đã dùng cho video này, đọc từ metadata.

    Thiếu key = video dub từ TRƯỚC khi có tính năng, lúc đó bản mix hoàn toàn
    không có giọng gốc -> trả 0. Không được rơi về mặc định config (50), vì như
    vậy chỉ tạo lại một đoạn cũng âm thầm nhét giọng gốc vào cả video cũ.
    """
    stored = background_meta.get("original_voice_percent")
    return 0.0 if stored is None else float(stored)


def is_dubbed(vid: str) -> bool:
    return video_dub_path(vid).exists()


def load_metadata(vid: str) -> dict:
    p = metadata_path(vid)
    return load_json(str(p)) if p.exists() else {}


def save_metadata(vid: str, metadata: dict) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with metadata_path(vid).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def _duration_min(metadata: dict) -> float | None:
    duration = metadata.get("duration")
    try:
        return round(float(duration) / 60.0, 3)
    except (TypeError, ValueError):
        return None


def _run_mode(whisper_preset: str | None = None, tts_engine: str | None = None) -> str:
    preset = (whisper_preset or CFG.whisper.default_preset or "").lower()
    # Prefix-match để nhận cả các preset mở rộng (cpu_tiny, gpu_small, ...).
    if preset.startswith("cpu"):
        return "cpu"
    if preset.startswith("gpu"):
        return "gpu"
    engine = (tts_engine or "").lower()
    if engine == "omnivoice":
        return "gpu"
    if engine == "supertonic":
        return "cpu"
    return preset or "unknown"


def _asr_engine(whisper_preset: str | None = None) -> str:
    preset_id = whisper_preset or CFG.whisper.default_preset
    preset = (CFG.whisper.presets or {}).get(preset_id) or {}
    engine = str(preset.get("engine") or CFG.whisper.engine or "unknown")
    model = str(preset.get("model") or CFG.whisper.en_model or "unknown")
    device = str(preset.get("device") or "unknown")
    compute_type = str(preset.get("compute_type") or "unknown")
    return f"{engine}:{model}:{device}:{compute_type}"


def _voice_label(tts_cfg: dict) -> str | None:
    engine = tts_cfg.get("engine")
    if engine == "supertonic":
        return {"M5": "Giọng nam", "F5": "Giọng nữ"}.get(str(tts_cfg.get("model") or ""))
    if engine == "omnivoice":
        voice_id = tts_cfg.get("voice_id")
        if voice_id == OMNIVOICE_SOURCE_VOICE_ID:
            return "Giọng gốc video"
        voice = next(
            (item for item in CFG.tts.omnivoice_voices if item.get("id") == voice_id),
            None,
        )
        if voice:
            return voice.get("label") or voice.get("id")
    return None


def _engine_label(engine: str | None) -> str:
    if engine == "omnivoice":
        return "OmniVoice - GPU"
    if engine == "supertonic":
        return "Supertonic - CPU"
    return str(engine or "unknown")


def _clean_meta_value(value):
    if value in (None, "", "NaN"):
        return None
    return value


def _tts_batch_size_for_log(tts_cfg: dict) -> int | None:
    if (tts_cfg.get("engine") or "").lower() != "omnivoice":
        return None
    try:
        batch_size = int(tts_cfg.get("batch_size") or 0)
    except (TypeError, ValueError):
        return None
    return batch_size if batch_size > 0 else None


def _asr_meta_block(run: dict, raw_tts: dict) -> dict:
    """Khối "asr" cho metadata.dubbing — phân biệt transcript lấy từ manual
    sub của kênh (Whisper không chạy, không hiển thị model ASR) với transcript
    do Whisper STT tạo thật."""
    engine = _clean_meta_value(run.get("asr_engine"))
    if engine == "manual_sub":
        return {
            "source": "manual_sub",
            "preset": None,
            "engine": None,
            "time_sec": _clean_meta_value(run.get("asr_time_sec")),
        }
    return {
        "source": "whisper" if engine else None,
        "preset": raw_tts.get("asr_preset") or raw_tts.get("speech_preset"),
        "engine": engine,
        "time_sec": _clean_meta_value(run.get("asr_time_sec")),
    }


def _latest_dubbing_metadata(
    *,
    tts_cfg: dict,
    raw_tts: dict,
    bg_meta: dict,
    run_id: str,
) -> dict:
    from backend.services.dubbing import run_log

    run = run_log.get_run(run_id) or {}
    translation = raw_tts.get("translation") if isinstance(raw_tts.get("translation"), dict) else {}
    translation_mode = translation.get("mode") or raw_tts.get("translation_mode")
    if translation_mode not in {"api", "manual"}:
        translation_mode = None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "tts": {
            "engine": tts_cfg.get("engine"),
            "engine_label": _engine_label(tts_cfg.get("engine")),
            "model": tts_cfg.get("model"),
            "device": tts_cfg.get("device"),
            "voice_id": tts_cfg.get("voice_id") or tts_cfg.get("model"),
            "voice_label": _voice_label(tts_cfg),
            "voice_mode": tts_cfg.get("voice_mode"),
            "num_step": tts_cfg.get("num_step"),
            "batch_size": _clean_meta_value(run.get("tts_batch_size")) or _tts_batch_size_for_log(tts_cfg),
            "speed_alpha": tts_cfg.get("speed_alpha"),
            "output_speed": tts_cfg.get("output_speed"),
        },
        "asr": _asr_meta_block(run, raw_tts),
        "translation": {
            "mode": translation_mode,
            "provider": translation.get("provider") or raw_tts.get("translation_provider"),
            "model": translation.get("model") or raw_tts.get("translation_model"),
        },
        "background": {
            "enabled": bool(bg_meta.get("enabled")),
            "source": bg_meta.get("source"),
            # Lưu lại để regenerate từng đoạn trộn ra đúng cân bằng như lần dub
            # đầu, thay vì rơi về mặc định config và lệch tiếng giữa các đoạn.
            "original_voice_percent": bg_meta.get("original_voice_percent"),
        },
        "timing": {
            "merge_max_chars": tts_cfg.get("merge_max_chars"),
            "wsola_limit": tts_cfg.get("wsola_limit"),
            "fit_audio": tts_cfg.get("fit_audio", True),
            "generation_delta_alpha": tts_cfg.get("generation_delta_alpha"),
            "generation_delta_min": tts_cfg.get("generation_delta_min"),
        },
        "run": {
            "mode": _clean_meta_value(run.get("mode")),
            "tts_time_sec": _clean_meta_value(run.get("tts_time_sec")),
            "total_time_sec": _clean_meta_value(run.get("total_time_sec")),
        },
    }


def _set_latest_run_id(vid: str, run_id: str) -> None:
    metadata = load_metadata(vid)
    metadata["latest_run_id"] = run_id
    save_metadata(vid, metadata)


@dataclass
class MixSources:
    """Ba nguồn audio + gain đã chốt cho một lần mux video.

    ``bed`` = nhạc nền (stem ``no_vocals``), ``voice`` = giọng gốc (stem
    ``vocals``). Cả hai đều có thể None: người dùng bỏ nhạc nền, bỏ giọng gốc,
    hoặc bỏ cả hai (chỉ còn giọng dub).
    """
    dub_volume: float
    meta: dict
    bed: str | None = None
    bed_volume: float = 0.0
    voice: str | None = None
    voice_volume: float = 0.0


@contextmanager
def _merge_background_config(
    vid: str,
    original_audio: str,
    tts_cfg: dict | None = None,
    *,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Iterator[MixSources]:
    """Chốt 3 nguồn audio (giọng dub / giọng gốc / nhạc nền) cho lần mux cuối.

    Hai lựa chọn của người dùng là ĐỘC LẬP với nhau:

    - ``keep_background``        -> nhạc nền có vào mix không
    - ``original_voice_percent`` -> giọng gốc to nhỏ thế nào (0 = bỏ)

    Demucs là CƠ CHẾ phục vụ cả hai (một lần chạy ra cả hai stem), nên nó chạy
    khi cần bất kỳ nửa nào — không phải một lựa chọn riêng của nhạc nền. Chỉ khi
    người dùng bỏ cả hai thì mới không cần tách, và lúc đó cũng không còn gì để
    cân với giọng dub nên bỏ luôn được cả bước đo.
    """
    from backend.services.dubbing.background import best_demucs_device, ensure_background_audio
    from backend.services.dubbing.common import voice_percent_to_gain

    cfg = dict(tts_cfg or {})
    mix_cfg = CFG.mix
    keep_background = bool(cfg.get("keep_background", False))
    voice_percent = cfg.get("original_voice_percent")
    if voice_percent is None:
        voice_percent = mix_cfg.original_voice_percent
    voice_percent = min(max(float(voice_percent), 0.0), 100.0)
    voice_volume = voice_percent_to_gain(voice_percent)

    if not keep_background and voice_percent <= 0:
        # Không nhạc nền, không giọng gốc -> chỉ còn giọng dub. Bỏ qua Demucs
        # (tiết kiệm vài phút) và không trộn nền nào vào.
        yield MixSources(
            dub_volume=mix_cfg.dub_volume_no_background,
            meta={
                "enabled": False,
                "source": "none",
                "dub_volume": mix_cfg.dub_volume_no_background,
                "original_voice_percent": 0.0,
            },
        )
        return

    # Quyết định GPU/CPU cho tách nhạc nền theo VRAM người dùng khai ở bước cấu
    # hình phần cứng (background_vram_gb): đủ VRAM + GPU còn trống -> GPU
    # (nhanh ~3-4x); khai 0 (test full CPU) hoặc GPU chật -> CPU. Regenerate
    # video cũ không có field này (None) -> best_demucs_device tự xét GPU thật.
    demucs_device = best_demucs_device(
        cfg.get("background_device"),
        vram_gb=cfg.get("background_vram_gb"),
    )

    # Luôn lấy stem giọng gốc, kể cả percent = 0: nó là MỐC đo độ to để canh
    # giọng dub (xem merge_video_audio). Nhờ vậy kéo thanh giọng gốc chỉ đổi mỗi
    # giọng gốc, không kéo theo độ to của giọng dub. Demucs sinh sẵn stem này
    # trong cùng một lần chạy nên không tốn thêm thời gian tách.
    with ensure_background_audio(
        vid,
        original_audio,
        cache_path=BACKGROUND_DIR / f"{vid}.wav",
        device=demucs_device,
        need_vocals=True,
        on_progress=on_progress,
    ) as (bg_path, voice_path):
        yield MixSources(
            bed=str(bg_path) if keep_background else None,
            bed_volume=mix_cfg.background_volume,
            dub_volume=mix_cfg.dub_volume_with_background,
            voice=str(voice_path) if voice_path else None,
            voice_volume=voice_volume,
            meta={
                "enabled": keep_background,
                "source": "demucs",
                "background_volume": mix_cfg.background_volume if keep_background else 0.0,
                "dub_volume": mix_cfg.dub_volume_with_background,
                "original_voice_percent": voice_percent,
                "original_voice_volume": voice_volume,
            },
        )


# ── Thư viện ─────────────────────────────────────────────────────────────────────

def list_library() -> list[dict]:
    """Danh sách video đã dub (đọc từ data/video_dub + metadata)."""
    if not VIDEO_DUB_DIR.is_dir():
        return []
    items = []
    for f in sorted(VIDEO_DUB_DIR.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True):
        vid = f.stem
        m = load_metadata(vid)
        items.append({
            "video_id": vid,
            "title": m.get("title") or vid,
            "channel": m.get("channel") or "—",
            "channel_avatar": m.get("channel_avatar", ""),
            "thumbnail": m.get("thumbnail", ""),
            "view_count": m.get("view_count"),
            "duration": m.get("duration"),
            "updated_at": f.stat().st_mtime,
        })
    return items


def list_drafts() -> list[dict]:
    """Videos đã load transcript/metadata nhưng chưa xuất video dub."""
    if not METADATA_DIR.is_dir():
        return []
    items: list[dict] = []
    seen: set[str] = set()
    for meta_file in sorted(METADATA_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        vid = meta_file.stem
        if vid in seen or is_dubbed(vid):
            continue
        sub_file = raw_subtitles_path(vid)
        if not sub_file.exists():
            continue
        seen.add(vid)
        m = load_metadata(vid)
        items.append({
            "video_id": vid,
            "title": m.get("title") or vid,
            "channel": m.get("channel") or "—",
            "channel_avatar": m.get("channel_avatar", ""),
            "thumbnail": m.get("thumbnail", ""),
            "view_count": m.get("view_count"),
            "duration": m.get("duration"),
            "webpage_url": m.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}",
            "updated_at": max(meta_file.stat().st_mtime, sub_file.stat().st_mtime),
        })
    return items


def delete_library_video(vid: str) -> dict:
    """Delete one library item and its cached/generated files."""
    from backend.services.dubbing.background import vocals_cache_path

    if not re.fullmatch(r"[A-Za-z0-9_-]+", vid or ""):
        raise ValueError("Video id không hợp lệ.")

    candidates = [
        metadata_path(vid),
        SUBTITLES_DIR / f"{vid}.json",
        CFG.paths.video_sub_dir / f"{vid}.mp4",
        audio_dub_path(vid),
        video_dub_path(vid),
        BACKGROUND_DIR / f"{vid}.wav",
        vocals_cache_path(BACKGROUND_DIR / f"{vid}.wav"),
        PROJECT_ROOT / "data" / "voice_clones" / f"{vid}.wav",
    ]
    candidates.extend(CFG.paths.audio_dir.glob(f"{vid}.*"))
    candidates.extend(CFG.paths.video_dir.glob(f"{vid}.*"))
    candidates.extend(AUDIO_DUB_DIR.glob(f".{vid}.*"))
    candidates.extend(VIDEO_DUB_DIR.glob(f".{vid}.*"))

    deleted: list[str] = []
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.exists() or not path.is_file():
            continue
        seen.add(path)
        path.unlink()
        try:
            deleted.append(str(path.relative_to(PROJECT_ROOT)))
        except ValueError:
            deleted.append(str(path))

    if not deleted:
        raise FileNotFoundError("Không tìm thấy video trong thư viện.")

    return {"video_id": vid, "deleted": deleted, "deleted_count": len(deleted)}


# ── Thêm video ───────────────────────────────────────────────────────────────────

def load_video(
    url: str,
    on_progress: Optional[Callable[[str], None]] = None,
    tts_engine: str = "supertonic",
    whisper_preset: str | None = None,
    manual_batch_size: int | None = None,
    api_batch_size: int | None = None,
    sentence_split_mode: str | None = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Fetch transcript + metadata (có cache), build prompts dịch.

    Trả về dict: {video_id, already_dubbed, metadata, prompts}.
    Nếu video đã dub rồi → already_dubbed=True, prompts rỗng.
    ``on_progress(msg)`` báo tiến độ (đặc biệt khi rơi vào Whisper STT — chậm).
    """
    from backend.services.youtube.transcript import fetch_transcript
    from backend.services.youtube.transcript_yt_dlp import fetch_metadata, ensure_metadata_chapters
    from backend.services.dubbing.generate_prompts import (
        build_chapter_translation_prompt,
        create_translation_prompts,
    )
    from backend.services.dubbing import run_log

    def progress(msg: str):
        # Checkpoint hủy nằm NGAY TRONG hàm báo tiến độ: hàm này được truyền
        # xuyên suốt xuống tận vòng lặp segment của Whisper (gọi mỗi 2%), nên
        # hủy giữa lúc STT chạy vẫn dừng trong vài giây mà không phải thêm
        # tham số should_cancel cho cả tầng transcript.
        if should_cancel and should_cancel():
            raise LoadCancelled()
        if on_progress:
            on_progress(msg)

    vid = extract_video_id(url)
    if is_dubbed(vid):
        return {"video_id": vid, "already_dubbed": True, "metadata": load_metadata(vid), "prompts": []}

    started = time.perf_counter()
    mode = _run_mode(whisper_preset, tts_engine)
    initial_metadata = load_metadata(vid)
    reusable_run_id = (
        initial_metadata.get("latest_run_id")
        if raw_subtitles_path(vid).exists()
        else None
    )
    run_id = reusable_run_id if run_log.has_run(reusable_run_id) else None
    created_current_run = False

    try:
        # Hủy trước khi tốn công gì: yt-dlp tải audio bên trong fetch_transcript
        # không phát tiến độ nên không có checkpoint nào suốt đoạn đó.
        progress("Chuẩn bị nạp video")
        transcript_started = time.perf_counter()
        # "complete" = ưu tiên câu trọn vẹn, tắt gần hẳn cắt-theo-pause; mọi
        # giá trị khác (kể cả None) = mặc định "khớp hình" (config.yaml).
        is_complete_mode = sentence_split_mode == "complete"
        _, transcript_source = fetch_transcript(
            url,
            languages=["en"],
            # progress (không phải on_progress) -> mỗi lần Whisper báo % là một
            # cơ hội hủy.
            on_progress=progress,
            whisper_preset=whisper_preset,
            sentence_pause_alpha=(
                CFG.whisper.sentence_pause_alpha_complete if is_complete_mode else None
            ),
            caption_pause_alpha=(
                CFG.whisper.caption_sentence_pause_alpha_complete if is_complete_mode else None
            ),
        )
        transcript_time = time.perf_counter() - transcript_started
        progress("Lấy thông tin video")
        fetch_metadata(url)
        ensure_metadata_chapters(url)
        metadata = load_metadata(vid)

        # Ghi nhớ transcript đến từ đâu (manual sub của kênh hay Whisper STT)
        # — quyết định hiển thị "ASR" hay "Phụ đề nguồn" ở trang video sau khi
        # dub. "cache" = file đã có sẵn, giữ nguyên giá trị lần fetch trước.
        if transcript_source in ("manual_sub", "whisper"):
            metadata["transcript_source"] = transcript_source
            save_metadata(vid, metadata)
        else:
            transcript_source = metadata.get("transcript_source")

        if not run_id:
            run_id = run_log.create_run(
                video_id=vid,
                duration_min=_duration_min(metadata),
                mode=mode,
                # Chỉ ghi engine Whisper khi Whisper THẬT SỰ chạy — manual sub
                # trước đây vẫn bị ghi preset đã chọn, gây hiển thị sai.
                asr_engine=("manual_sub" if transcript_source == "manual_sub"
                            else _asr_engine(whisper_preset)),
                asr_time_sec=transcript_time,
                total_time_sec=time.perf_counter() - started,
                status="loaded",
            )
            created_current_run = True
        _set_latest_run_id(vid, run_id)

        progress("Tạo prompts")
        # Client có thể override số câu/prompt cho MỖI chế độ dịch riêng —
        # clamp về khoảng hợp lý, bỏ qua giá trị vô lý thay vì raise lỗi giữa
        # chừng load.
        resolved_manual_batch_size = CFG.translation.manual_batch_size
        if (
            manual_batch_size is not None
            and CFG.translation.manual_min_batch_size
            <= int(manual_batch_size)
            <= CFG.translation.manual_max_batch_size
        ):
            resolved_manual_batch_size = int(manual_batch_size)
        resolved_api_batch_size = CFG.translation.api_batch_size
        if (
            api_batch_size is not None
            and CFG.translation.api_min_batch_size
            <= int(api_batch_size)
            <= CFG.translation.api_max_batch_size
        ):
            resolved_api_batch_size = int(api_batch_size)
        prompts = create_translation_prompts(
            str(metadata_path(vid)),
            str(raw_subtitles_path(vid)),
            batch_size=resolved_manual_batch_size,
        )
        api_prompts = create_translation_prompts(
            str(metadata_path(vid)),
            str(raw_subtitles_path(vid)),
            batch_size=resolved_api_batch_size,
            max_chars_per_batch=CFG.translation.api_max_chars_per_batch,
        )
        chapter_prompt = build_chapter_translation_prompt(metadata)
        if created_current_run:
            run_log.update_run(run_id, total_time_sec=time.perf_counter() - started)
        return {
            "video_id": vid,
            "already_dubbed": False,
            "metadata": load_metadata(vid),
            "prompts": prompts,
            "api_prompts": api_prompts,
            "chapter_prompt": chapter_prompt,
            "translation_batching": {
                "manual_batch_size": resolved_manual_batch_size,
                "manual_min_batch_size": CFG.translation.manual_min_batch_size,
                "manual_max_batch_size": CFG.translation.manual_max_batch_size,
                "api_batch_size": resolved_api_batch_size,
                "api_min_batch_size": CFG.translation.api_min_batch_size,
                "api_max_batch_size": CFG.translation.api_max_batch_size,
                "api_max_chars_per_batch": CFG.translation.api_max_chars_per_batch,
                "api_concurrency": CFG.translation.api_concurrency,
                "api_job_timeout_sec": CFG.translation.api_job_timeout_sec,
            },
            "transcript_mode": "source",
        }
    except Exception as exc:
        metadata = load_metadata(vid)
        if run_id:
            run_log.update_run(
                run_id,
                status="error",
                error=str(exc),
                total_time_sec=time.perf_counter() - started,
            )
        else:
            run_id = run_log.create_run(
                video_id=vid,
                duration_min=_duration_min(metadata),
                mode=mode,
                asr_engine=_asr_engine(whisper_preset),
                asr_time_sec=None,
                total_time_sec=time.perf_counter() - started,
                status="error",
                error=str(exc),
            )
            if metadata:
                _set_latest_run_id(vid, run_id)
        raise


def validate_response(
    prompt_index: int,
    response: str,
    expected: int = 0,
    engine: str = "supertonic",
    budgets: list[int] | None = None,
) -> dict:
    """Parse response ChatGPT → segments cho TTS.

    ``expected`` = số dòng đáng lẽ phải có (frontend đếm từ prompt). Nếu lệch →
    báo lỗi để người dùng dịch lại batch đó (tránh IndexError lúc dub).
    Trả về {ok, error, segments}.
    """
    from backend.services.dubbing.translation_prepare import prepare_translations_for_tts

    if not response.strip():
        return {"ok": False, "error": "Chưa nhập nội dung", "segments": []}
    try:
        batch = prepare_translations_for_tts(
            response,
            f"batch_{prompt_index + 1}",
            engine=engine,
            budgets=budgets,
        )
    except ValueError as e:
        return {"ok": False, "error": str(e), "segments": []}
    if expected and len(batch) != expected:
        return {"ok": False, "segments": [],
                "error": f"Thiếu/thừa dòng: có {len(batch)}, cần {expected}. Dịch lại đủ {expected} dòng."}
    errors = [
        (index + 1, error)
        for index, segment in enumerate(batch)
        for error in segment.get("normalization", {}).get("errors", [])
    ]
    if errors:
        details = "; ".join(f"dòng {index}: {error}" for index, error in errors[:5])
        suffix = "" if len(errors) <= 5 else f"; và {len(errors) - 5} dòng khác"
        return {"ok": False, "segments": batch, "error": f"{details}{suffix}"}
    warnings = [
        warning
        for segment in batch
        for warning in segment.get("normalization", {}).get("warnings", [])
    ]
    return {"ok": True, "error": "", "segments": batch, "warnings": warnings}


def validate_chapter_response(vid: str, response: str) -> dict:
    """Validate translated chapter titles against the source metadata only."""
    from backend.services.dubbing.generate_prompts import parse_chapter_translation_response

    metadata = load_metadata(extract_video_id(vid))
    if not metadata:
        return {"ok": False, "error": "Không tìm thấy metadata video.", "titles": []}
    return parse_chapter_translation_response(response, metadata)


def chapter_translation_prompt(vid: str) -> str | None:
    """Return the current chapter prompt for restoring a pre-feature draft."""
    from backend.services.dubbing.generate_prompts import build_chapter_translation_prompt

    return build_chapter_translation_prompt(load_metadata(extract_video_id(vid)))


def _chapter_titles_for_dubbing(metadata: dict, titles: list[str] | None) -> list[str]:
    chapters = metadata.get("chapters") if isinstance(metadata.get("chapters"), list) else []
    if not chapters:
        return []
    values = titles
    if values is None:
        values = [chapter.get("title_vi") if isinstance(chapter, dict) else None for chapter in chapters]
    if not isinstance(values, list) or len(values) != len(chapters):
        raise ValueError("Video có phân cảnh; cần xác nhận đủ bản dịch tiêu đề trước khi dubbing.")
    clean = [" ".join(str(value or "").split()) for value in values]
    if any(not value for value in clean):
        raise ValueError("Video có phân cảnh; cần xác nhận đủ bản dịch tiêu đề trước khi dubbing.")
    return clean


def _apply_chapter_titles(metadata: dict, titles: list[str]) -> None:
    chapters = metadata.get("chapters") if isinstance(metadata.get("chapters"), list) else []
    if not chapters:
        return
    if len(chapters) != len(titles):
        raise ValueError("Số tiêu đề phân cảnh không khớp metadata video.")
    for chapter, title in zip(chapters, titles):
        if isinstance(chapter, dict):
            chapter["title_vi"] = title


def _extract_llm_text(response: object) -> str:
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


def translate_prompt_with_api(
    prompt_index: int,
    prompt: str,
    provider: str,
    model: str,
    report: Optional[Callable[[int, str], None]] = None,
) -> dict:
    """Translate one prepared prompt with the selected LLM API."""
    from langchain_core.messages import HumanMessage

    from backend.llm.providers import make_llm

    text = (prompt or "").strip()
    if not text:
        raise ValueError("Prompt dịch đang trống.")

    def progress(value: int, stage: str) -> None:
        if report:
            report(value, stage)

    progress(10, "Chuẩn bị mô hình dịch")
    llm = make_llm(provider=provider, model=model)
    progress(35, f"Đang dịch prompt {prompt_index + 1}")
    response = llm.invoke([HumanMessage(content=text)])
    translated = _extract_llm_text(response)
    if not translated:
        raise RuntimeError("LLM không trả về bản dịch.")
    return {
        "prompt_index": prompt_index,
        "provider": provider,
        "model": model,
        "response": translated,
    }


# ── TTS model (cache toàn tiến trình) ────────────────────────────────────────────
_TTS_CACHE: dict = {}
_REGENERATE_LOCKS: dict[str, threading.Lock] = {}
_DUB_LOCKS: dict[str, threading.Lock] = {}


class LoadCancelled(Exception):
    """Ném ra khi người dùng hủy bước nạp video giữa chừng.

    Cùng kiểu hủy HỢP TÁC như ``DubCancelled``: không kill được thread, nên
    ``load_video`` kiểm tra cờ ở checkpoint rồi thoát sạch. Checkpoint chính
    nằm ngay trong hàm báo tiến độ — Whisper gọi nó mỗi 2%, nên hủy giữa lúc
    STT chạy vẫn dừng trong vài giây thay vì phải chờ hết video.

    Load không ghi metadata cho tới khi transcript xong, và các file đã tải
    (audio/subtitle) đều là cache dùng lại được, nên hủy giữa chừng không để
    lại trạng thái hỏng.
    """


class DubCancelled(Exception):
    """Ném ra khi người dùng yêu cầu hủy dubbing giữa chừng.

    Hủy HỢP TÁC: pipeline kiểm tra ``should_cancel()`` ở các checkpoint an toàn
    (đầu mỗi câu TTS, trước bước trộn) rồi ném exception này để thoát sạch —
    lock per-video được nhả trong ``finally``, chưa ghi metadata nên video
    KHÔNG bị đánh dấu đã dub. jobs.run bắt exception + thấy cờ hủy -> trạng
    thái 'cancelled' (không phải 'error')."""


def list_tts_models() -> dict:
    whisper_presets = []
    for preset_id, preset in (CFG.whisper.presets or {}).items():
        whisper_presets.append({
            "id": preset_id,
            "label": preset.get("label") or preset_id,
            "description": preset.get("description") or "",
            "engine": preset.get("engine") or CFG.whisper.engine,
            "model": preset.get("model"),
            "device": preset.get("device"),
            "compute_type": preset.get("compute_type"),
            "batch_size": preset.get("batch_size"),
            "language": preset.get("language"),
        })
    return {
        "default": CFG.tts.default_model,
        "models": CFG.tts.models,
        "default_engine": "supertonic",
        "default_speech_preset": CFG.whisper.default_preset,
        # Mặc định thanh "giọng gốc" — lấy từ config.yaml để UI và backend dùng
        # CHUNG một nguồn, đổi số trong config là UI đổi theo.
        "default_original_voice_percent": CFG.mix.original_voice_percent,
        "speech_presets": whisper_presets,
        "engines": [
            {
                "id": "supertonic",
                "label": "Supertonic - CPU",
                "description": "Nhanh, nhẹ, chạy CPU; phù hợp tạo thử hoặc máy không có GPU.",
                "models": CFG.tts.models,
                "default_model": CFG.tts.default_model,
                "devices": ["cpu"],
                "supports_clone": False,
            },
            {
                "id": "omnivoice",
                "label": "OmniVoice - GPU",
                "description": "Chất lượng tự nhiên hơn, bám thời lượng tốt hơn, hỗ trợ clone giọng; cần GPU.",
                "models": CFG.tts.omnivoice_models,
                "default_model": CFG.tts.omnivoice_model,
                "devices": ["cuda"],
                "supports_clone": True,
                "default_voice_id": "academic_male",
                "voices": [
                    {"id": voice.get("id"), "label": voice.get("label", voice.get("id"))}
                    for voice in CFG.tts.omnivoice_voices
                    if voice.get("id")
                ] + [{
                    "id": OMNIVOICE_SOURCE_VOICE_ID,
                    "label": "Giọng gốc video",
                }],
            },
        ],
    }


def resolve_tts_config(tts: dict | None = None, tts_model: str | None = None) -> dict:
    """Chuẩn hoá payload TTS từ UI.

    Cả hai engine dùng cùng policy timing: text bám slot gốc, không merge segment
    và không retime output mặc định. Khác biệt chỉ nằm ở model/voice/quality.
    """
    cfg = dict(tts or {})
    engine = str(cfg.get("engine") or "supertonic")
    if engine not in TTS_POLICIES:
        raise ValueError(f"TTS engine không hợp lệ: {engine!r}")
    # Mặc định TẮT nhạc nền — đồng bộ với DEFAULT_TTS_CONFIG phía frontend.
    keep_background = bool(cfg.get("keep_background", False))
    # Thanh "giọng gốc" 0-100 (chỉ có tác dụng khi tách nền). None = client
    # không gửi field này -> lấy mặc định config, tức dub MỚI cư xử như UI hiện
    # tại. Các luồng tạo lại/dub lại video cũ KHÔNG rơi vào nhánh này: chúng
    # đọc qua _stored_voice_percent() và ra 0 để giữ đúng bản mix ban đầu.
    raw_voice_percent = cfg.get("original_voice_percent")
    if raw_voice_percent in (None, ""):
        original_voice_percent = CFG.mix.original_voice_percent
    else:
        try:
            original_voice_percent = float(raw_voice_percent)
        except (TypeError, ValueError):
            raise ValueError("original_voice_percent phải là số trong khoảng 0-100.")
        if not 0 <= original_voice_percent <= 100:
            raise ValueError("original_voice_percent phải nằm trong khoảng 0-100.")
    # VRAM người dùng khai ở bước cấu hình phần cứng — quyết định tách nhạc nền
    # (Demucs) chạy GPU hay CPU, độc lập với engine TTS. Giữ None nếu UI không
    # gửi (vd luồng cũ) để best_demucs_device xét theo GPU thật.
    raw_bg_vram = cfg.get("background_vram_gb")
    background_vram_gb = float(raw_bg_vram) if raw_bg_vram not in (None, "") else None

    if engine == "supertonic":
        model = str(cfg.get("model") or tts_model or CFG.tts.default_model)
        if model not in CFG.tts.models:
            raise ValueError(f"TTS model không hợp lệ: {model!r}")
        num_step = int(cfg.get("num_step") or TTS_POLICIES["supertonic"]["num_step"])
        if not 5 <= num_step <= 12:
            raise ValueError("Supertonic num_step phải nằm trong khoảng 5-12.")
        return {
            **TTS_POLICIES["supertonic"],
            "engine": engine,
            "model": model,
            "device": "cpu",
            "voice_mode": "default",
            "voice_id": None,
            "reference_audio_id": None,
            "reference_text": "",
            "num_step": num_step,
            "keep_background": keep_background,
            "original_voice_percent": original_voice_percent,
            "background_vram_gb": background_vram_gb,
        }

    model = str(cfg.get("model") or CFG.tts.omnivoice_model)
    if model not in CFG.tts.omnivoice_models:
        raise ValueError(f"OmniVoice model không hợp lệ: {model!r}")
    num_step = int(cfg.get("num_step") or TTS_POLICIES["omnivoice"]["num_step"])
    if num_step not in {16, 24, 32, 48}:
        raise ValueError("OmniVoice num_step phải là 16, 24, 32 hoặc 48.")
    preset_id = None
    voice_mode = str(cfg.get("voice_mode") or "default")
    if voice_mode not in {"default", "clone"}:
        raise ValueError(f"OmniVoice mode không hợp lệ: {voice_mode!r}")
    voice_id = cfg.get("voice_id") or "academic_male"
    voice = {}
    if voice_id:
        if voice_id == OMNIVOICE_SOURCE_VOICE_ID:
            voice = {"source": "video"}
        else:
            voice = next((item for item in CFG.tts.omnivoice_voices if item.get("id") == voice_id), None)
        if voice is None:
            raise ValueError(f"OmniVoice voice không hợp lệ: {voice_id!r}")
        voice_mode = "clone"
    return {
        "engine": engine,
        "model": model,
        "device": str(cfg.get("device") or "cuda"),
        "voice_preset_id": preset_id,
        "voice_mode": voice_mode,
        "voice_id": voice_id,
        "reference_source": voice.get("source"),
        "reference_audio_id": (
            cfg.get("reference_audio_id") or
            cfg.get("ref_audio") or
            voice.get("reference_audio") or
            None
        ),
        "reference_text": cfg.get("reference_text") or cfg.get("ref_text") or voice.get("reference_text") or "",
        "language": cfg.get("language") or "vi",
        **TTS_POLICIES["omnivoice"],
        "num_step": num_step,
        # UI có thể gửi batch tính từ VRAM người dùng NHẬP TAY (khác VRAM
        # detect) — ưu tiên nó; 0/thiếu = auto theo VRAM detect lúc synth.
        "batch_size": int(cfg.get("batch_size") or 0) or TTS_POLICIES["omnivoice"]["batch_size"],
        "keep_background": keep_background,
        "original_voice_percent": original_voice_percent,
        "background_vram_gb": background_vram_gb,
    }


def init_tts(model: str | None = None):
    voice_name = model or CFG.tts.default_model
    if voice_name not in CFG.tts.models:
        raise ValueError(f"TTS model không hợp lệ: {voice_name!r}")
    if "tts" not in _TTS_CACHE:
        from supertonic import TTS
        # intra_op_threads=0 -> None: để ONNX Runtime tự chọn theo core máy.
        # Model cache global nên đổi giá trị trong config cần restart backend.
        _TTS_CACHE["tts"] = TTS(
            intra_op_num_threads=CFG.tts.supertonic.intra_op_threads or None,
        )
        _TTS_CACHE["styles"] = {}

    styles = _TTS_CACHE.setdefault("styles", {})
    if voice_name not in styles:
        styles[voice_name] = _TTS_CACHE["tts"].get_voice_style(voice_name=voice_name)
    return _TTS_CACHE["tts"], styles[voice_name]


def _segment_output_speed(segment: dict, default: float = 1.0) -> float:
    playback = segment.get("playback") if isinstance(segment.get("playback"), dict) else {}
    return float(playback.get("speed") or segment.get("output_speed") or default)


def _update_segment_playback_tts(segment: dict, output_speed: float) -> None:
    if not output_speed or output_speed <= 0:
        return
    tts = segment.get("tts") if isinstance(segment.get("tts"), dict) else {}
    if tts.get("start") is None or tts.get("end") is None:
        return
    playback = dict(segment.get("playback") or {})
    start = float(tts["start"]) / output_speed
    end = float(tts["end"]) / output_speed
    playback["speed"] = output_speed
    playback["tts_start"] = round(start, 3)
    playback["tts_end"] = round(end, 3)
    playback["tts_duration"] = round(max(0.0, end - start), 3)
    segment["playback"] = playback


def regenerate_segment(
    vid: str,
    segment_index: int,
    text_vi: str,
    pronunciation_map: dict[str, str] | None = None,
    num_step: int = 48,
    report: Optional[Callable[[int, str], None]] = None,
) -> str:
    """Regenerate one segment with the engine stored in that segment metadata."""
    vid = extract_video_id(vid)
    subtitle_file = subtitles_path(vid)
    if not subtitle_file.exists():
        raise ValueError("Video chưa có phụ đề để tạo lại đoạn.")
    segments = load_json(str(subtitle_file))
    if segment_index < 0 or segment_index >= len(segments):
        raise ValueError(f"Segment không hợp lệ: {segment_index + 1}")
    stored_tts = segments[segment_index].get("tts")
    engine = stored_tts.get("engine") if isinstance(stored_tts, dict) else None
    if engine == "omnivoice":
        return regenerate_omnivoice_segment(
            vid, segment_index, text_vi,
            pronunciation_map=pronunciation_map,
            num_step=num_step,
            report=report,
        )
    if engine == "supertonic":
        return regenerate_supertonic_segment(
            vid, segment_index, text_vi,
            pronunciation_map=pronunciation_map,
            num_step=num_step,
            report=report,
        )
    raise ValueError("Đoạn này không có metadata TTS hợp lệ để tạo lại.")


def regenerate_full_dubbing(
    vid: str,
    report: Optional[Callable[[int, str], None]] = None,
) -> str:
    """Re-run full dubbing from the current saved translated subtitles."""
    vid = extract_video_id(vid)
    metadata = load_metadata(vid)
    if not metadata:
        raise FileNotFoundError("Không tìm thấy metadata video.")

    subtitle_file = subtitles_path(vid)
    if not subtitle_file.exists():
        raise FileNotFoundError("Không tìm thấy phụ đề hiện tại để tạo lại toàn bộ audio.")
    segments = load_json(str(subtitle_file))
    if not segments:
        raise ValueError("Phụ đề hiện tại đang rỗng.")

    dubbing_meta = metadata.get("dubbing") if isinstance(metadata.get("dubbing"), dict) else {}
    tts_meta = dubbing_meta.get("tts") if isinstance(dubbing_meta.get("tts"), dict) else {}
    if not tts_meta.get("engine"):
        raise ValueError("Video chưa có metadata TTS hợp lệ để tạo lại toàn bộ audio.")

    background_meta = dubbing_meta.get("background") if isinstance(dubbing_meta.get("background"), dict) else {}
    translation_meta = dubbing_meta.get("translation") if isinstance(dubbing_meta.get("translation"), dict) else {}
    asr_meta = dubbing_meta.get("asr") if isinstance(dubbing_meta.get("asr"), dict) else {}

    tts_payload = {
        "engine": tts_meta.get("engine"),
        "model": tts_meta.get("model"),
        "device": tts_meta.get("device"),
        "voice_mode": tts_meta.get("voice_mode"),
        "voice_id": tts_meta.get("voice_id"),
        "num_step": tts_meta.get("num_step"),
        "speed_alpha": tts_meta.get("speed_alpha"),
        "output_speed": tts_meta.get("output_speed"),
        "keep_background": bool(background_meta.get("enabled", True)),
        "original_voice_percent": _stored_voice_percent(background_meta),
        "translation": translation_meta,
        "asr_preset": asr_meta.get("preset"),
    }
    url = metadata.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}"
    return run_dubbing(url, segments, tts=tts_payload, report=report)


def regenerate_supertonic_segment(
    vid: str,
    segment_index: int,
    text_vi: str,
    pronunciation_map: dict[str, str] | None = None,
    num_step: int = 8,
    report: Optional[Callable[[int, str], None]] = None,
) -> str:
    """Replace one Supertonic slot, preserving every other generated segment."""
    from backend.services.dubbing.audio_fit import active_range_samples, fit_to_slot
    from backend.services.dubbing.common import merge_video_audio
    from backend.services.dubbing.duration_budget import count_spoken_units
    from backend.services.dubbing.glossary import load_glossary
    from backend.services.dubbing.text_normalizer import (
        apply_pronunciation_map,
        canonicalize_text,
        normalize_for_tts,
    )
    from backend.services.dubbing.translation_prepare import NORMALIZATION_VERSION
    from backend.services.youtube.download import download_audio, download_video

    def r(percent: int, stage: str):
        if report:
            report(percent, stage)

    vid = extract_video_id(vid)
    lock = _REGENERATE_LOCKS.setdefault(vid, threading.Lock())
    with lock:
        subtitle_file = engine_subtitles_path(vid, "supertonic")
        if not subtitle_file.exists():
            subtitle_file = subtitles_path(vid)
        audio_file = audio_dub_path(vid)
        video_file = video_dub_path(vid)
        if not subtitle_file.exists() or not audio_file.exists() or not video_file.exists():
            raise ValueError("Video chưa có đủ dữ liệu dubbing để tạo lại đoạn.")

        segments = load_json(str(subtitle_file))
        if segment_index < 0 or segment_index >= len(segments):
            raise ValueError(f"Segment không hợp lệ: {segment_index + 1}")
        segment = segments[segment_index]
        stored_tts = segment.get("tts") if isinstance(segment.get("tts"), dict) else {}
        if stored_tts.get("engine") != "supertonic":
            raise ValueError("Đoạn này không thuộc video sinh bằng Supertonic.")

        text_vi = canonicalize_text(text_vi)
        text_tts, pronunciation_map = apply_pronunciation_map(text_vi, pronunciation_map)
        text_tts, applied_rules = normalize_for_tts(
            text_tts, glossary=load_glossary(),
        )
        written_units = count_spoken_units(text_vi)
        spoken_units = count_spoken_units(text_tts)
        if pronunciation_map:
            applied_rules = ["pronunciation_map", *applied_rules]
        normalization = {
            "engine": "supertonic",
            "version": NORMALIZATION_VERSION,
            "applied_rules": applied_rules,
            "written_units": written_units,
            "spoken_units": spoken_units,
            "budget": None,
            "budget_tolerance": 0,
            "normalization_expansion": max(0, spoken_units - written_units),
            "allowed_units": None,
            "duration": None,
            "density": None,
            "target_units": None,
            "max_units": None,
            "warnings": [],
            "errors": [],
        }

        config = resolve_tts_config({
            "engine": "supertonic",
            "model": stored_tts.get("model"),
            "num_step": num_step,
        })
        metadata = load_metadata(vid)
        url = metadata.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}"
        original_audio = download_audio(url)
        _bg = _background_meta(metadata)
        config["keep_background"] = bool(_bg.get("enabled", True))
        config["original_voice_percent"] = _stored_voice_percent(_bg)

        token = uuid.uuid4().hex
        candidate_audio_path = AUDIO_DUB_DIR / f".{vid}.{token}.wav"
        candidate_video_path = VIDEO_DUB_DIR / f".{vid}.{token}.mp4"
        candidate_subtitle_path = subtitle_file.parent / f".{vid}.{token}.json"
        temporary_paths = [candidate_audio_path, candidate_video_path, candidate_subtitle_path]

        from backend.services.dubbing.engines.supertonic import (
            TTS_SPEED_MAX, TTS_SPEED_MIN, adaptive_speed,
        )

        try:
            r(5, "Đang tải model Supertonic")
            tts_engine, style = init_tts(config["model"])
            sample_rate = tts_engine.sample_rate
            r(25, "Đang tạo lại giọng đọc")

            slot_start = float(stored_tts.get("slot_start", segment.get("start") or 0.0))
            target_duration = float(
                stored_tts.get("target_duration") or
                stored_tts.get("duration") or
                segment.get("duration") or
                0.5
            )
            speed = adaptive_speed(text_tts, target_duration, config.get("speed_alpha") or 1.2)
            speed = max(TTS_SPEED_MIN, min(TTS_SPEED_MAX, speed))
            wav, _ = tts_engine.synthesize(
                text=text_tts,
                voice_style=style,
                lang="vi",
                total_steps=int(config["num_step"]),
                speed=speed,
                max_chunk_length=1000,
                silence_duration=0.05,
            )
            replacement = wav.flatten()

            slot_samples = int(max(0.1, target_duration) * sample_rate)
            replacement, speech_samples, fit_meta = fit_to_slot(replacement, slot_samples, sample_rate)
            active_start, active_end = active_range_samples(replacement, sample_rate)
            speech_samples = max(0, active_end - active_start)
            voice_start = slot_start + active_start / sample_rate
            voice_end = min(slot_start + target_duration, slot_start + active_end / sample_rate)
            fit_meta["active_start"] = round(active_start / sample_rate, 3)
            fit_meta["active_end"] = round(active_end / sample_rate, 3)

            import numpy as np
            import soundfile as sf

            full_audio, current_rate = sf.read(str(audio_file), dtype="float32")
            if current_rate != sample_rate:
                raise ValueError("Sample rate của đoạn mới không khớp audio dubbing hiện tại.")
            if full_audio.ndim != 1:
                raise ValueError("Audio dubbing phải là mono để tạo lại từng đoạn.")

            start_sample = int(slot_start * sample_rate)
            end_sample = min(start_sample + slot_samples, len(full_audio))
            replacement = replacement[:max(0, end_sample - start_sample)]
            full_audio[start_sample:end_sample] = 0
            full_audio[start_sample:start_sample + len(replacement)] = replacement
            peak = float(np.max(np.abs(full_audio))) if full_audio.size else 0.0
            if peak > 0.97:
                full_audio *= 0.97 / peak
            sf.write(str(candidate_audio_path), full_audio, sample_rate)

            r(72, "Đang ghép lại video")
            output_speed = _segment_output_speed(segment, TTS_POLICIES["supertonic"]["output_speed"])
            with _merge_background_config(
                vid,
                original_audio,
                config,
                on_progress=lambda msg: r(72, msg),
            ) as mix:
                merge_video_audio(
                    video_path=download_video(url),
                    audio_dub=str(candidate_audio_path),
                    output_path=str(candidate_video_path),
                    audio_bed=mix.bed,
                    bed_volume=mix.bed_volume,
                    dub_volume=mix.dub_volume,
                    playback_speed=output_speed,
                    audio_voice=mix.voice,
                    voice_volume=mix.voice_volume,
                )

            speech_duration = round(speech_samples / sample_rate, 3)
            segment["text_vi"] = text_vi
            segment["text_tts"] = text_tts
            if pronunciation_map:
                segment["pronunciation_map"] = pronunciation_map
            else:
                segment.pop("pronunciation_map", None)
            segment["normalization"] = normalization
            if fit_meta.get("warnings"):
                warnings = segment["normalization"].setdefault("warnings", [])
                for warning in fit_meta["warnings"]:
                    if warning not in warnings:
                        warnings.append(warning)
            segment["tts"] = {
                **stored_tts,
                "model": config["model"],
                "speed": round(speed, 3),
                "num_step": config["num_step"],
                "start": round(voice_start, 3),
                "end": round(voice_end, 3),
                "duration": round(max(0.0, voice_end - voice_start), 3),
                "slot_start": round(slot_start, 3),
                "slot_end": round(slot_start + target_duration, 3),
                "actual_duration": speech_duration,
                "speech_duration": speech_duration,
                "target_duration": round(target_duration, 3),
                "fit": fit_meta,
            }
            _update_segment_playback_tts(segment, output_speed)
            with open(candidate_subtitle_path, "w", encoding="utf-8") as handle:
                json.dump(segments, handle, ensure_ascii=False, indent=2)

            os.replace(candidate_audio_path, audio_file)
            os.replace(candidate_video_path, video_file)
            os.replace(candidate_subtitle_path, subtitle_file)
            r(100, "Hoàn tất")
            return vid
        finally:
            for path in temporary_paths:
                if path.exists():
                    path.unlink()


def regenerate_omnivoice_segment(
    vid: str,
    segment_index: int,
    text_vi: str,
    pronunciation_map: dict[str, str] | None = None,
    num_step: int = 48,
    report: Optional[Callable[[int, str], None]] = None,
) -> str:
    """Replace one OmniVoice slot, preserving every other generated segment."""
    from backend.services.dubbing.common import merge_video_audio
    from backend.services.dubbing.engines.omnivoice import (
        prepare_source_voice_reference,
        synthesize_omnivoice,
    )
    from backend.services.gpu_runtime import GPU_MODEL_LOCK
    from backend.services.dubbing.duration_budget import (
        count_spoken_units,
        tts_density_check,
    )
    from backend.services.dubbing.glossary import load_glossary
    from backend.services.dubbing.text_normalizer import (
        apply_pronunciation_map,
        canonicalize_text,
        normalize_for_tts,
    )
    from backend.services.dubbing.translation_prepare import NORMALIZATION_VERSION
    from backend.services.youtube.download import download_audio, download_video

    def r(percent: int, stage: str):
        if report:
            report(percent, stage)

    vid = extract_video_id(vid)
    lock = _REGENERATE_LOCKS.setdefault(vid, threading.Lock())
    with lock:
        subtitle_file = engine_subtitles_path(vid, "omnivoice")
        if not subtitle_file.exists():
            subtitle_file = subtitles_path(vid)
        audio_file = audio_dub_path(vid)
        video_file = video_dub_path(vid)
        if not subtitle_file.exists() or not audio_file.exists() or not video_file.exists():
            raise ValueError("Video chưa có đủ dữ liệu dubbing để tạo lại đoạn.")

        segments = load_json(str(subtitle_file))
        if segment_index < 0 or segment_index >= len(segments):
            raise ValueError(f"Segment không hợp lệ: {segment_index + 1}")
        segment = segments[segment_index]
        stored_tts = segment.get("tts") if isinstance(segment.get("tts"), dict) else {}
        if stored_tts.get("engine") != "omnivoice":
            raise ValueError("Chỉ hỗ trợ tạo lại đoạn cho video sinh bằng OmniVoice.")

        text_vi = canonicalize_text(text_vi)
        text_tts, pronunciation_map = apply_pronunciation_map(text_vi, pronunciation_map)
        text_tts, applied_rules = normalize_for_tts(
            text_tts, glossary=load_glossary(),
        )
        written_units = count_spoken_units(text_vi)
        spoken_units = count_spoken_units(text_tts)
        density_meta = tts_density_check(
            text_tts,
            duration=float(segment.get("duration") or 0.0),
        )
        # Câu dài chỉ cảnh báo (lưu vào normalization.warnings), không chặn regenerate.
        if pronunciation_map:
            applied_rules = ["pronunciation_map", *applied_rules]
        normalization = {
            "engine": "omnivoice",
            "version": NORMALIZATION_VERSION,
            "applied_rules": applied_rules,
            "written_units": written_units,
            "spoken_units": spoken_units,
            "budget": None,
            "budget_tolerance": 0,
            "normalization_expansion": max(0, spoken_units - written_units),
            "allowed_units": None,
            "duration": density_meta.get("duration"),
            "density": density_meta.get("density"),
            "target_units": density_meta.get("target_units"),
            "max_units": density_meta.get("max_units"),
            "warnings": density_meta.get("warnings") or [],
            "errors": [],
        }

        config = resolve_tts_config({
            "engine": "omnivoice",
            "model": stored_tts.get("model"),
            "device": stored_tts.get("device"),
            "voice_mode": stored_tts.get("mode"),
            "voice_id": stored_tts.get("voice_id"),
            "num_step": num_step,
        })
        metadata = load_metadata(vid)
        url = metadata.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}"
        original_audio = download_audio(url)
        _bg = _background_meta(metadata)
        config["keep_background"] = bool(_bg.get("enabled", True))
        config["original_voice_percent"] = _stored_voice_percent(_bg)
        if config.get("reference_source") == "video":
            reference_path = PROJECT_ROOT / "data" / "voice_clones" / f"{vid}.wav"
            reference_audio, reference_text = prepare_source_voice_reference(
                segments, original_audio, str(reference_path),
            )
            config["reference_audio_id"] = reference_audio
            config["reference_text"] = reference_text

        token = uuid.uuid4().hex
        generated_path = AUDIO_DUB_DIR / f".{vid}.{token}.segment.wav"
        candidate_audio_path = AUDIO_DUB_DIR / f".{vid}.{token}.wav"
        candidate_video_path = VIDEO_DUB_DIR / f".{vid}.{token}.mp4"
        candidate_subtitle_path = subtitle_file.parent / f".{vid}.{token}.json"
        temporary_paths = [
            generated_path, candidate_audio_path, candidate_video_path,
            candidate_subtitle_path,
        ]

        try:
            r(5, "Đang tải model OmniVoice")
            local_item = {
                "text_tts": text_tts,
                "start": 0.0,
                "duration": float(segment.get("duration") or 0.0),
                "source_indices": [segment_index],
            }
            with GPU_MODEL_LOCK:
                timings = synthesize_omnivoice(
                    [local_item],
                    str(generated_path),
                    config,
                    on_progress=lambda done, total: r(10 + int(55 * done / total), "Đang tạo lại giọng đọc"),
                )

            import numpy as np
            import soundfile as sf

            full_audio, sample_rate = sf.read(str(audio_file), dtype="float32")
            replacement, replacement_rate = sf.read(str(generated_path), dtype="float32")
            if sample_rate != replacement_rate:
                raise ValueError("Sample rate của đoạn mới không khớp audio dubbing hiện tại.")
            if full_audio.ndim != 1 or replacement.ndim != 1:
                raise ValueError("Audio dubbing phải là mono để tạo lại từng đoạn.")

            start_sample = int(float(segment.get("start") or 0.0) * sample_rate)
            slot_samples = int(float(segment.get("duration") or 0.0) * sample_rate)
            end_sample = min(start_sample + slot_samples, len(full_audio))
            replacement = replacement[:max(0, end_sample - start_sample)]
            full_audio[start_sample:end_sample] = 0
            full_audio[start_sample:start_sample + len(replacement)] = replacement
            peak = float(np.max(np.abs(full_audio))) if full_audio.size else 0.0
            if peak > 0.97:
                full_audio *= 0.97 / peak
            sf.write(str(candidate_audio_path), full_audio, sample_rate)

            r(72, "Đang ghép lại video")
            output_speed = _segment_output_speed(segment, 1.0)
            with _merge_background_config(
                vid,
                original_audio,
                config,
                on_progress=lambda msg: r(72, msg),
            ) as mix:
                merge_video_audio(
                    video_path=download_video(url),
                    audio_dub=str(candidate_audio_path),
                    output_path=str(candidate_video_path),
                    audio_bed=mix.bed,
                    bed_volume=mix.bed_volume,
                    dub_volume=mix.dub_volume,
                    playback_speed=output_speed,
                    audio_voice=mix.voice,
                    voice_volume=mix.voice_volume,
                )

            segment["text_vi"] = text_vi
            segment["text_tts"] = text_tts
            if pronunciation_map:
                segment["pronunciation_map"] = pronunciation_map
            else:
                segment.pop("pronunciation_map", None)
            segment["normalization"] = normalization
            speech_duration = float(timings[0].get("speech_duration") or segment.get("duration") or 0.0)
            tts_start = float(segment.get("start") or 0.0)
            segment["tts"] = {
                **stored_tts,
                "device": config.get("device", stored_tts.get("device")),
                "num_step": config["num_step"],
                "start": round(tts_start, 3),
                "end": round(tts_start + speech_duration, 3),
                "duration": round(speech_duration, 3),
                "actual_duration": round(speech_duration, 3),
            }
            _update_segment_playback_tts(segment, output_speed)
            with open(candidate_subtitle_path, "w", encoding="utf-8") as handle:
                json.dump(segments, handle, ensure_ascii=False, indent=2)

            os.replace(candidate_audio_path, audio_file)
            os.replace(candidate_video_path, video_file)
            os.replace(candidate_subtitle_path, subtitle_file)
            r(100, "Hoàn tất")
            return vid
        finally:
            for path in temporary_paths:
                if path.exists():
                    path.unlink()


def run_dubbing(
    url: str,
    segments: list[str],
    tts: dict | None = None,
    tts_model: str | None = None,
    chapter_titles: list[str] | None = None,
    report: Optional[Callable[[int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> str:
    """Chạy toàn bộ: lưu bản dịch → TTS → trộn video. Trả về video_id.

    Chống 2 lần dub trùng cùng video (ghi đè subtitle/audio/mp4 lẫn nhau) bằng
    lock non-blocking per-video. ``report(percent, stage)`` báo tiến độ 0–100%.
    ``should_cancel()`` (tùy chọn) trả True khi người dùng yêu cầu hủy — pipeline
    kiểm tra ở checkpoint rồi ném ``DubCancelled``.
    """
    vid = extract_video_id(url)
    lock = _DUB_LOCKS.setdefault(vid, threading.Lock())
    if not lock.acquire(blocking=False):
        raise ValueError("Video này đang được lồng tiếng; đợi lần chạy trước hoàn tất.")
    try:
        return _run_dubbing_impl(
            url, segments, tts, tts_model, chapter_titles, report, should_cancel,
        )
    finally:
        lock.release()


def _run_dubbing_impl(
    url: str,
    segments: list[str],
    tts: dict | None = None,
    tts_model: str | None = None,
    chapter_titles: list[str] | None = None,
    report: Optional[Callable[[int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> str:
    """Thân dubbing thực tế. Phân bổ %: chuẩn bị 5% · TTS 5→80% · trộn 80→100%."""
    from backend.services.dubbing.common import (
        save_translations_to_file, save_tts_timings_to_file,
        save_playback_timings_to_file, merge_video_audio, merge_segments,
    )
    from backend.services.dubbing.engines.supertonic import text_to_speech
    from backend.services.dubbing import run_log
    from backend.services.youtube.download import download_audio, download_video

    def r(percent: int, stage: str):
        if report:
            report(percent, stage)

    def _ckpt():
        """Checkpoint hủy: ném DubCancelled nếu người dùng đã bấm hủy."""
        if should_cancel and should_cancel():
            raise DubCancelled()

    vid = extract_video_id(url)
    metadata = load_metadata(vid)
    confirmed_chapter_titles = _chapter_titles_for_dubbing(metadata, chapter_titles)
    # Prefetch video (nặng nhất, 1080p) song song với TTS. Audio để luồng chính
    # tự tải vì OmniVoice cần nó sớm để trích giọng — tránh tải trùng cùng file.
    # Lỗi ở đây được nuốt: bước mux gọi download_video lại (skip_if_exists) và
    # sẽ raise đúng lỗi nếu thật sự không tải được.
    def _prefetch_video():
        try:
            download_video(url)
        except Exception:
            pass

    video_prefetch = threading.Thread(target=_prefetch_video, daemon=True)
    video_prefetch.start()
    os.makedirs(AUDIO_DUB_DIR, exist_ok=True)
    os.makedirs(VIDEO_DUB_DIR, exist_ok=True)
    audio_dub = str(audio_dub_path(vid))
    raw_tts = dict(tts or {})
    tts_cfg = resolve_tts_config(tts, tts_model)
    subtitle_file = engine_subtitles_path(vid, tts_cfg["engine"])
    subtitle_file.parent.mkdir(parents=True, exist_ok=True)
    original_audio = None
    run_id = metadata.get("latest_run_id")
    if not run_log.has_run(run_id):
        run_id = run_log.create_run(
            video_id=vid,
            duration_min=_duration_min(metadata),
            mode=_run_mode(tts_engine=tts_cfg["engine"]),
            asr_engine=None,
            asr_time_sec=None,
            total_time_sec=None,
            status="loaded",
        )
        _set_latest_run_id(vid, run_id)
    dub_started = time.perf_counter()
    loaded_total = run_log.numeric((run_log.get_run(run_id) or {}).get("total_time_sec"), 0.0)

    from backend.services.dubbing.translation_prepare import renormalize_segments

    source_file = raw_subtitles_path(vid)
    source_segments = load_json(str(source_file))
    durations = [
        float(segment.get("duration") or 0.0)
        for segment in source_segments
    ]
    segments = renormalize_segments(
        segments,
        tts_cfg["engine"],
        budgets=None,
        durations=durations,
    )
    # Câu dịch dài quá thời lượng chỉ là cảnh báo (lưu trong normalization của
    # segment, hiển thị ở UI) — không chặn dub. audio_fit sẽ nén tempo để vừa slot.

    _ckpt()
    r(2, "Lưu bản dịch")
    save_translations_to_file(segments, str(subtitle_file))
    data_tts = merge_segments(
        load_json(str(subtitle_file)),
        max_chars=int(tts_cfg.get("merge_max_chars", 0)),
    )

    r(5, "Đang tải model giọng nói")
    # TTS chiếm 5→80%, báo theo từng segment. Checkpoint hủy ở ĐÂY (gọi sau mỗi
    # câu/batch) là điểm dừng chính — TTS là pha lâu nhất nên hủy ăn trong ~1 câu.
    def tts_progress(done: int, total: int):
        _ckpt()
        r(5 + int(75 * done / total), f"Tổng hợp giọng nói {done}/{total}")

    tts_started = time.perf_counter()
    try:
        if tts_cfg["engine"] == "omnivoice":
            from backend.services.dubbing.engines.omnivoice import (
                prepare_source_voice_reference,
                synthesize_omnivoice,
            )
            from backend.services.gpu_runtime import GPU_MODEL_LOCK
            if tts_cfg.get("reference_source") == "video":
                r(4, "Trích giọng gốc từ video")
                original_audio = download_audio(url)
                reference_path = PROJECT_ROOT / "data" / "voice_clones" / f"{vid}.wav"
                # source_segments (transcript gốc, chưa dịch/merge) — reference
                # audio cắt từ audio gốc nên cần transcript CÙNG NGÔN NGỮ với
                # nó; data_tts đã dịch sang tiếng Việt nên không khớp audio.
                reference_audio, reference_text = prepare_source_voice_reference(
                    source_segments, original_audio, str(reference_path),
                )
                tts_cfg["reference_audio_id"] = reference_audio
                tts_cfg["reference_text"] = reference_text
            with GPU_MODEL_LOCK:
                tts_timings = synthesize_omnivoice(data_tts, audio_dub, tts_cfg, on_progress=tts_progress)
        else:
            tts_engine, style = init_tts(tts_cfg["model"])
            tts_timings = text_to_speech(
                data_tts,
                audio_dub,
                tts_engine,
                style,
                speed_alpha=float(tts_cfg.get("speed_alpha") or 1.2),
                total_steps=int(tts_cfg.get("num_step") or 8),
                wsola_limit=float(tts_cfg.get("wsola_limit") or 1.05),
                on_progress=tts_progress,
            )
        tts_time = time.perf_counter() - tts_started
    except DubCancelled:
        # Hủy giữa TTS: không phải lỗi thật -> để run_log ở trạng thái cũ, nhả
        # lock (finally ở run_dubbing), chưa ghi metadata nên video sạch.
        raise
    except Exception as exc:
        run_log.update_run(
            run_id,
            tts_engine=tts_cfg["engine"],
            tts_batch_size=_tts_batch_size_for_log(tts_cfg),
            status="error",
            error=str(exc),
            total_time_sec=loaded_total + (time.perf_counter() - dub_started),
        )
        raise
    save_tts_timings_to_file(tts_timings, str(subtitle_file), tts_cfg)
    output_speed = float(tts_cfg.get("output_speed") or 1.0)
    save_playback_timings_to_file(str(subtitle_file), output_speed)

    # Checkpoint cuối trước khi trộn: sau đây là ffmpeg subprocess (vài giây,
    # không hủy giữa chừng được) nên hủy tại đây là cơ hội dừng cuối.
    _ckpt()
    video_prefetch.join()  # đảm bảo file video đã ghi xong trước khi mux đọc
    try:
        original_audio_path = original_audio or download_audio(url)
        with _merge_background_config(
            vid,
            original_audio_path,
            tts_cfg,
            on_progress=lambda msg: r(82, msg),
        ) as mix:
            # Đặt SAU khi tách nhạc nền xong (with-block chỉ vào tới đây khi
            # Demucs/cache bên trong _merge_background_config đã chạy hết) ->
            # thứ tự stage luôn đơn điệu: [Tách nhạc nền ->] Trộn video ->
            # Hoàn tất. Frontend dựa vào đúng chữ này để chọn ô đang chạy
            # trong chuỗi bước hiển thị.
            r(82, "Tải & trộn video")
            bg_meta = mix.meta
            merge_video_audio(
                video_path=download_video(url),
                audio_dub=audio_dub,
                output_path=str(video_dub_path(vid)),
                audio_bed=mix.bed,
                bed_volume=mix.bed_volume,
                dub_volume=mix.dub_volume,
                playback_speed=output_speed,
                audio_voice=mix.voice,
                voice_volume=mix.voice_volume,
            )
    except DubCancelled:
        raise
    except Exception as exc:
        run_log.update_run(
            run_id,
            tts_engine=tts_cfg["engine"],
            tts_batch_size=_tts_batch_size_for_log(tts_cfg),
            tts_time_sec=tts_time,
            status="error",
            error=str(exc),
            total_time_sec=loaded_total + (time.perf_counter() - dub_started),
        )
        raise
    run_log.update_run(
        run_id,
        tts_engine=tts_cfg["engine"],
        tts_batch_size=_tts_batch_size_for_log(tts_cfg),
        tts_time_sec=tts_time,
        total_time_sec=loaded_total + (time.perf_counter() - dub_started),
        status="dubbed",
        error=None,
    )
    metadata = load_metadata(vid)
    _apply_chapter_titles(metadata, confirmed_chapter_titles)
    metadata["dubbing"] = _latest_dubbing_metadata(
        tts_cfg=tts_cfg,
        raw_tts=raw_tts,
        bg_meta=bg_meta,
        run_id=run_id,
    )
    save_metadata(vid, metadata)
    r(100, "Hoàn tất")
    return vid
