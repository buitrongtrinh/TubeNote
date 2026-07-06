"""Shared dubbing helpers that are not tied to a specific TTS engine."""

import json
import os
import uuid

from backend.config import CFG
from backend.services.video.timing import clear_generated_timing, source_range


def load_json(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _atomic_write_json(data, file_path: str) -> None:
    """Ghi JSON qua file tạm rồi os.replace → không để lại subtitle hỏng nếu
    tiến trình chết giữa chừng (os.replace là atomic trên cùng filesystem)."""
    tmp = f"{file_path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, file_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def merge_segments(data: list, max_chars: int = 200, gap_threshold: float = 0.3) -> list:
    if not data:
        return []

    merged = []
    current_group = [(0, data[0])]

    for i in range(1, len(data)):
        prev = data[i - 1]
        curr = data[i]

        gap = abs(curr['start'] - (prev['start'] + prev['duration']))
        current_chars = sum(len(item.get('text_tts') or item.get('text_vi') or item.get('text') or '') for _, item in current_group)
        next_chars = len(curr.get('text_tts') or curr.get('text_vi') or curr.get('text') or '')

        if gap <= gap_threshold and current_chars + next_chars <= max_chars:
            current_group.append((i, curr))
        else:
            merged.append(current_group)
            current_group = [(i, curr)]

    merged.append(current_group)

    result = []
    for group in merged:
        first = group[0][1]
        last = group[-1][1]
        result.append({
            "source_indices": [idx for idx, _ in group],
            "source_texts": [
                (item.get('text_vi') or item.get('text_tts') or item.get('text') or '').strip()
                for _, item in group
            ],
            "text_tts": " ".join((item.get('text_tts') or '').strip() for _, item in group).strip(),
            "start": first['start'],
            "duration": (last['start'] + last['duration']) - first['start'],
        })

    return result


def save_merged_segments(data: list, output_path: str, **kwargs):
    merged = merge_segments(data, **kwargs)
    _atomic_write_json(merged, output_path)
    print(f"Đã lưu {len(merged)} segments → {output_path}")

def save_translations_to_file(translations: list, file_path: str):
    """Lưu bản dịch vào subtitle JSON của engine đang dùng.

    translations[i] có thể là:
      - dict {"vi": ..., "tts": ...}  → lưu cả text_vi (hiển thị) + text_tts (đọc)
      - str (tương thích ngược)        → chỉ lưu text_tts
    """
    data = load_json(file_path)
    if len(translations) != len(data):
        raise ValueError(
            f"Số dòng dịch ({len(translations)}) không khớp số câu ({len(data)}). "
            "Kiểm tra lại các batch đã dịch đủ chưa."
        )
    for i in range(len(data)):
        seg = translations[i]
        clear_generated_timing(data[i])
        if isinstance(seg, dict):
            data[i]['text_vi'] = seg.get('vi', '')
            data[i]['text_tts'] = seg.get('tts', '')
            if isinstance(seg.get('normalization'), dict):
                data[i]['normalization'] = seg['normalization']
            else:
                data[i].pop('normalization', None)
            if isinstance(seg.get('pronunciation_map'), dict) and seg['pronunciation_map']:
                data[i]['pronunciation_map'] = seg['pronunciation_map']
            else:
                data[i].pop('pronunciation_map', None)
        else:
            data[i]['text_tts'] = seg
    _atomic_write_json(data, file_path)


def save_tts_timings_to_file(timings: list[dict], file_path: str, tts_config: dict | None = None):
    """Ghi timing hiển thị phụ đề theo audio TTS thực tế vào subtitle JSON.

    ``timings`` là timing theo chunk TTS đã merge. Nếu một chunk gồm nhiều câu
    gốc, phân bổ duration theo độ dài text hiển thị của từng câu. Đây là xấp xỉ
    tốt hơn transcript gốc; forced alignment từng âm tiết sẽ chính xác hơn nhưng
    nặng hơn nhiều.
    """
    data = load_json(file_path)
    for seg in data:
        seg.pop("tts", None)
        for key in ("tts_start", "tts_end", "tts_duration"):
            seg.pop(key, None)

    cfg = dict(tts_config or {})
    engine = cfg.get("engine", "supertonic")
    tts_base = {
        "engine": engine,
        "model": cfg.get("model"),
        "voice_preset_id": cfg.get("voice_preset_id"),
        "mode": cfg.get("voice_mode") or cfg.get("mode") or "default",
        "voice_id": cfg.get("voice_id"),
        "device": cfg.get("device", "cpu"),
        "speed": cfg.get("speed", 1.0),
        "instruction": cfg.get("instruction"),
        "instruction_tags": cfg.get("instruction_tags"),
        "postprocess_output": cfg.get("postprocess_output"),
        "num_step": cfg.get("num_step"),
    }

    for timing in timings:
        indices = [i for i in timing.get("source_indices", []) if 0 <= i < len(data)]
        if not indices:
            continue

        start = float(timing.get("start", data[indices[0]].get("start", 0.0)))
        chunk_duration = max(0.0, float(timing.get("speech_duration", 0.0)))
        fit_meta = timing.get("fit") if isinstance(timing.get("fit"), dict) else {}
        weights = [
            max(1, len(data[i].get("text_vi") or data[i].get("text_tts") or data[i].get("text") or ""))
            for i in indices
        ]
        total = sum(weights) or 1

        t = start
        for pos, idx in enumerate(indices):
            target_start, target_end = source_range(data, idx)
            target_duration = max(0.0, target_end - target_start)
            if engine == "supertonic":
                t = max(target_start, min(start, target_end))
                if len(indices) == 1:
                    segment_speech_duration = chunk_duration
                elif chunk_duration > 0:
                    segment_speech_duration = chunk_duration * weights[pos] / total
                else:
                    segment_speech_duration = target_duration
                end = min(target_end, t + max(segment_speech_duration, 0.2))
            elif pos == len(indices) - 1:
                end = start + chunk_duration
            else:
                end = t + chunk_duration * weights[pos] / total
            if end <= t:
                end = t + max(target_duration, 0.2)
            segment_duration = round(end - t, 3)
            actual_duration = segment_duration
            if engine == "supertonic":
                actual_duration = segment_duration
            data[idx]["tts"] = {
                **{k: v for k, v in tts_base.items() if v is not None},
                "start": round(t, 3),
                "end": round(end, 3),
                "duration": segment_duration,
                "slot_start": round(target_start, 3),
                "slot_end": round(target_end, 3),
                "target_duration": round(target_duration, 3),
                "actual_duration": actual_duration,
                "speech_duration": actual_duration,
            }
            if fit_meta:
                data[idx]["tts"]["fit"] = fit_meta
                if fit_meta.get("speed") is not None:
                    data[idx]["tts"]["speed"] = fit_meta["speed"]
                warnings = fit_meta.get("warnings") or []
                if warnings:
                    normalization = data[idx].setdefault("normalization", {})
                    existing = normalization.get("warnings")
                    if not isinstance(existing, list):
                        existing = []
                    for warning in warnings:
                        if warning not in existing:
                            existing.append(warning)
                    normalization["warnings"] = existing
            t = end

    _atomic_write_json(data, file_path)


def save_playback_timings_to_file(file_path: str, speed: float):
    """Ghi timing dùng khi file MP4 output đã được retime theo ``speed``.

    Không sửa ``start``/``duration`` gốc vì chúng còn dùng cho prompt/TTS khi
    chạy lại dubbing. Player/API sẽ ưu tiên ``playback`` nếu có.
    """
    speed = float(speed or 1.0)
    data = load_json(file_path)

    for i, seg in enumerate(data):
        start, end = source_range(data, i)

        playback = {
            "speed": speed,
            "start": round(start / speed, 3),
            "end": round(end / speed, 3),
            "duration": round((end - start) / speed, 3),
        }

        tts = seg.get("tts")
        if isinstance(tts, dict) and tts.get("start") is not None and tts.get("end") is not None:
            tts_start = float(tts["start"])
            tts_end = float(tts["end"])
            if tts_end <= tts_start:
                tts_end = tts_start + max(float(tts.get("duration", 0.0)), 0.2)
            playback["tts_start"] = round(tts_start / speed, 3)
            playback["tts_end"] = round(tts_end / speed, 3)
            playback["tts_duration"] = round((tts_end - tts_start) / speed, 3)

        seg["playback"] = playback
        for key in (
            "output_speed",
            "playback_start",
            "playback_end",
            "playback_duration",
            "tts_playback_start",
            "tts_playback_end",
            "tts_playback_duration",
        ):
            seg.pop(key, None)

    _atomic_write_json(data, file_path)


def merge_video_audio(
    video_path: str,
    audio_dub: str,
    audio_original: str,
    output_path: str,
    original_volume: float | None = None,
    dub_volume: float | None = None,
    playback_speed: float = 1.0,
):
    import ffmpeg

    mix_cfg = CFG.mix
    original_volume = mix_cfg.original_volume if original_volume is None else original_volume
    dub_volume = mix_cfg.dub_volume_no_background if dub_volume is None else dub_volume

    video = ffmpeg.input(video_path)
    dub = ffmpeg.input(audio_dub)
    original = ffmpeg.input(audio_original)

    original_attenuated = ffmpeg.filter(original, 'volume', original_volume)
    dub_attenuated = ffmpeg.filter(dub, 'volume', dub_volume)
    # amix mặc định normalize=1 → tự chia đôi biên độ tổng khi trộn 2 input,
    # làm bản dub nhỏ hơn dự kiến một cách hệ thống dù đã tăng dub_volume.
    # Tắt normalize, giữ cân bằng dub:nền qua volume ở trên, rồi chuẩn hoá độ
    # to cảm nhận ở bước loudnorm bên dưới thay vì dựa vào amix.
    mixed_audio = ffmpeg.filter(
        [dub_attenuated, original_attenuated], 'amix', inputs=2, duration='first', normalize=0,
    )
    # loudnorm (EBU R128): đưa độ to cảm nhận về mức mục tiêu, khắc phục việc
    # peak-clip cũ chỉ hạ âm lượng khi vượt đỉnh, không bao giờ nâng khi cả
    # bài nhìn chung nhỏ.
    mixed_audio = ffmpeg.filter(
        mixed_audio, 'loudnorm',
        i=mix_cfg.loudnorm_i, tp=mix_cfg.loudnorm_tp, lra=mix_cfg.loudnorm_lra,
    )

    playback_speed = float(playback_speed or 1.0)
    if abs(playback_speed - 1.0) > 0.001:
        video_out = ffmpeg.filter(video.video, 'setpts', f'PTS/{playback_speed}')
        audio_out = ffmpeg.filter(mixed_audio, 'atempo', playback_speed)
        stream = ffmpeg.output(
            video_out,
            audio_out,
            output_path,
            vcodec='libx264',
            acodec='aac',
            preset='veryfast',
            crf=20,
            movflags='+faststart',
        )
    else:
        stream = ffmpeg.output(video.video, mixed_audio, output_path, vcodec='copy', acodec='aac')

    stream.overwrite_output().run()
    print(f"✓ Đã lưu → {output_path}")
