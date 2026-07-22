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


# Dưới mức này, stem "giọng" của Demucs coi như không chứa tiếng nói thật (chỉ
# còn nhiễu tách) nên không dùng làm mốc canh độ to cho dub. Loudnorm đo theo
# EBU R128 có gating bỏ qua khoảng lặng, nên giọng nói thật — kể cả thu nhỏ —
# vẫn đo cao hơn ngưỡng này khá xa.
_VOICE_ANCHOR_MIN_LUFS = -45.0


def _measure_loudness(path: str, target_i: float, target_tp: float, target_lra: float) -> dict | None:
    """Pass 1 của two-pass loudnorm: đo loudness thực tế của 1 file audio đứng
    riêng (ghi ra /dev/null, không tạo file), trả về input_i/tp/lra/thresh để
    pass 2 dùng ở chế độ linear. Trả None nếu đo lỗi (file quá ngắn/silence...)
    để nơi gọi tự rơi về hành vi cũ (không pre-normalize track đó)."""
    import ffmpeg

    try:
        _, stderr = (
            ffmpeg
            .input(path)
            .filter('loudnorm', i=target_i, tp=target_tp, lra=target_lra, print_format='json')
            .output('-', format='null')
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error:
        return None

    text = stderr.decode('utf-8', errors='ignore')
    start, end = text.rfind('{'), text.rfind('}')
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


def _prenormalize_loudness(stream, measured: dict | None, target_i: float, target_tp: float, target_lra: float):
    """Pass 2: đưa 1 track về đúng target_i đo thực tế (linear=true), thay vì
    chế độ dynamic một-pass mặc định. Dùng để dub và bed cùng khởi điểm từ 1
    mức loudness chung trước khi áp gain tỉ lệ cố định (dub_volume/
    original_volume) — nếu không, 2 gain đó phải gánh luôn cả phần chênh lệch
    loudness tự nhiên giữa các nguồn (TTS thường có level khác hẳn track gốc
    hay nhạc nền tuỳ video), nên cùng 1 bộ gain cho mọi video sẽ không ổn định."""
    import ffmpeg

    if not measured:
        return stream
    try:
        return ffmpeg.filter(
            stream, 'loudnorm',
            i=target_i, tp=target_tp, lra=target_lra,
            measured_i=measured['input_i'], measured_tp=measured['input_tp'],
            measured_lra=measured['input_lra'], measured_thresh=measured['input_thresh'],
            linear='true',
        )
    except KeyError:
        return stream


def voice_percent_to_gain(percent: float | None, curve: float | None = None) -> float:
    """Thanh 0-100 của UI -> hệ số nhân biên độ cho giọng gốc.

    Dùng đường cong mũ (mặc định bình phương) thay vì chia thẳng cho 100: tai
    người nghe -6dB (gain 0.5) ra khoảng 70% chứ không phải một nửa, phải giảm
    ~10dB mới thấy "còn một nửa". Với curve=2, 50 -> 0.25 (-12dB) nên con số
    người dùng nhập khớp với cái họ thực sự nghe được.
    """
    mix_cfg = CFG.mix
    curve = mix_cfg.original_voice_curve if curve is None else curve
    percent = mix_cfg.original_voice_percent if percent is None else percent
    ratio = min(max(float(percent), 0.0), 100.0) / 100.0
    return round(ratio ** float(curve), 6)


def merge_video_audio(
    video_path: str,
    audio_dub: str,
    output_path: str,
    audio_bed: str | None = None,
    bed_volume: float | None = None,
    dub_volume: float | None = None,
    playback_speed: float = 1.0,
    audio_voice: str | None = None,
    voice_volume: float = 0.0,
):
    """Mux video với audio đã trộn từ tối đa 3 nguồn.

    - ``audio_dub``    — giọng dub (TTS sinh ra). Luôn có.
    - ``audio_voice``  — giọng gốc (stem ``vocals`` của Demucs). Có khi đã tách.
      Dùng làm MỐC chuẩn hoá độ to cho giọng dub, và được trộn thêm vào mix khi
      ``voice_volume`` > 0.
    - ``audio_bed``    — nhạc nền (stem ``no_vocals``), hoặc audio gốc nguyên
      khối khi không tách. ``None`` = không trộn nền vào.

    Cả ba đều tuỳ chọn trừ giọng dub, nên hàm phục vụ được mọi tổ hợp mà UI cho
    chọn, kể cả "chỉ giọng dub" (không nhạc nền, không giọng gốc).
    """
    import ffmpeg

    mix_cfg = CFG.mix
    bed_volume = mix_cfg.original_volume if bed_volume is None else bed_volume
    dub_volume = mix_cfg.dub_volume_no_background if dub_volume is None else dub_volume

    video = ffmpeg.input(video_path)
    dub = ffmpeg.input(audio_dub)

    voice_measured = (
        _measure_loudness(audio_voice, mix_cfg.loudnorm_i, mix_cfg.loudnorm_tp, mix_cfg.loudnorm_lra)
        if audio_voice else None
    )
    # Mốc chuẩn hoá dub: ĐỘ TO THẬT CỦA GIỌNG GỐC khi tách được nó ra, thay cho
    # con số -16 LUFS cố định. Tác giả video đã cân giọng với nhạc sẵn rồi, nên
    # đặt dub đúng chỗ giọng gốc từng đứng là dub thừa hưởng luôn tỉ lệ đó,
    # riêng cho từng video. Mốc này KHÔNG phụ thuộc thanh giọng gốc — kéo thanh
    # đó lên xuống không được làm đổi độ to của chính giọng dub.
    dub_target_i = mix_cfg.loudnorm_i
    if voice_measured:
        try:
            measured_i = float(voice_measured["input_i"])
        except (KeyError, TypeError, ValueError):
            measured_i = None
        # Video gần như không có tiếng nói (nhạc thuần, đoạn instrumental) ->
        # stem giọng chỉ còn nhiễu tách, đo ra cực thấp. Lấy nó làm mốc là kéo
        # dub xuống theo cho tới mức chìm hẳn dưới nhạc, nên dưới ngưỡng này
        # coi như "không có giọng gốc để canh" và quay về mốc cố định.
        if measured_i is not None and measured_i >= _VOICE_ANCHOR_MIN_LUFS:
            # loudnorm chỉ nhận i trong [-70, -5] -> kẹp cho chắc.
            dub_target_i = min(max(measured_i, -70.0), -5.0)

    dub_measured = _measure_loudness(audio_dub, dub_target_i, mix_cfg.loudnorm_tp, mix_cfg.loudnorm_lra)
    dub = _prenormalize_loudness(dub, dub_measured, dub_target_i, mix_cfg.loudnorm_tp, mix_cfg.loudnorm_lra)

    mix_inputs = [ffmpeg.filter(dub, 'volume', dub_volume)]

    if audio_bed:
        bed = ffmpeg.input(audio_bed)
        # Có giọng gốc: nhạc nền CỐ TÌNH không chuẩn hoá — nó phải giữ nguyên
        # quan hệ tự nhiên với giọng gốc, thứ mà giọng dub vừa được canh theo;
        # chuẩn hoá riêng lẻ nhạc nền là phá đúng cái cân bằng vừa mượn được.
        # Không có giọng gốc (chưa tách): bed là audio gốc nguyên khối, không
        # có mốc nào để mượn -> đưa về -16 LUFS như giọng dub.
        if not audio_voice:
            bed_measured = _measure_loudness(
                audio_bed, mix_cfg.loudnorm_i, mix_cfg.loudnorm_tp, mix_cfg.loudnorm_lra,
            )
            bed = _prenormalize_loudness(
                bed, bed_measured, mix_cfg.loudnorm_i, mix_cfg.loudnorm_tp, mix_cfg.loudnorm_lra,
            )
        mix_inputs.append(ffmpeg.filter(bed, 'volume', bed_volume))

    if audio_voice and voice_volume > 0:
        mix_inputs.append(ffmpeg.filter(ffmpeg.input(audio_voice), 'volume', voice_volume))

    if len(mix_inputs) == 1:
        # Chỉ có giọng dub -> không cần amix, đi thẳng vào loudnorm.
        mixed_audio = mix_inputs[0]
    else:
        # amix mặc định normalize=1 → tự chia biên độ tổng cho số input khi
        # trộn, làm giọng dub nhỏ hơn dự kiến một cách hệ thống dù đã tăng
        # dub_volume (và càng lệch khi thêm input thứ 3). Tắt normalize, giữ
        # cân bằng qua volume ở trên, rồi chuẩn hoá độ to cảm nhận ở loudnorm.
        mixed_audio = ffmpeg.filter(
            mix_inputs, 'amix', inputs=len(mix_inputs), duration='first', normalize=0,
        )
    # loudnorm (EBU R128) lần cuối trên track đã mix: an toàn cho biến động
    # còn lại sau amix (2 nguồn đã pre-normalize + gain tỉ lệ ở trên không cho
    # ra đúng target tuyệt đối), khắc phục việc peak-clip cũ chỉ hạ âm lượng
    # khi vượt đỉnh, không bao giờ nâng khi cả bài nhìn chung nhỏ.
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
