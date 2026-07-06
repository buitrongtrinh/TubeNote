"""Shared audio fitting helpers for dubbing engines."""

from __future__ import annotations

import subprocess

from backend.config import CFG


def fit_by_silence(
    wav,
    target_len: int,
    sample_rate: int,
    min_pause: float = 0.08,
    frame_ms: int = 20,
    silence_db: float = -40.0,
    protect_ratio: float = 0.3,
    protect_min_pause: float | None = None,
):
    """Shorten audio by trimming silence while preserving voiced samples.

    Không cắt mọi khoảng lặng theo cùng 1 sàn: các run lặng DÀI NHẤT (top
    ``protect_ratio``, khả năng cao là chỗ ngắt câu/dấu phẩy — tạo nhịp thở tự
    nhiên) được giữ sàn cao hơn (``protect_min_pause``), còn run ngắn (khoảng
    trống vi mô giữa 2 từ do model tự sinh) vẫn dùng sàn ``min_pause`` thấp
    hơn — cắt trước, cắt nhiều hơn, không ảnh hưởng nhịp nghỉ của câu.
    """
    import numpy as np

    n = len(wav)
    if n <= target_len:
        return wav

    frame = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = n // frame
    if n_frames == 0:
        return wav

    rms = np.sqrt(np.mean(wav[:n_frames * frame].reshape(n_frames, frame) ** 2, axis=1) + 1e-9)
    thresh = max(rms.max(), 1e-6) * (10 ** (silence_db / 20))
    is_sil = rms < thresh

    runs = []
    i = 0
    while i < n_frames:
        if is_sil[i]:
            j = i
            while j < n_frames and is_sil[j]:
                j += 1
            runs.append((i * frame, j * frame))
            i = j
        else:
            i += 1

    if not runs:
        return wav

    min_pause_n = int(sample_rate * min_pause)
    protect_min_pause_n = int(sample_rate * (protect_min_pause if protect_min_pause is not None else min_pause * 3))

    order = sorted(range(len(runs)), key=lambda i: runs[i][1] - runs[i][0], reverse=True)
    protect_count = round(len(runs) * protect_ratio) if len(runs) > 1 else 0
    protected = set(order[:protect_count])

    trimmable = [
        max(0, (e - s) - (protect_min_pause_n if idx in protected else min_pause_n))
        for idx, (s, e) in enumerate(runs)
    ]
    total = sum(trimmable)
    if total <= 0:
        return wav

    to_remove = min(n - target_len, total)
    out, prev = [], 0
    for (s, e), t in zip(runs, trimmable):
        out.append(wav[prev:s])
        cut = int(round(to_remove * t / total)) if total else 0
        out.append(wav[s:e - cut])
        prev = e
    out.append(wav[prev:])
    return np.concatenate(out).astype(np.float32)


def wsola_stretch(wav, rate: float):
    """Time-stretch audio with WSOLA while preserving pitch."""
    import numpy as np
    from audiotsm import wsola
    from audiotsm.io.array import ArrayReader, ArrayWriter

    reader = ArrayReader(wav.reshape(1, -1))
    writer = ArrayWriter(channels=1)
    wsola(channels=1, speed=rate).run(reader, writer)
    return writer.data.flatten().astype(np.float32)


def _ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        import shutil

        executable = shutil.which("ffmpeg")
        if executable:
            return executable
        raise RuntimeError("Cần cài imageio-ffmpeg hoặc ffmpeg để fit audio bằng atempo.")


def _atempo_chain(rate: float) -> str:
    rate = max(0.01, float(rate or 1.0))
    parts: list[float] = []
    while rate > 100.0:
        parts.append(100.0)
        rate /= 100.0
    while rate < 0.5:
        parts.append(0.5)
        rate /= 0.5
    parts.append(rate)
    return ",".join(f"atempo={part:.8g}" for part in parts)


def atempo_stretch(wav, rate: float, sample_rate: int):
    """Change tempo using ffmpeg atempo, preserving pitch."""
    import numpy as np
    import soundfile as sf
    from io import BytesIO

    if len(wav) == 0 or abs(float(rate or 1.0) - 1.0) < 0.001:
        return wav.astype(np.float32, copy=False)

    src = BytesIO()
    sf.write(src, wav.astype(np.float32), sample_rate, format="WAV", subtype="FLOAT")
    cmd = [
        _ffmpeg_executable(),
        "-hide_banner", "-loglevel", "error",
        "-f", "wav", "-i", "pipe:0",
        "-af", _atempo_chain(rate),
        "-f", "wav", "pipe:1",
    ]
    proc = subprocess.run(cmd, input=src.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg atempo failed: {proc.stderr.decode('utf-8', 'ignore')}")
    audio, out_rate = sf.read(BytesIO(proc.stdout), dtype="float32", always_2d=False)
    if out_rate != sample_rate:
        raise RuntimeError(f"ffmpeg atempo sample rate mismatch: {out_rate} != {sample_rate}")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio[:, 0]
    return np.asarray(audio, dtype=np.float32)


def fit_to_slot(
    wav,
    slot: int,
    sample_rate: int,
    *,
    wsola_limit: float = 1.0,
    silence_min_pause: float | None = None,
    silence_db: float | None = None,
    protect_ratio: float | None = None,
    protect_min_pause: float | None = None,
):
    """Fit generated speech into a fixed slot without dropping words when possible.

    Tham số cắt lặng mặc định lấy từ ``CFG.audio_fit`` (config.yaml) — dùng
    chung cho cả 2 engine, chỉ cần override khi caller có nhu cầu khác.
    """
    import numpy as np

    audio_fit_cfg = CFG.audio_fit
    silence_min_pause = audio_fit_cfg.silence_min_pause if silence_min_pause is None else silence_min_pause
    silence_db = audio_fit_cfg.silence_db if silence_db is None else silence_db
    protect_ratio = audio_fit_cfg.protect_ratio if protect_ratio is None else protect_ratio
    protect_min_pause = (
        silence_min_pause * audio_fit_cfg.protect_min_pause_multiplier
        if protect_min_pause is None else protect_min_pause
    )

    meta = {
        "raw_duration": round(len(wav) / sample_rate, 3) if sample_rate else 0.0,
        "target_duration": round(slot / sample_rate, 3) if sample_rate else 0.0,
        "silence_cut_duration": 0.0,
        "wsola_ratio": 1.0,
        "atempo_ratio": 1.0,
        "fit_ratio": 1.0,
        "trimmed_duration": 0.0,
        "warnings": [],
    }
    if slot <= 0:
        meta["warnings"].append("Slot audio không hợp lệ.")
        return wav.astype(np.float32, copy=False), active_samples(wav, sample_rate), meta

    raw_len = len(wav)
    if raw_len > slot:
        before = len(wav)
        wav = fit_by_silence(
            wav,
            slot,
            sample_rate,
            min_pause=silence_min_pause,
            silence_db=silence_db,
            protect_ratio=protect_ratio,
            protect_min_pause=protect_min_pause,
        )
        meta["silence_cut_duration"] = round(max(0, before - len(wav)) / sample_rate, 3)

    if len(wav) > slot and wsola_limit > 1.001:
        ratio = len(wav) / slot
        wsola_ratio = min(ratio, max(1.0, float(wsola_limit or 1.0)))
        if wsola_ratio > 1.001:
            wav = wsola_stretch(wav, wsola_ratio)
            meta["wsola_ratio"] = round(wsola_ratio, 3)

    if len(wav) > slot:
        atempo_ratio = len(wav) / slot
        wav = atempo_stretch(wav, atempo_ratio, sample_rate)
        meta["atempo_ratio"] = round(atempo_ratio, 3)

    if len(wav) > slot:
        over = len(wav) - slot
        meta["trimmed_duration"] = round(over / sample_rate, 3)
        wav = wav[:slot]

    speech_len = active_samples(wav, sample_rate)
    if len(wav) < slot:
        wav = np.pad(wav, (0, slot - len(wav)), mode="constant")

    if raw_len > slot:
        fit_ratio = raw_len / max(1, slot)
        meta["fit_ratio"] = round(fit_ratio, 3)
        if fit_ratio > 1.75:
            meta["warnings"].append(
                f"Đoạn này dài hơn slot {fit_ratio:.2f}x; đã ép tempo mạnh, nên rút gọn bản dịch."
            )
        elif fit_ratio > 1.35:
            meta["warnings"].append(
                f"Đoạn này dài hơn slot {fit_ratio:.2f}x; audio có thể nghe nhanh."
            )
        elif fit_ratio > 1.15:
            meta["warnings"].append(
                f"Đoạn này dài hơn slot {fit_ratio:.2f}x; đã nén thời gian nhẹ."
            )
    if meta["trimmed_duration"] > 0.03:
        meta["warnings"].append(
            f"ffmpeg atempo còn lệch {meta['trimmed_duration']:.2f}s; đã trim phần dư rất nhỏ."
        )
    return wav.astype(np.float32), speech_len, meta


def active_samples(
    wav,
    sample_rate: int,
    frame_ms: int = 20,
    silence_db: float | None = None,
    tail_ms: int | None = None,
) -> int:
    """Estimate when speech ends, ignoring silence padding at the end."""
    return active_range_samples(
        wav,
        sample_rate,
        frame_ms=frame_ms,
        silence_db=silence_db,
        tail_ms=tail_ms,
    )[1]


def active_range_samples(
    wav,
    sample_rate: int,
    frame_ms: int = 20,
    silence_db: float | None = None,
    head_ms: int | None = None,
    tail_ms: int | None = None,
) -> tuple[int, int]:
    """Estimate the active speech range inside a padded waveform.

    Ngưỡng mặc định lấy từ ``CFG.audio_fit`` (config.yaml) — dùng chung cho cả
    2 engine để neo timing hiển thị theo tiếng nói thực tế.
    """
    import numpy as np

    audio_fit_cfg = CFG.audio_fit
    silence_db = audio_fit_cfg.active_range_silence_db if silence_db is None else silence_db
    head_ms = audio_fit_cfg.active_range_head_ms if head_ms is None else head_ms
    tail_ms = audio_fit_cfg.active_range_tail_ms if tail_ms is None else tail_ms

    if len(wav) == 0:
        return 0, 0
    peak = float(np.max(np.abs(wav)))
    if peak <= 1e-6:
        return 0, len(wav)
    frame = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = len(wav) // frame
    if n_frames == 0:
        return 0, len(wav)
    body = wav[:n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(body ** 2, axis=1) + 1e-9)
    thresh = peak * (10 ** (silence_db / 20))
    voiced = np.flatnonzero(rms >= thresh)
    if voiced.size == 0:
        return 0, len(wav)
    start = int(voiced[0] * frame - sample_rate * head_ms / 1000)
    end = int((voiced[-1] + 1) * frame + sample_rate * tail_ms / 1000)
    start = max(0, min(start, len(wav)))
    end = min(len(wav), max(start + frame, end))
    return start, end
