"""Background music extraction (Demucs) with an on-disk cache.

The extracted stems are expensive (Demucs runs a full separation over the whole
track). Since the source audio of a given video never changes, they are cached
under ``data/background/{video_id}.wav`` (no-vocals) and
``data/background/{video_id}.vocals.wav`` (original voice) and reused across the
full dub and every per-segment regeneration.

Demucs ``--two-stems=vocals`` always produces both stems in one pass, so keeping
the vocal stem costs no extra compute — it is what lets the final mix retain the
original speaker underneath the dub.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Iterator

import imageio_ffmpeg


# Demucs htdemucs đo thật cần ~0.9GB VRAM (gồm cả CUDA context của subprocess).
# Đòi ngưỡng cao hơn để trừ hao dao động activation + phân mảnh -> còn dưới
# ngưỡng thì lùi CPU cho chắc thay vì để dub vỡ vì OOM.
_DEMUCS_MIN_FREE_VRAM = 1536 * 1024 * 1024
# Cổng Ý ĐỊNH theo VRAM người dùng khai ở bước cấu hình phần cứng: dưới mức
# này (kể cả 0 = cố tình test full CPU / máy không GPU) thì tách nhạc nền chạy
# CPU dù máy có card. Thấp hơn ngưỡng OmniVoice nhiều vì Demucs nhẹ hơn hẳn.
_DEMUCS_MIN_CONFIG_VRAM_GB = 2.0


def best_demucs_device(requested: str | None = None, *, vram_gb: float | None = None) -> str:
    """Resolve the Demucs device from user config + real GPU state.

    ``requested`` cụ thể ("cpu"/"cuda") -> tôn trọng nguyên văn (override thủ
    công / test). "auto"/rỗng/None -> tự chọn theo 2 cổng:

    1. Ý ĐỊNH — ``vram_gb`` là VRAM người dùng khai ở bước cấu hình phần cứng.
       < ``_DEMUCS_MIN_CONFIG_VRAM_GB`` (0 = test full CPU hoặc máy không GPU)
       -> ``cpu``, dù máy thật có card. Cho phép người dùng chủ động chạy toàn
       bộ pipeline trên CPU. ``vram_gb=None`` (không truyền — vd regenerate) =
       bỏ qua cổng này, chỉ xét GPU thật.
    2. AN TOÀN — máy thật có CUDA và ``mem_get_info`` còn đủ VRAM trống mới
       dùng ``cuda`` (Demucs ~0.9GB, nhanh ~3-4x CPU). Bước tách nhạc nền chạy
       sau TTS nhưng OmniVoice có thể còn giữ model trên GPU; đo thật vẫn còn
       ~4GB trống trên card 6GB nên dư. GPU bị lấp gần hết -> lùi ``cpu``,
       không OOM.
    """
    if requested and requested not in ("auto", ""):
        return requested
    if vram_gb is not None and float(vram_gb or 0) < _DEMUCS_MIN_CONFIG_VRAM_GB:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            if free >= _DEMUCS_MIN_FREE_VRAM:
                return "cuda"
    except Exception:
        pass
    return "cpu"


def vocals_cache_path(cache_path: str | Path) -> Path:
    """Vocal-stem cache path sitting next to the no-vocals one.

    ``data/background/{vid}.wav`` -> ``data/background/{vid}.vocals.wav``.
    """
    cache_path = Path(cache_path)
    return cache_path.with_name(f"{cache_path.stem}.vocals{cache_path.suffix}")


def _run_demucs(source: Path, out_dir: Path, device: str,
                progress: Callable[[str], None]) -> tuple[Path, Path]:
    """Run Demucs into ``out_dir`` and return ``(no_vocals, vocals)`` paths."""
    env = dict(os.environ)
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    env["FFMPEG_BINARY"] = ffmpeg_bin
    env["PATH"] = f"{Path(ffmpeg_bin).parent}{os.pathsep}{env.get('PATH', '')}"

    progress(f"Tách nhạc nền bằng Demucs ({device})")
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals", "-n", "htdemucs",
        "--device", device,
        "--out", str(out_dir),
        str(source),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if "No module named demucs" in detail:
            detail = "Thiếu Demucs. Cài bằng: python -m pip install -r requirements.txt"
        raise RuntimeError(f"Không tách được nhạc nền bằng Demucs. {detail}")

    stem_dir = out_dir / "htdemucs" / source.stem
    no_vocals = stem_dir / "no_vocals.wav"
    vocals = stem_dir / "vocals.wav"
    if not no_vocals.exists():
        raise RuntimeError(f"Demucs không tạo file no_vocals.wav tại {stem_dir}")
    if not vocals.exists():
        raise RuntimeError(f"Demucs không tạo file vocals.wav tại {stem_dir}")
    return no_vocals, vocals


def _persist(src: Path, dest: Path) -> None:
    """Copy a stem into the cache atomically (staging file + os.replace)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    shutil.copyfile(src, staging)
    os.replace(staging, dest)


def _is_usable(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


@contextmanager
def ensure_background_audio(
    video_id: str,
    source_audio: str,
    *,
    cache_path: str | Path | None = None,
    device: str = "cpu",
    need_vocals: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> Iterator[tuple[Path, Path | None]]:
    """Yield ``(no_vocals, vocals)`` stems, extracting with Demucs only if needed.

    ``vocals`` is ``None`` when ``need_vocals`` is False — the caller does not
    want the original speaker in the mix, so the stem is neither cached nor
    kept. When ``cache_path`` is given, a hit returns immediately without
    running Demucs; a miss runs Demucs once and persists the stems there.
    Without a cache path the stems live only inside a TemporaryDirectory
    (legacy behavior).

    Caches written before the vocal stem existed hold only the no-vocals file.
    Asking for vocals against such a cache is a miss, so Demucs re-runs once and
    both stems are persisted from then on.
    """
    source = Path(source_audio)
    if not source.exists():
        raise FileNotFoundError(f"Không tìm thấy audio gốc để tách nhạc nền: {source}")

    progress = on_progress or (lambda _msg: None)
    device = best_demucs_device(device)

    if cache_path is not None:
        cache_path = Path(cache_path)
        vocals_cache = vocals_cache_path(cache_path)
        if _is_usable(cache_path) and (not need_vocals or _is_usable(vocals_cache)):
            progress("Dùng nhạc nền đã tách (cache)")
            yield cache_path, (vocals_cache if need_vocals else None)
            return

        with tempfile.TemporaryDirectory(prefix=f"tubenote-demucs-{video_id}-") as tmp:
            no_vocals, vocals = _run_demucs(source, Path(tmp), device, progress)
            _persist(no_vocals, cache_path)
            # Luôn cache cả stem giọng dù lần này không cần: Demucs đã sinh sẵn
            # nó rồi, ghi thêm ~vài chục MB rẻ hơn nhiều so với bắt lần sau chạy
            # lại toàn bộ separation chỉ vì người dùng bật thanh giọng gốc lên.
            _persist(vocals, vocals_cache)
        yield cache_path, (vocals_cache if need_vocals else None)
        return

    with tempfile.TemporaryDirectory(prefix=f"tubenote-demucs-{video_id}-") as tmp:
        no_vocals, vocals = _run_demucs(source, Path(tmp), device, progress)
        yield no_vocals, (vocals if need_vocals else None)
