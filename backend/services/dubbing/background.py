"""Background music extraction (Demucs) with an on-disk cache.

The extracted no-vocals stem is expensive (Demucs runs a full separation over
the whole track). Since the source audio of a given video never changes, the
stem is cached under ``data/background/{video_id}.wav`` and reused across the
full dub and every per-segment regeneration.
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


def _run_demucs(video_id: str, source: Path, out_dir: Path, device: str,
                progress: Callable[[str], None]) -> Path:
    """Run Demucs into ``out_dir`` and return the no_vocals stem path."""
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

    no_vocals = out_dir / "htdemucs" / source.stem / "no_vocals.wav"
    if not no_vocals.exists():
        raise RuntimeError(f"Demucs không tạo file no_vocals.wav tại {no_vocals.parent}")
    return no_vocals


@contextmanager
def ensure_background_audio(
    video_id: str,
    source_audio: str,
    *,
    cache_path: str | Path | None = None,
    device: str = "cpu",
    on_progress: Callable[[str], None] | None = None,
) -> Iterator[Path]:
    """Yield a no-vocals background stem, extracting with Demucs only if needed.

    When ``cache_path`` is given, a hit returns immediately without running
    Demucs; a miss runs Demucs once and persists the stem there. Without a
    cache path the stem lives only inside a TemporaryDirectory (legacy behavior).
    """
    source = Path(source_audio)
    if not source.exists():
        raise FileNotFoundError(f"Không tìm thấy audio gốc để tách nhạc nền: {source}")

    progress = on_progress or (lambda _msg: None)
    device = best_demucs_device(device)

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            progress("Dùng nhạc nền đã tách (cache)")
            yield cache_path
            return

        with tempfile.TemporaryDirectory(prefix=f"tubenote-demucs-{video_id}-") as tmp:
            no_vocals = _run_demucs(video_id, source, Path(tmp), device, progress)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            staging = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
            shutil.copyfile(no_vocals, staging)
            os.replace(staging, cache_path)
        yield cache_path
        return

    with tempfile.TemporaryDirectory(prefix=f"tubenote-demucs-{video_id}-") as tmp:
        yield _run_demucs(video_id, source, Path(tmp), device, progress)
