"""Đo VRAM/thời gian thực tế của OmniVoice và Whisper để calibrate bảng
``hardware:`` trong backend/config.yaml.

Chạy tay trên máy thật (không được import bởi app):

    # Đo OmniVoice với batch 4, in peak VRAM -> điền omnivoice_batch_by_vram
    python scripts/measure_vram.py --omnivoice --batch 4 --num-step 32

    # Đo Whisper theo preset (gpu / gpu_small / cpu ...) với 1 file audio
    python scripts/measure_vram.py --whisper gpu --audio data/audio/<id>.mp3

Đọc số "peak allocated" làm ngưỡng tham khảo; VRAM thật khi chạy app sẽ nhỉnh
hơn chút (fragmentation + CUDA context), nên cộng dư ~10-15% khi điền bảng.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_TEXTS = [
    "Trong machine learning, gradient descent là thuật toán tối ưu hóa phổ biến nhất.",
    "Mô hình ngôn ngữ lớn được huấn luyện trên lượng dữ liệu văn bản khổng lồ.",
    "Docker giúp đóng gói ứng dụng cùng toàn bộ dependency thành một image duy nhất.",
    "Hôm nay chúng ta sẽ tìm hiểu cách container hoạt động bên trong hệ điều hành.",
]


def _cuda_stats(torch) -> dict:
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
        "total_vram_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1,
        ),
    }


def measure_omnivoice(batch: int, num_step: int) -> None:
    import torch
    from backend.config import CFG
    from backend.services.dubbing.engines.omnivoice import init_omnivoice

    if not torch.cuda.is_available():
        print("Không có CUDA — OmniVoice cần GPU NVIDIA.")
        return

    print(f"Load OmniVoice ({CFG.tts.omnivoice_model})…")
    model, device = init_omnivoice(CFG.tts.omnivoice_model)
    torch.cuda.reset_peak_memory_stats()
    after_load = _cuda_stats(torch)
    print(f"  Sau khi load weights: allocated {after_load['peak_allocated_gb']}GB")

    texts = (SAMPLE_TEXTS * ((batch // len(SAMPLE_TEXTS)) + 1))[:batch]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.generate(
        text=texts,
        language=["vi"] * batch,
        duration=[5.0] * batch,
        num_step=num_step,
        postprocess_output=False,
    )
    elapsed = time.perf_counter() - started
    stats = _cuda_stats(torch)

    print(f"\nOmniVoice batch={batch}, num_step={num_step} ({device}):")
    print(f"  Peak allocated : {stats['peak_allocated_gb']} GB (riêng bước generate)")
    print(f"  Peak reserved  : {stats['peak_reserved_gb']} GB")
    print(f"  VRAM tổng      : {stats['total_vram_gb']} GB")
    print(f"  Thời gian      : {elapsed:.1f}s ({elapsed / batch:.1f}s/câu)")
    need = after_load["peak_allocated_gb"] + stats["peak_allocated_gb"]
    print(
        f"\nGợi ý: batch {batch} cần ~{need:.1f}GB (+10-15% dư) — điền vào"
        f" hardware.omnivoice_batch_by_vram trong backend/config.yaml."
    )


def measure_whisper(preset_id: str, audio: str) -> None:
    import resource

    from backend.config import CFG
    from backend.services.youtube.transcript_whisper import _get_faster_model, _preset_config

    cfg = _preset_config(preset_id)
    print(f"Preset {preset_id}: {cfg.get('model')} · {cfg.get('device')} · {cfg.get('compute_type')}")

    use_cuda = cfg.get("device") == "cuda"
    torch = None
    if use_cuda:
        import torch
        torch.cuda.reset_peak_memory_stats()

    model = _get_faster_model(cfg)
    started = time.perf_counter()
    segments, info = model.transcribe(audio, word_timestamps=True)
    n_segments = sum(1 for _ in segments)
    elapsed = time.perf_counter() - started

    print(f"\nWhisper {preset_id}: {n_segments} segments, audio {getattr(info, 'duration', 0):.0f}s")
    print(f"  Thời gian: {elapsed:.1f}s (x{getattr(info, 'duration', 1) / max(elapsed, 0.1):.1f} realtime)")
    if use_cuda and torch is not None:
        stats = _cuda_stats(torch)
        print(f"  Peak VRAM allocated: {stats['peak_allocated_gb']} GB — tham khảo cho hardware.asr_gpu_by_vram")
    else:
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        print(f"  Peak RAM (RSS): {rss_gb:.1f} GB — tham khảo cho hardware.asr_cpu_by_ram")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--omnivoice", action="store_true", help="đo OmniVoice")
    parser.add_argument("--batch", type=int, default=4, help="batch size OmniVoice (mặc định 4)")
    parser.add_argument("--num-step", type=int, default=32, help="num_step OmniVoice (mặc định 32)")
    parser.add_argument("--whisper", metavar="PRESET", help="đo Whisper theo preset id (cpu, gpu, gpu_small…)")
    parser.add_argument("--audio", help="file audio cho --whisper")
    args = parser.parse_args()

    if args.omnivoice:
        measure_omnivoice(args.batch, args.num_step)
    elif args.whisper:
        if not args.audio:
            parser.error("--whisper cần --audio <file>")
        measure_whisper(args.whisper, args.audio)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
