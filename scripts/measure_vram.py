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


def _process_vram_mb(pid: int) -> int:
    """VRAM thật của process qua nvidia-smi.

    faster-whisper/CTranslate2 là runtime C++ riêng, KHÔNG dùng allocator của
    PyTorch — torch.cuda.max_memory_allocated() luôn trả 0 dù model đang chiếm
    VRAM thật (đã xác minh: model medium.en chiếm ~1.9GB nhưng torch stats vẫn
    0). Phải đọc trực tiếp từ nvidia-smi mới ra số đúng.
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], timeout=2,
        ).decode()
    except Exception:
        return 0
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        p, m = [x.strip() for x in line.split(",")]
        if int(p) == pid:
            return int(m)
    return 0


def measure_whisper(preset_id: str, audio: str) -> None:
    import os
    import resource

    from backend.services.youtube.transcript_whisper import _get_faster_model, _preset_config

    cfg = _preset_config(preset_id)
    print(f"Preset {preset_id}: {cfg.get('model')} · {cfg.get('device')} · {cfg.get('compute_type')}")

    use_cuda = cfg.get("device") == "cuda"
    pid = os.getpid()

    model = _get_faster_model(cfg)
    vram_after_load = _process_vram_mb(pid) if use_cuda else 0
    started = time.perf_counter()
    segments, info = model.transcribe(audio, word_timestamps=True)
    n_segments = sum(1 for _ in segments)
    elapsed = time.perf_counter() - started
    vram_peak = max(vram_after_load, _process_vram_mb(pid)) if use_cuda else 0

    print(f"\nWhisper {preset_id}: {n_segments} segments, audio {getattr(info, 'duration', 0):.0f}s")
    print(f"  Thời gian: {elapsed:.1f}s (x{getattr(info, 'duration', 1) / max(elapsed, 0.1):.1f} realtime)")
    if use_cuda:
        print(f"  VRAM thật (nvidia-smi): {vram_peak / 1024:.2f} GB — tham khảo cho hardware.asr_gpu_by_vram")
    else:
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        print(f"  Peak RAM (RSS): {rss_gb:.1f} GB — tham khảo cho hardware.asr_cpu_by_ram")


def _run_once(model, batch_size: int, audio: str, pid: int, use_cuda: bool) -> dict:
    """Chạy 1 lần transcribe (batched nếu batch_size>1, else tuần tự thường).

    Ép tiêu thụ hết generator (``list(segments)``) TRƯỚC khi đọc elapsed/VRAM —
    faster-whisper trả segment lười (generator), không consume hết thì đo sai.
    """
    kwargs = dict(
        word_timestamps=True,
        condition_on_previous_text=False,
        without_timestamps=False,
        log_progress=False,
    )
    started = time.perf_counter()
    if batch_size > 1:
        from faster_whisper import BatchedInferencePipeline
        batched = BatchedInferencePipeline(model=model)
        segments, info = batched.transcribe(audio, batch_size=batch_size, **kwargs)
    else:
        segments, info = model.transcribe(audio, **kwargs)
    segments = list(segments)
    elapsed = time.perf_counter() - started
    n_words = sum(len(s.words or []) for s in segments)
    covered_sec = (segments[-1].end - segments[0].start) if segments else 0.0
    preview = " ".join(s.text.strip() for s in segments[:3])
    vram_peak_mb = _process_vram_mb(pid) if use_cuda else 0
    return {
        "n_segments": len(segments),
        "n_words": n_words,
        "covered_sec": covered_sec,
        "elapsed_sec": elapsed,
        "vram_peak_mb": vram_peak_mb,
        "preview": preview,
        "duration": getattr(info, "duration", 0.0),
    }


def sweep_whisper(models: list[str], batches: list[int], device: str, compute_type: str, audio: str) -> list[dict]:
    """Quét model x batch_size thật (BatchedInferencePipeline khi batch>1),
    in bảng so sánh + diff giữa batch=1 (baseline) và batch cao nhất — dùng để
    calibrate hardware.asr_gpu_by_vram và quyết định preset nào bật
    ``progressive: false`` (batch thật) trong config.yaml.

    Model được load 1 lần rồi tái dùng cho mọi batch_size (BatchedInferencePipeline
    bọc quanh model đã load, không load lại weight) — giữa các MODEL thì giải
    phóng VRAM qua ``_release_models()`` để mô phỏng đúng hành vi "dọn giữa các
    engine" của app thật.
    """
    import os

    from backend.services.youtube.transcript_whisper import _get_faster_model, _release_models

    pid = os.getpid()
    use_cuda = device == "cuda"
    results: list[dict] = []

    for model_name in models:
        print(f"\n=== Load {model_name} ({device}/{compute_type}) ===", flush=True)
        cfg = {
            "model": model_name, "device": device, "compute_type": compute_type,
            "cpu_threads": 0, "num_workers": 2,
        }
        model = _get_faster_model(cfg)
        vram_after_load = _process_vram_mb(pid) if use_cuda else 0

        for batch in batches:
            r = _run_once(model, batch, audio, pid, use_cuda)
            r["vram_peak_mb"] = max(r["vram_peak_mb"], vram_after_load)
            r["model"] = model_name
            r["batch"] = batch
            results.append(r)
            realtime = r["duration"] / max(r["elapsed_sec"], 0.01)
            print(
                f"  batch={batch:>2}  {r['elapsed_sec']:>6.1f}s  x{realtime:>5.1f}rt  "
                f"segs={r['n_segments']:>3}  words={r['n_words']:>4}  "
                f"covered={r['covered_sec']:>6.1f}s  vram={r['vram_peak_mb'] / 1024:.2f}GB",
                flush=True,
            )
            print(f"    preview: {r['preview'][:160]}", flush=True)

        model = None
        _release_models()

    print("\n=== So sánh batch=1 (baseline) vs batch cao nhất, theo model ===")
    by_model: dict[str, dict[int, dict]] = {}
    for r in results:
        by_model.setdefault(r["model"], {})[r["batch"]] = r
    for model_name, by_batch in by_model.items():
        if 1 not in by_batch:
            continue
        base = by_batch[1]
        max_b = max(by_batch)
        if max_b == 1:
            continue
        cand = by_batch[max_b]
        speed_gain = (base["elapsed_sec"] / max(cand["elapsed_sec"], 0.01) - 1) * 100
        seg_diff = abs(cand["n_segments"] - base["n_segments"]) / max(base["n_segments"], 1) * 100
        word_diff = abs(cand["n_words"] - base["n_words"]) / max(base["n_words"], 1) * 100
        cov_diff = abs(cand["covered_sec"] - base["covered_sec"]) / max(base["covered_sec"], 0.01) * 100
        print(
            f"{model_name}: batch1 -> batch{max_b}  tốc độ {speed_gain:+.0f}%  "
            f"seg_diff {seg_diff:.1f}%  word_diff {word_diff:.1f}%  cov_diff {cov_diff:.1f}%"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--omnivoice", action="store_true", help="đo OmniVoice")
    parser.add_argument("--batch", type=int, default=4, help="batch size OmniVoice (mặc định 4)")
    parser.add_argument("--num-step", type=int, default=32, help="num_step OmniVoice (mặc định 32)")
    parser.add_argument("--whisper", metavar="PRESET", help="đo Whisper theo preset id (cpu, gpu, gpu_small…)")
    parser.add_argument("--audio", help="file audio cho --whisper / --whisper-sweep")
    parser.add_argument("--whisper-sweep", action="store_true", help="quét model x batch_size thật (batched pipeline)")
    parser.add_argument("--models", default="small.en,medium.en", help="model list, phẩy cách nhau (--whisper-sweep)")
    parser.add_argument("--batches", default="1,4,8,16", help="batch_size list, phẩy cách nhau (--whisper-sweep)")
    parser.add_argument("--device", default="cuda", help="cuda hoặc cpu (--whisper-sweep)")
    parser.add_argument("--compute-type", default="float16", help="float16/int8_float16/int8 (--whisper-sweep)")
    args = parser.parse_args()

    if args.omnivoice:
        measure_omnivoice(args.batch, args.num_step)
    elif args.whisper:
        if not args.audio:
            parser.error("--whisper cần --audio <file>")
        measure_whisper(args.whisper, args.audio)
    elif args.whisper_sweep:
        if not args.audio:
            parser.error("--whisper-sweep cần --audio <file>")
        sweep_whisper(
            models=[m.strip() for m in args.models.split(",") if m.strip()],
            batches=[int(b.strip()) for b in args.batches.split(",") if b.strip()],
            device=args.device,
            compute_type=args.compute_type,
            audio=args.audio,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
