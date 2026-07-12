"""Phát hiện phần cứng (RAM/VRAM/cores) và sinh bộ tham số đề xuất.

Người dùng nhập RAM/VRAM ở màn hình "Tạo lồng tiếng" (auto-detect chỉ để
pre-fill); ``recommend_setup`` tra các bảng tier trong ``config.yaml`` mục
``hardware`` để chọn ASR preset, TTS engine, batch OmniVoice và số thread —
tối ưu chất lượng/tốc độ theo từng máy. Các hàm ``pick_*``/``recommend_*``
là hàm thuần, test được không cần GPU thật.
"""
from __future__ import annotations

import os

from backend.config import CFG, HardwareCfg


def detect_ram_gb() -> float | None:
    """Tổng RAM (GB) từ /proc/meminfo; None nếu không đọc được (non-Linux)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as file:
            for line in file:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except Exception:
        pass
    return None


def detect_gpu() -> dict | None:
    """GPU CUDA đầu tiên: {name, vram_gb}; None nếu không có CUDA/torch."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return {
            "name": str(props.name),
            "vram_gb": round(props.total_memory / 1024 ** 3, 1),
        }
    except Exception:
        return None


def detect_cpu_cores() -> int:
    return os.cpu_count() or 4


def pick_tier(value: float | None, table: dict[float, object]):
    """Lấy giá trị của dòng có ngưỡng LỚN NHẤT mà ``value`` >= ngưỡng; None nếu
    không đạt dòng nào (hoặc value/table rỗng)."""
    if value is None or not table:
        return None
    best = None
    for threshold in sorted(table):
        if value >= threshold:
            best = table[threshold]
    return best


def pick_batch_size_for_vram(vram_gb: float | None, table: dict[float, int]) -> int:
    """Batch OmniVoice theo VRAM; không đạt dòng nào -> 1."""
    picked = pick_tier(vram_gb, table)
    return max(1, int(picked)) if picked else 1


def _min_threshold_by_value(table: dict[float, object]) -> dict[str, float]:
    """Map option id -> ngưỡng thấp nhất trong bảng tier."""
    result: dict[str, float] = {}
    for threshold, value in table.items():
        key = str(value)
        if key not in result or threshold < result[key]:
            result[key] = float(threshold)
    return result


def _format_requirement(kind: str, value: float) -> str:
    return f"Cần {kind} >= {value:g}GB"


def hardware_availability(
    ram_gb: float | None,
    vram_gb: float | None,
    cfg: HardwareCfg | None = None,
) -> dict:
    """Danh sách option phần cứng nào có thể chọn với RAM/VRAM hiện tại."""
    cfg = cfg or CFG.hardware
    vram = float(vram_gb) if vram_gb and vram_gb > 0 else None
    ram = float(ram_gb) if ram_gb and ram_gb > 0 else None
    asr_gpu_min = _min_threshold_by_value(cfg.asr_gpu_by_vram)
    asr_cpu_min = _min_threshold_by_value(cfg.asr_cpu_by_ram)

    asr: dict[str, dict] = {}
    for preset_id, preset in (CFG.whisper.presets or {}).items():
        device = str(preset.get("device") or "").lower()
        available = True
        reason = ""
        if device == "cuda":
            required = asr_gpu_min.get(preset_id)
            if required is None:
                available = bool(vram)
                reason = "" if available else "Cần GPU CUDA"
            else:
                available = bool(vram and vram >= required)
                reason = "" if available else _format_requirement("VRAM", required)
        elif device == "cpu":
            # Preset không nằm trong asr_cpu_by_ram (vd cpu_medium — cố ý loại
            # khỏi auto đề xuất) vẫn cần ngưỡng riêng, nếu không sẽ bị hiểu
            # nhầm là "không giới hạn RAM" và không bao giờ bị xám.
            required = asr_cpu_min.get(preset_id, cfg.asr_cpu_advanced_min_ram_gb.get(preset_id))
            if required is not None and ram is not None and ram < required:
                available = False
                reason = _format_requirement("RAM", required)
        asr[preset_id] = {"available": available, "reason": reason}

    omni_ok = bool(vram and vram >= cfg.omnivoice_min_vram_gb)
    return {
        "asr": asr,
        "tts": {
            "supertonic": {"available": True, "reason": ""},
            "omnivoice": {
                "available": omni_ok,
                "reason": "" if omni_ok else _format_requirement("VRAM", cfg.omnivoice_min_vram_gb),
            },
        },
    }


def auto_omnivoice_batch_size() -> int:
    """Batch OmniVoice khi config để auto (batch_size=0): tra bảng theo VRAM."""
    gpu = detect_gpu()
    return pick_batch_size_for_vram(
        gpu.get("vram_gb") if gpu else None,
        CFG.hardware.omnivoice_batch_by_vram,
    )


def recommend_setup(
    ram_gb: float | None,
    vram_gb: float | None,
    cpu_cores: int,
    cfg: HardwareCfg | None = None,
) -> dict:
    """Bộ tham số đề xuất cho cặp (RAM, VRAM) người dùng nhập/detect được.

    VRAM <= 0 hoặc None nghĩa là không có GPU NVIDIA.
    """
    cfg = cfg or CFG.hardware
    vram = float(vram_gb) if vram_gb and vram_gb > 0 else None
    ram = float(ram_gb) if ram_gb and ram_gb > 0 else None
    threads = max(1, min(int(cpu_cores or 4), cfg.max_auto_threads))
    notes: list[str] = []

    asr_preset = pick_tier(vram, cfg.asr_gpu_by_vram)
    if asr_preset:
        notes.append(f"VRAM {vram:g}GB đủ chạy Whisper trên GPU ({asr_preset}).")
    else:
        if vram:
            notes.append(f"VRAM {vram:g}GB chưa đủ cho Whisper GPU — dùng CPU.")
        asr_preset = pick_tier(ram, cfg.asr_cpu_by_ram) or "cpu"
        if ram and ram < 6:
            notes.append(f"RAM {ram:g}GB thấp — dùng model Whisper nhỏ hơn cho an toàn.")
    # Preset không tồn tại trong config (bảng tier trỏ sai) -> fallback mặc định.
    if asr_preset not in (CFG.whisper.presets or {}):
        asr_preset = CFG.whisper.default_preset or "cpu"

    if vram and vram >= cfg.omnivoice_min_vram_gb:
        tts_engine = "omnivoice"
        batch = pick_batch_size_for_vram(vram, cfg.omnivoice_batch_by_vram)
        notes.append(f"VRAM {vram:g}GB đủ cho OmniVoice, batch {batch}.")
    else:
        tts_engine = "supertonic"
        batch = 0
        if vram:
            notes.append(
                f"VRAM {vram:g}GB dưới ngưỡng {cfg.omnivoice_min_vram_gb:g}GB của OmniVoice — dùng Supertonic (CPU)."
            )
        else:
            notes.append("Không có GPU NVIDIA — dùng Supertonic (CPU).")
    notes.append(f"{threads} thread CPU cho Whisper/Supertonic.")

    return {
        "asr_preset": asr_preset,
        "tts_engine": tts_engine,
        "omnivoice_batch_size": batch,
        "whisper_cpu_threads": threads,
        "supertonic_intra_op_threads": threads,
        "availability": hardware_availability(ram, vram, cfg=cfg),
        "notes": notes,
    }


def hardware_report() -> dict:
    """Payload cho GET /api/hardware: máy phát hiện được + đề xuất tương ứng."""
    ram_gb = detect_ram_gb()
    gpu = detect_gpu()
    cpu_cores = detect_cpu_cores()
    return {
        "detected": {"ram_gb": ram_gb, "gpu": gpu, "cpu_cores": cpu_cores},
        "recommendation": recommend_setup(
            ram_gb, gpu.get("vram_gb") if gpu else None, cpu_cores,
        ),
    }
