"""Duration and speech-density checks for dubbing segments."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TtsDensityPolicy:
    source_units_per_sec: float = 6.0
    min_units_per_sec: float = 3.2
    target_units_per_sec: float = 4.5
    max_units_per_sec: float = 5.2
    tolerance_ratio: float = 0.4
    tolerance_min: int = 3


def _tts_density_policy() -> TtsDensityPolicy:
    try:
        from backend.config import CFG
        # Backward-compatible config location. The policy is now used by both
        # TTS engines, but the YAML key predates that refactor.
        policy = CFG.tts.omnivoice_budget
        return TtsDensityPolicy(
            source_units_per_sec=float(policy.source_units_per_sec),
            min_units_per_sec=float(policy.min_units_per_sec),
            target_units_per_sec=float(policy.target_units_per_sec),
            max_units_per_sec=float(policy.max_units_per_sec),
            tolerance_ratio=float(policy.tolerance_ratio),
            tolerance_min=int(policy.tolerance_min),
        )
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
    return TtsDensityPolicy()


def count_spoken_units(text: str) -> int:
    """Đếm số 'tiếng' (âm tiết) TTS sẽ đọc trong ``text``.

    Token có dấu tiếng Việt tính 1 âm tiết/token (chính tả tiếng Việt viết mỗi
    âm tiết cách nhau bằng khoảng trắng). Token thuần ASCII giữ nguyên dạng
    (từ tiếng Anh chưa được ``text_normalizer`` xử lý, vd "machine",
    "container" — không phải acronym viết hoa nên không bị đánh vần) được ước
    lượng qua số cụm nguyên âm thay vì tính cứng 1, vì đây thường là từ đa âm
    tiết. Heuristic không tuyệt đối chính xác (vd lệch dư 1 với "machine" do
    quy tắc "silent e" tiếng Anh) nhưng lệch dư là hướng an toàn hơn: audio dư
    giây được ``fit_to_slot`` cắt khoảng lặng êm, còn thiếu giây dễ ép TTS nói
    nhanh/mất chữ.
    """
    tokens = re.findall(r"[\wÀ-ỹ]+", text.replace("-", " "), flags=re.UNICODE)
    total = 0
    for token in tokens:
        if re.search(r"[À-ỹ]", token):
            total += 1
        else:
            total += max(1, len(re.findall(r"[aeiouyAEIOUY]+", token)))
    return total


def natural_duration_seconds(text: str, policy: TtsDensityPolicy | None = None) -> float:
    """Ước tính số giây câu này cần để nói ở tốc độ tự nhiên (không ép nhanh/chậm).

    Dùng chung cho cả 2 engine để tính phần "dư/thiếu" so với slot gốc: OmniVoice
    cộng phần dư vào ``generation_duration`` (xem ``engines/omnivoice.py``),
    Supertonic quy đổi ra ``speed`` cần thiết (xem ``dubbing.py::resolve_tts_config``).
    """
    policy = policy or _tts_density_policy()
    return count_spoken_units(text) / policy.target_units_per_sec


def estimate_expansion_units(
    text: str,
    engine: str = "supertonic",
    glossary: dict | None = None,
) -> int:
    """Số tiếng PHÁT SINH khi TTS đọc các token giữ-nguyên-dạng-viết.

    Chạy thử cascade chuẩn hoá trên ``text`` (nguồn hoặc bản dịch đều được, vì
    số/đơn vị/acronym được giữ verbatim qua bước dịch) rồi lấy chênh lệch số
    tiếng. Dùng để trừ trước vào budget ở bước tạo prompt.
    """
    from backend.services.dubbing.text_normalizer import normalize_for_engine

    if not text:
        return 0
    expanded, _ = normalize_for_engine(text, engine, glossary=glossary)
    return max(0, count_spoken_units(expanded) - count_spoken_units(text))


def tts_density_check(
    text: str,
    *,
    duration: float | None = None,
    budget: int | None = None,
) -> dict:
    """Return duration-aware length metadata for TTS pass-through text."""
    policy = _tts_density_policy()
    if duration is None and budget:
        duration = max(0.1, float(budget) / policy.source_units_per_sec)
    if duration is None or duration <= 0:
        return {
            "duration": None,
            "density": None,
            "min_units": None,
            "target_units": None,
            "max_units": None,
            "base_max_units": None,
            "tolerance_units": None,
            "warnings": [],
            "errors": [],
        }

    units = count_spoken_units(text)
    min_units = max(1, math.floor(duration * policy.min_units_per_sec))
    target_units = max(2, round(duration * policy.target_units_per_sec))
    base_max_units = max(target_units, math.floor(duration * policy.max_units_per_sec))
    tolerance_units = max(
        policy.tolerance_min,
        math.ceil(base_max_units * policy.tolerance_ratio),
    )
    max_units = base_max_units + tolerance_units
    density = units / duration
    warnings: list[str] = []
    # Vượt ngân sách chỉ là CẢNH BÁO, không chặn dub: TTS vẫn đọc được (audio_fit
    # sẽ nén tempo), nhưng nghe nhanh/gấp nên báo để người dùng cân nhắc rút gọn.
    if units > max_units:
        warnings.append(
            f"Bản đọc có {units} tiếng trong {duration:.1f}s, vượt gợi ý {max_units} tiếng — "
            f"TTS sẽ đọc nhanh/gấp. Nên rút gọn còn khoảng {target_units}-{max_units} tiếng."
        )
    return {
        "duration": round(duration, 3),
        "density": round(density, 3),
        "min_units": min_units,
        "target_units": target_units,
        "max_units": max_units,
        "base_max_units": base_max_units,
        "tolerance_units": tolerance_units,
        "warnings": warnings,
        "errors": [],
    }

