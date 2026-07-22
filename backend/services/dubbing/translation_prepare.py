"""Prepare validated translation text for TTS generation."""
from __future__ import annotations

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
from backend.services.dubbing.translation_parser import parse_translation_result


NORMALIZATION_VERSION = 4


def normalize_translation_segment(
    segment: dict,
    engine: str,
    budget: int | None = None,
    duration: float | None = None,
    pronunciation_map: dict | None = None,
) -> dict:
    text_vi = canonicalize_text(str(segment.get("vi") or segment.get("text_vi") or ""))
    # Cascade sinh text_tts: pronunciation map của user (ưu tiên cao nhất, chủ
    # yếu ở luồng regenerate) → glossary + rule expand (số, đơn vị, acronym...).
    mapped, cleaned_map = apply_pronunciation_map(text_vi, pronunciation_map)
    text_tts, applied_rules = normalize_for_tts(mapped, glossary=load_glossary())
    if cleaned_map:
        applied_rules = ["pronunciation_map", *applied_rules]
    warnings: list[str] = []
    written_units = count_spoken_units(text_vi)
    spoken_units = count_spoken_units(text_tts)
    normalization_expansion = max(0, spoken_units - written_units)
    budget_tolerance = 0
    allowed_units = None
    # Budget trong prompt đã bị trừ surcharge lúc tạo prompt; bù lại expansion
    # thực tế để suy ngược đúng duration gốc trước khi so mật độ.
    effective_budget = (budget + normalization_expansion) if budget else budget
    density_meta = tts_density_check(text_tts, duration=duration, budget=effective_budget)
    warnings.extend(density_meta.get("warnings") or [])
    errors = list(density_meta.get("errors") or [])
    result = {
        "vi": text_vi,
        "tts": text_tts,
        "normalization": {
            "engine": engine,
            "version": NORMALIZATION_VERSION,
            "applied_rules": applied_rules,
            "written_units": written_units,
            "spoken_units": spoken_units,
            "budget": budget,
            "budget_tolerance": budget_tolerance,
            "normalization_expansion": normalization_expansion,
            "allowed_units": allowed_units,
            "duration": density_meta.get("duration"),
            "density": density_meta.get("density"),
            "min_units": density_meta.get("min_units"),
            "target_units": density_meta.get("target_units"),
            "max_units": density_meta.get("max_units"),
            "base_max_units": density_meta.get("base_max_units"),
            "tolerance_units": density_meta.get("tolerance_units"),
            "warnings": warnings,
            "errors": errors,
        },
    }
    if cleaned_map:
        result["pronunciation_map"] = cleaned_map
    return result


def renormalize_segments(
    segments: list[dict],
    engine: str,
    budgets: list[int] | None = None,
    durations: list[float] | None = None,
    pronunciation_map: dict | None = None,
) -> list[dict]:
    return [
        normalize_translation_segment(
            segment if isinstance(segment, dict) else {"vi": str(segment)},
            engine,
            budgets[index] if budgets and index < len(budgets) else None,
            durations[index] if durations and index < len(durations) else None,
            (
                pronunciation_map
                if pronunciation_map is not None
                else (
                    segment.get("pronunciation_map")
                    if isinstance(segment, dict) and isinstance(segment.get("pronunciation_map"), dict)
                    else None
                )
            ),
        )
        for index, segment in enumerate(segments)
    ]


def prepare_translations_for_tts(
    text: str,
    batch_id: str,
    engine: str = "supertonic",
    budgets: list[int] | None = None,
    pronunciation_map: dict | None = None,
) -> list[dict]:
    batch_name, translations = parse_translation_result(text)
    # Header con khi retry-chia-nhỏ có hậu tố "_r{depth}_{phần}" (xem
    # splitPromptForRetry ở frontend) — vẫn coi là khớp batch gốc.
    if batch_name != batch_id and not batch_name.startswith(f"{batch_id}_r"):
        raise ValueError(f"Batch ID không khớp: nhận '{batch_name}', cần '{batch_id}'")
    if not translations:
        raise ValueError("Không tìm thấy dòng dịch nào")

    result = []
    for index, item in enumerate(translations):
        if not item[1]:
            raise ValueError(f"Dòng {item[0]} có phần dịch rỗng")
        if item[0] - 1 != index:
            raise ValueError(f"Dòng {item[0]} không đúng vị trí (mong đợi {index + 1})")
        budget = budgets[index] if budgets and index < len(budgets) else None
        result.append(normalize_translation_segment(
            {"vi": item[1]},
            engine,
            budget,
            pronunciation_map=pronunciation_map,
        ))
    return result
