"""Route cho luồng dubbing: nạp video, kiểm tra bản dịch, chạy dub.

Cả /load (Whisper có thể chậm) và /dub (TTS chậm) đều chạy NỀN qua jobs chung
rồi client polling /load/{id} · /dub/{id}.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from backend.pipeline import dubbing
from backend.services.hardware import detect_cpu_cores, hardware_report, recommend_setup
from backend.workers import jobs

router = APIRouter(prefix="/api", tags=["dubbing"])


@router.get("/hardware")
def hardware():
    """RAM/VRAM/cores máy chủ + bộ tham số đề xuất cho màn hình tạo lồng tiếng."""
    return hardware_report()


@router.get("/hardware/recommend")
def hardware_recommend(ram_gb: float = 0, vram_gb: float = 0):
    """Bộ tham số đề xuất cho RAM/VRAM người dùng nhập tay (cores vẫn lấy từ máy)."""
    return recommend_setup(ram_gb, vram_gb, detect_cpu_cores())


class LoadReq(BaseModel):
    url: str
    engine: str = "supertonic"
    speech_preset: str | None = None


class ValidateReq(BaseModel):
    prompt_index: int
    response: str
    expected: int = 0   # số dòng cần có (frontend đếm từ prompt)
    engine: str = "supertonic"
    budgets: list[int] = Field(default_factory=list)


class TranslateReq(BaseModel):
    prompt_index: int
    prompt: str = Field(min_length=1)
    provider: str = "openai"
    model: str = ""


class DubReq(BaseModel):
    url: str
    # mỗi segment là {"vi": <hiển thị>, "tts": <đọc>} từ /api/validate
    segments: list[dict]
    tts: dict | None = None
    tts_model: str | None = None


class RegenerateSegmentReq(BaseModel):
    text_vi: str = Field(min_length=1)
    pronunciation_map: dict[str, str] = Field(default_factory=dict)
    num_step: int = 48


def _load_progress(stage: str) -> int:
    stage = stage or ""
    if "Whisper STT thành công" in stage:
        return 100
    if "fallback Whisper" in stage or "Chuẩn bị Whisper STT" in stage:
        return 0
    if "tải model Whisper" in stage or "chuẩn bị nhận diện giọng nói" in stage:
        return 0
    if "progressive" in stage:
        return 0
    if "nhận diện giọng nói" in stage:
        match = re.search(r"(\d{1,3})%", stage)
        if match:
            return max(0, min(100, int(match.group(1))))
        return 0
    if "Lấy thông tin video" in stage or "Tạo prompts" in stage:
        return 100
    return 0


def _job_or_404(job_id: str) -> dict:
    job = jobs.status(job_id)
    if not job:
        raise HTTPException(404, "Không tìm thấy job")
    return job


# ── Load (chạy nền vì Whisper STT có thể vài phút) ───────────────────────────────
@router.post("/load")
def load(req: LoadReq, bg: BackgroundTasks):
    """Khởi động nạp video chạy nền, trả job_id. Kết quả ở result:
    {video_id, already_dubbed, metadata, prompts}."""
    job_id = jobs.create()
    bg.add_task(
        jobs.run, job_id,
        lambda update: dubbing.load_video(
            req.url,
            on_progress=lambda s: update(stage=s, progress=_load_progress(s)),
            tts_engine=req.engine,
            whisper_preset=req.speech_preset,
        ),
    )
    return {"job_id": job_id}


@router.get("/load/{job_id}")
def load_status(job_id: str):
    return _job_or_404(job_id)


@router.post("/validate")
def validate(req: ValidateReq):
    """Kiểm tra response ChatGPT → segments cho TTS."""
    return dubbing.validate_response(
        req.prompt_index,
        req.response,
        req.expected,
        engine=req.engine,
        budgets=req.budgets,
    )


@router.get("/translation/models")
def translation_models():
    """Danh sách LLM provider/model dùng cho dịch transcript bằng API."""
    from backend.config import CFG

    labels = {
        "deepseek": "DeepSeek",
        "google": "Gemini",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
    }
    providers = []
    for provider_id in ("deepseek", "openai", "google", "anthropic"):
        models = CFG.llm.provider_models(provider_id)
        if not models:
            continue
        providers.append({
            "id": provider_id,
            "label": labels.get(provider_id, provider_id),
            "models": models,
            "default_model": models[0],
        })
    default_provider = "deepseek" if any(item["id"] == "deepseek" for item in providers) else (
        CFG.llm.provider if any(item["id"] == CFG.llm.provider for item in providers) else (providers[0]["id"] if providers else "")
    )
    return {
        "default_provider": default_provider,
        "providers": providers,
    }


@router.post("/translate")
def translate(req: TranslateReq, bg: BackgroundTasks):
    """Dịch một prompt bằng LLM API, chạy nền để tránh request timeout."""
    job_id = jobs.create()
    bg.add_task(
        jobs.run,
        job_id,
        lambda update: dubbing.translate_prompt_with_api(
            req.prompt_index,
            req.prompt,
            provider=req.provider,
            model=req.model,
            report=lambda p, s: update(progress=p, stage=s),
        ),
    )
    return {"job_id": job_id}


@router.get("/translate/{job_id}")
def translate_status(job_id: str):
    return _job_or_404(job_id)


@router.get("/tts/models")
def tts_models():
    """Danh sách TTS model/voice frontend có thể chọn."""
    return dubbing.list_tts_models()


# ── Dub (chạy nền vì TTS chậm) ───────────────────────────────────────────────────
@router.post("/dub")
def dub(req: DubReq, bg: BackgroundTasks):
    """Khởi động dubbing chạy nền, trả job_id. Kết quả result = video_id."""
    job_id = jobs.create()
    bg.add_task(
        jobs.run, job_id,
        lambda update: dubbing.run_dubbing(
            req.url, req.segments, tts=req.tts, tts_model=req.tts_model,
            report=lambda p, s: update(progress=p, stage=s),
        ),
    )
    return {"job_id": job_id}


@router.get("/dub/{job_id}")
def dub_status(job_id: str):
    return _job_or_404(job_id)


@router.post("/video/{vid}/segments/{segment_index}/regenerate")
def regenerate_segment(
    vid: str,
    segment_index: int,
    req: RegenerateSegmentReq,
    bg: BackgroundTasks,
):
    """Regenerate one segment and remux the existing dubbed video."""
    job_id = jobs.create()
    bg.add_task(
        jobs.run,
        job_id,
        lambda update: dubbing.regenerate_segment(
            vid,
            segment_index,
            req.text_vi,
            pronunciation_map=req.pronunciation_map,
            num_step=req.num_step,
            report=lambda p, s: update(progress=p, stage=s),
        ),
    )
    return {"job_id": job_id}


@router.post("/video/{vid}/regenerate")
def regenerate_video(
    vid: str,
    bg: BackgroundTasks,
):
    """Regenerate the full dubbed audio/video from current saved subtitles."""
    job_id = jobs.create()
    bg.add_task(
        jobs.run,
        job_id,
        lambda update: dubbing.regenerate_full_dubbing(
            vid,
            report=lambda p, s: update(progress=p, stage=s),
        ),
    )
    return {"job_id": job_id}


@router.get("/regenerate/{job_id}")
def regenerate_status(job_id: str):
    return _job_or_404(job_id)
