"""Load config.yaml + .env. Env vars override YAML."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# config.yaml nằm cạnh file này, trong backend/. PROJECT_ROOT vẫn là repo root
# (để neo data/ và .env).
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class LLMCfg:
    provider: str
    providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def provider_opts(self, name: str) -> Dict[str, Any]:
        """LLM builder kwargs cho provider — đã strip ``model``/``models``
        (routing info, không phải LLM constructor kwargs)."""
        opts = dict(self.providers.get(name, {}))
        opts.pop("model", None)
        opts.pop("models", None)
        return opts

    def provider_models(self, name: str) -> List[str]:
        """Priority list models cho provider (top → bottom).

        Backward compat: nếu config dùng ``model: X`` (single) → wrap thành list.
        """
        cfg = self.providers.get(name, {})
        if isinstance(cfg.get("models"), list) and cfg["models"]:
            return list(cfg["models"])
        if cfg.get("model"):
            return [cfg["model"]]
        return []

@dataclass
class PathsCfg:
    audio_dir: Path
    video_dir: Path
    metadata_dir: Path
    subtitles_dir: Path
    video_sub_dir: Path
    chroma_dir: Path
    glossary_file: Path


@dataclass
class WhisperCfg:
    engine: str            # "faster" hoặc "openai"
    vi_model: str           # multilingual cho VN
    en_model: str           # English-only
    initial_prompt: str     # vocab hint cho acronyms tech
    default_preset: str
    presets: Dict[str, Dict[str, Any]]
    # Dựng lại "câu" từ word_timestamps thay vì dùng thẳng ranh giới segment
    # thô của Whisper — xem transcript_whisper.py::_split_into_entries.
    sentence_pause_alpha: float
    sentence_max_words: int
    sentence_min_words: int


@dataclass
class TranslationCfg:
    model: str              # LLM dịch transcript → VN
    manual_batch_size: int
    api_batch_size: int
    api_min_batch_size: int
    api_max_chars_per_batch: int
    api_concurrency: int
    api_job_timeout_sec: int


@dataclass
class OmniVoiceBudgetCfg:
    source_units_per_sec: float
    min_units_per_sec: float
    target_units_per_sec: float
    max_units_per_sec: float
    tolerance_ratio: float
    tolerance_min: int


@dataclass
class SupertonicEngineCfg:
    """Tham số tuning cho engine Supertonic — trước nằm rải trong TTS_POLICIES."""
    num_step: int
    wsola_limit: float
    speed_alpha: float
    merge_max_chars: int
    output_speed: float
    # Thread ONNX Runtime (intra_op); 0 = auto để ORT tự chọn theo core.
    intra_op_threads: int


@dataclass
class OmniVoiceEngineCfg:
    """Tham số tuning cho engine OmniVoice — trước nằm rải trong TTS_POLICIES."""
    num_step: int
    wsola_limit: float
    batch_size: int
    fit_audio: bool
    generation_delta_alpha: float
    generation_delta_min: float
    merge_max_chars: int
    output_speed: float
    postprocess_output: bool


@dataclass
class AudioFitCfg:
    """Tham số cắt khoảng lặng dùng chung cho cả 2 engine (audio_fit.py)."""
    silence_min_pause: float
    silence_db: float
    protect_ratio: float          # top N% run lặng dài nhất được bảo vệ nhiều hơn
    protect_min_pause_multiplier: float  # sàn cho nhóm được bảo vệ = min_pause * hệ số này
    # Ngưỡng phát hiện "khoảng tiếng nói thực sự" (active_range_samples) —
    # khác mục đích với silence_db ở trên (đó là để CẮT bớt cho vừa slot, còn
    # đây là để NEO điểm bắt đầu/kết thúc tiếng nói, dùng cho timing hiển thị).
    active_range_silence_db: float
    active_range_head_ms: int
    active_range_tail_ms: int


@dataclass
class MixCfg:
    """Tham số trộn audio dub + nền gốc/background khi mux video."""
    original_volume: float
    dub_volume_no_background: float
    background_volume: float
    dub_volume_with_background: float
    loudnorm_i: float
    loudnorm_tp: float
    loudnorm_lra: float


@dataclass
class HardwareCfg:
    """Bảng tier phần cứng: (RAM, VRAM) -> ASR preset / TTS engine / batch / threads."""
    omnivoice_min_vram_gb: float
    omnivoice_batch_by_vram: Dict[float, int]
    asr_gpu_by_vram: Dict[float, str]
    asr_cpu_by_ram: Dict[float, str]
    max_auto_threads: int


@dataclass
class TtsCfg:
    default_model: str
    models: List[str]
    omnivoice_model: str
    omnivoice_models: List[str]
    omnivoice_voices: List[Dict[str, Any]]
    omnivoice_budget: OmniVoiceBudgetCfg
    supertonic: SupertonicEngineCfg
    omnivoice: OmniVoiceEngineCfg


@dataclass
class TranscriptCfg:
    default_languages: List[str]


@dataclass
class CookiesCfg:
    dir: Optional[Path]
    single_file: Optional[Path]


@dataclass
class RagCfg:
    similarity_threshold: float
    fetch_k: int      # mỗi retriever lấy bao nhiêu để fuse
    final_k: int      # sau RRF, cap trả về cho LLM
    fallback_k: int


@dataclass
class EmbeddingCfg:
    provider: str        # "huggingface"
    model: str
    device: str
    normalize: bool
    local_files_only: bool


@dataclass
class WebSearchCfg:
    enabled: bool
    base_url: str
    max_results: int
    auto_threshold: float


@dataclass
class AppCfg:
    llm: LLMCfg
    transcript: TranscriptCfg
    cookies: CookiesCfg
    paths: PathsCfg
    rag: RagCfg
    embedding: EmbeddingCfg
    web_search: WebSearchCfg
    whisper: WhisperCfg
    translation: TranslationCfg
    tts: TtsCfg
    audio_fit: AudioFitCfg
    mix: MixCfg
    hardware: HardwareCfg

def _abs(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load() -> AppCfg:

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    llm_raw = raw.get("llm", {})
    provider = os.getenv("LLM_PROVIDER", llm_raw.get("provider", "google"))
    providers_raw = {k: v for k, v in llm_raw.items() if isinstance(v, dict)}
    if model_env := os.getenv("LLM_MODEL"):
        if provider in providers_raw:
            # Env override → force single specific model, clear priority list
            providers_raw[provider].pop("models", None)
            providers_raw[provider]["model"] = model_env
    llm = LLMCfg(provider=provider, providers=providers_raw)

    tr = raw.get("transcript", {})
    transcript = TranscriptCfg(default_languages=list(tr.get("default_languages", ["vi", "en"])))

    ck = raw.get("cookies", {})
    cookies = CookiesCfg(
        dir=_abs(os.getenv("YT_COOKIES_DIR") or ck.get("dir")),
        single_file=_abs(os.getenv("YT_COOKIES_PATH") or ck.get("single_file")),
    )

    paths_raw = raw.get("paths", {})
    paths = PathsCfg(
        audio_dir=_abs(paths_raw.get("audio_dir") or "data/audio"),
        video_dir=_abs(paths_raw.get("video_dir") or "data/video"),
        metadata_dir=_abs(paths_raw.get("metadata_dir") or "data/metadata"),
        subtitles_dir=_abs(paths_raw.get("subtitles_dir") or "data/subtitles"),
        video_sub_dir=_abs(paths_raw.get("video_sub_dir") or "data/video_sub"),
        chroma_dir=_abs(paths_raw.get("chroma_dir") or "data/chroma"),
        glossary_file=_abs(paths_raw.get("glossary_file") or "data/glossary.json"),
    )

    ws_raw_whisper = raw.get("whisper", {})
    whisper_presets = ws_raw_whisper.get("presets") if isinstance(ws_raw_whisper.get("presets"), dict) else {}
    whisper = WhisperCfg(
        engine=str(ws_raw_whisper.get("engine", "faster")),
        vi_model=str(ws_raw_whisper.get("vi_model", "openai/whisper-large-v3-turbo")),
        en_model=str(ws_raw_whisper.get("en_model", "openai/whisper-large-v3-turbo")),
        initial_prompt=str(ws_raw_whisper.get("initial_prompt", "")).strip(),
        default_preset=str(ws_raw_whisper.get("default_preset", "gpu")),
        presets={key: dict(value) for key, value in whisper_presets.items() if isinstance(value, dict)},
        sentence_pause_alpha=float(ws_raw_whisper.get("sentence_pause_alpha", 0.02)),
        sentence_max_words=int(ws_raw_whisper.get("sentence_max_words", 16)),
        sentence_min_words=int(ws_raw_whisper.get("sentence_min_words", 2)),
    )

    tr_raw = raw.get("translation", {})
    translation = TranslationCfg(
        model=str(tr_raw.get("model", "gemini-2.5-flash-lite")),
        manual_batch_size=int(tr_raw.get("manual_batch_size", 50)),
        api_batch_size=int(tr_raw.get("api_batch_size", 25)),
        api_min_batch_size=int(tr_raw.get("api_min_batch_size", 5)),
        api_max_chars_per_batch=int(tr_raw.get("api_max_chars_per_batch", 4000)),
        api_concurrency=int(tr_raw.get("api_concurrency", 8)),
        api_job_timeout_sec=int(tr_raw.get("api_job_timeout_sec", 300)),
    )

    tts_raw = raw.get("tts", {})
    tts_models = list(tts_raw.get("models") or ["M5", "F5"])
    tts_default = os.getenv("TTS_MODEL") or str(tts_raw.get("default_model") or tts_models[0])
    if tts_default not in tts_models:
        tts_models.insert(0, tts_default)
    omni_raw = tts_raw.get("omnivoice") or {}
    budget_raw = omni_raw.get("budget") or {}
    omni_budget = OmniVoiceBudgetCfg(
        source_units_per_sec=float(budget_raw.get("source_units_per_sec", 6.0)),
        min_units_per_sec=float(budget_raw.get("min_units_per_sec", 3.2)),
        target_units_per_sec=float(budget_raw.get("target_units_per_sec", 4.5)),
        max_units_per_sec=float(budget_raw.get("max_units_per_sec", 5.2)),
        tolerance_ratio=float(budget_raw.get("tolerance_ratio", 0.4)),
        tolerance_min=int(budget_raw.get("tolerance_min", 3)),
    )
    omni_models = list(omni_raw.get("models") or ["k2-fsa/OmniVoice"])
    omni_default = os.getenv("OMNIVOICE_MODEL") or str(omni_raw.get("default_model") or omni_models[0])
    if omni_default not in omni_models:
        omni_models.insert(0, omni_default)
    omni_voices = []
    for voice in omni_raw.get("voices") or []:
        if not isinstance(voice, dict):
            continue
        item = dict(voice)
        if item.get("reference_audio"):
            item["reference_audio"] = str(_abs(item["reference_audio"]))
        omni_voices.append(item)

    supertonic_raw = tts_raw.get("supertonic") or {}
    supertonic_engine = SupertonicEngineCfg(
        num_step=int(supertonic_raw.get("num_step", 8)),
        wsola_limit=float(supertonic_raw.get("wsola_limit", 1.05)),
        speed_alpha=float(supertonic_raw.get("speed_alpha", 1.2)),
        merge_max_chars=int(supertonic_raw.get("merge_max_chars", 0)),
        output_speed=float(supertonic_raw.get("output_speed", 1.0)),
        intra_op_threads=int(supertonic_raw.get("intra_op_threads", 0)),
    )
    omni_engine = OmniVoiceEngineCfg(
        num_step=int(omni_raw.get("num_step", 32)),
        wsola_limit=float(omni_raw.get("wsola_limit", 1.05)),
        batch_size=int(omni_raw.get("batch_size", 4)),
        fit_audio=bool(omni_raw.get("fit_audio", True)),
        generation_delta_alpha=float(omni_raw.get("generation_delta_alpha", 1.2)),
        generation_delta_min=float(omni_raw.get("generation_delta_min", 0.3)),
        merge_max_chars=int(omni_raw.get("merge_max_chars", 0)),
        output_speed=float(omni_raw.get("output_speed", 1.0)),
        postprocess_output=bool(omni_raw.get("postprocess_output", False)),
    )
    tts = TtsCfg(
        default_model=tts_default,
        models=tts_models,
        omnivoice_model=omni_default,
        omnivoice_models=omni_models,
        omnivoice_voices=omni_voices,
        omnivoice_budget=omni_budget,
        supertonic=supertonic_engine,
        omnivoice=omni_engine,
    )

    audio_fit_raw = raw.get("audio_fit", {})
    audio_fit = AudioFitCfg(
        silence_min_pause=float(audio_fit_raw.get("silence_min_pause", 0.04)),
        silence_db=float(audio_fit_raw.get("silence_db", -35.0)),
        protect_ratio=float(audio_fit_raw.get("protect_ratio", 0.3)),
        protect_min_pause_multiplier=float(audio_fit_raw.get("protect_min_pause_multiplier", 3.0)),
        active_range_silence_db=float(audio_fit_raw.get("active_range_silence_db", -45.0)),
        active_range_head_ms=int(audio_fit_raw.get("active_range_head_ms", 40)),
        active_range_tail_ms=int(audio_fit_raw.get("active_range_tail_ms", 180)),
    )

    mix_raw = raw.get("mix", {})
    mix = MixCfg(
        original_volume=float(mix_raw.get("original_volume", 0.05)),
        dub_volume_no_background=float(mix_raw.get("dub_volume_no_background", 1.6)),
        background_volume=float(mix_raw.get("background_volume", 1.0)),
        dub_volume_with_background=float(mix_raw.get("dub_volume_with_background", 1.35)),
        loudnorm_i=float(mix_raw.get("loudnorm_i", -16)),
        loudnorm_tp=float(mix_raw.get("loudnorm_tp", -1.5)),
        loudnorm_lra=float(mix_raw.get("loudnorm_lra", 11)),
    )

    hw_raw = raw.get("hardware", {}) or {}

    def _tier_table(key: str, cast, default: dict) -> dict:
        """Bảng {ngưỡng: giá trị} từ yaml; khóa ép float, giá trị ép ``cast``."""
        table_raw = hw_raw.get(key)
        table: dict = {}
        if isinstance(table_raw, dict):
            for k, v in table_raw.items():
                try:
                    table[float(k)] = cast(v)
                except (TypeError, ValueError):
                    continue
        return table or default

    hardware = HardwareCfg(
        omnivoice_min_vram_gb=float(hw_raw.get("omnivoice_min_vram_gb", 5)),
        omnivoice_batch_by_vram=_tier_table(
            "omnivoice_batch_by_vram", int, {5.0: 4, 12.0: 6, 16.0: 8},
        ),
        asr_gpu_by_vram=_tier_table(
            "asr_gpu_by_vram", str, {2.0: "gpu_small", 3.5: "gpu"},
        ),
        asr_cpu_by_ram=_tier_table(
            "asr_cpu_by_ram", str, {0.0: "cpu_tiny", 4.0: "cpu_base", 6.0: "cpu"},
        ),
        max_auto_threads=int(hw_raw.get("max_auto_threads", 16)),
    )

    rag_raw = raw.get("rag", {})
    rag = RagCfg(
        similarity_threshold=float(rag_raw.get("similarity_threshold", 0.55)),
        fetch_k=int(rag_raw.get("fetch_k", 30)),
        final_k=int(rag_raw.get("final_k", 10)),
        fallback_k=int(rag_raw.get("fallback_k", 3)),
    )

    emb_raw = raw.get("embedding", {})
    embedding = EmbeddingCfg(
        provider=os.getenv("EMBEDDING_PROVIDER") or emb_raw.get("provider", "huggingface"),
        model=os.getenv("EMBEDDING_MODEL") or emb_raw.get("model", "BAAI/bge-m3"),
        device=os.getenv("EMBEDDING_DEVICE") or emb_raw.get("device", "auto"),
        normalize=bool(emb_raw.get("normalize", True)),
        local_files_only=bool(emb_raw.get("local_files_only", False)),
    )

    ws_raw = raw.get("web_search", {})
    web_search = WebSearchCfg(
        enabled=bool(ws_raw.get("enabled", True)),
        base_url=os.getenv("SEARXNG_URL") or ws_raw.get("base_url", "http://localhost:8888"),
        max_results=int(ws_raw.get("max_results", 5)),
        auto_threshold=float(ws_raw.get("auto_threshold", 0.55)),
    )

    return AppCfg(
        llm=llm, transcript=transcript, cookies=cookies,
        paths=paths, rag=rag, embedding=embedding, web_search=web_search,
        whisper=whisper, translation=translation, tts=tts,
        audio_fit=audio_fit, mix=mix, hardware=hardware,
    )


CFG = load()
