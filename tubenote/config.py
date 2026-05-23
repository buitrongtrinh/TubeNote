"""Load config.yaml + .env. Env vars override YAML."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

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

@dataclass
class TranscriptCfg:
    default_languages: List[str]


@dataclass
class CookiesCfg:
    dir: Optional[Path]
    single_file: Optional[Path]


@dataclass
class AgentCfg:
    max_iterations: int


@dataclass
class RagCfg:
    similarity_threshold: float
    fetch_k: int      # mỗi retriever lấy bao nhiêu để fuse
    final_k: int      # sau RRF, cap trả về cho LLM
    fallback_k: int


@dataclass
class EmbeddingCfg:
    model: str           # "auto" hoặc Ollama tag cụ thể
    base_url: str


@dataclass
class WebSearchCfg:
    enabled: bool
    base_url: str
    max_results: int
    auto_threshold: float


@dataclass
class UICfg:
    title: str
    page_icon: str
    caption: str


@dataclass
class AppCfg:
    llm: LLMCfg
    transcript: TranscriptCfg
    cookies: CookiesCfg
    agent: AgentCfg
    ui: UICfg
    paths: PathsCfg
    rag: RagCfg
    embedding: EmbeddingCfg
    web_search: WebSearchCfg

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

    ag = raw.get("agent", {})
    agent = AgentCfg(max_iterations=int(ag.get("max_iterations", 6)))

    ui_raw = raw.get("ui", {})
    ui = UICfg(
        title=ui_raw.get("title", "YouTube Summarizer"),
        page_icon=ui_raw.get("page_icon", "🎬"),
        caption=ui_raw.get("caption", ""),
    )
    paths_raw = raw.get("paths", {})

    paths = PathsCfg(
        audio_dir=_abs(paths_raw.get("audio_dir") or "output/audio"),
        video_dir=_abs(paths_raw.get("video_dir") or "output/video"),
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
        model=os.getenv("EMBEDDING_MODEL") or emb_raw.get("model", "auto"),
        base_url=os.getenv("OLLAMA_BASE_URL") or emb_raw.get("base_url", "http://localhost:11434"),
    )

    ws_raw = raw.get("web_search", {})
    web_search = WebSearchCfg(
        enabled=bool(ws_raw.get("enabled", True)),
        base_url=os.getenv("SEARXNG_URL") or ws_raw.get("base_url", "http://localhost:8888"),
        max_results=int(ws_raw.get("max_results", 5)),
        auto_threshold=float(ws_raw.get("auto_threshold", 0.55)),
    )

    return AppCfg(
        llm=llm, transcript=transcript, cookies=cookies, agent=agent,
        ui=ui, paths=paths, rag=rag, embedding=embedding, web_search=web_search,
    )


CFG = load()
