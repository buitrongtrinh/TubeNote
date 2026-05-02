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
        return dict(self.providers.get(name, {}))


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

    return AppCfg(llm=llm, transcript=transcript, cookies=cookies, agent=agent, ui=ui)


CFG = load()
