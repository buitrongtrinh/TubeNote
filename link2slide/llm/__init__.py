"""Multi-provider LLM factory."""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from ..config import CFG

_BUILDERS = {}


def _get_builder(provider: str):
    if provider in _BUILDERS:
        return _BUILDERS[provider]
    if provider == "google":
        from .google import build as b
    elif provider == "openai":
        from .openai import build as b
    elif provider == "anthropic":
        from .anthropic import build as b
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")
    _BUILDERS[provider] = b
    return b


def make_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **overrides: Any,
) -> BaseChatModel:
    """Build a LangChain chat model for the given provider.

    provider: google | openai | anthropic (default: config.yaml)
    model, api_key, overrides: runtime overrides over config.yaml values.
    """
    prov = provider or CFG.llm.provider
    opts = CFG.llm.provider_opts(prov).copy()
    if model:
        opts["model"] = model
    if api_key:
        opts["api_key"] = api_key
    opts.update(overrides)
    return _get_builder(prov)(**opts)
