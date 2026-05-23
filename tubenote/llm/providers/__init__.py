"""Multi-provider LLM factory.

Mỗi provider có ``models: [...]`` trong config (priority list). Model đầu tiên
là default. User pick cụ thể qua ``model=<name>``, hoặc UI dropdown.
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from ...config import CFG

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
    elif provider == "local":
        from .local import build as b
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
    """Build LLM cho provider.

    Args:
        provider: ``google`` | ``openai`` | ``anthropic`` | ``local``
            (default: ``CFG.llm.provider``).
        model: tên model cụ thể. Nếu ``None`` → dùng model đầu trong
            ``CFG.llm.provider_models(provider)``.
        api_key: override key trong .env.
        **overrides: kwargs khác (temperature, num_ctx, ...).
    """
    prov = provider or CFG.llm.provider
    base_opts = CFG.llm.provider_opts(prov)

    if not model:
        models = CFG.llm.provider_models(prov)
        if not models:
            raise ValueError(
                f"Không có model nào cho provider {prov!r}. "
                f"Set ``models: [...]`` trong config.yaml."
            )
        model = models[0]

    if api_key:
        base_opts["api_key"] = api_key
    base_opts.update(overrides)

    return _get_builder(prov)(model=model, **base_opts)
