"""Route cho RAG hỏi đáp trên video."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from backend.config import CFG
from backend.llm.providers import make_llm
from backend.pipeline.qa import QAPipeline
from backend.services.rag.pipeline import ingest_video_id
from backend.services.rag.summary import generate_summary, load_summary

router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.get("/ping")
def ping():
    return {"ok": True}


@router.get("/models")
def rag_models():
    labels = {
        "deepseek": "DeepSeek",
        "google": "Gemini",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
    }
    providers = []
    for provider_id in ("deepseek", "google", "openai", "anthropic"):
        models = CFG.llm.provider_models(provider_id)
        if not models:
            continue
        providers.append({
            "id": provider_id,
            "label": labels.get(provider_id, provider_id),
            "models": models,
            "default_model": models[0],
        })
    default_provider = CFG.llm.provider if any(item["id"] == CFG.llm.provider for item in providers) else (providers[0]["id"] if providers else "")
    return {
        "default_provider": default_provider,
        "providers": providers,
    }


class RagHistoryMessage(BaseModel):
    role: str
    content: str


class RagAskRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[RagHistoryMessage] = Field(default_factory=list)
    summary: str = ""
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    # "off": không search web. "auto": chỉ search khi RAG yếu (top score dưới
    # hardware.web_search.auto_threshold). "always": search web mọi câu hỏi.
    web_mode: Literal["off", "auto", "always"] = "off"


def _history_messages(items: list[RagHistoryMessage]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in items[-6:]:
        content = item.content.strip()
        if not content:
            continue
        if item.role == "assistant":
            messages.append(AIMessage(content=content[:1200]))
        elif item.role == "user":
            messages.append(HumanMessage(content=content[:800]))
    return messages


def _source_payload(doc) -> dict:
    meta = doc.metadata or {}
    return {
        "start": meta.get("start"),
        "end": meta.get("end"),
        "text": doc.page_content,
        "source": meta.get("source"),
        "segment_start": meta.get("segment_start"),
        "segment_end": meta.get("segment_end"),
    }


def _web_source_payload(result) -> dict:
    return {
        "title": result.title,
        "url": result.url,
        "domain": result.domain,
        "snippet": result.snippet,
    }


@router.post("/video/{vid}/ask")
def ask_video(vid: str, req: RagAskRequest):
    try:
        question = req.question.strip()
        if not question:
            raise ValueError("Câu hỏi không được để trống.")
        ingest_video_id(vid)
        llm = make_llm(provider=req.provider, model=req.model)
        cached_summary = load_summary(vid)
        result = QAPipeline(llm=llm).run(
            video_id=vid,
            question=question,
            history=_history_messages(req.history),
            video_summary=req.summary.strip() or (cached_summary or {}).get("summary", ""),
            web_mode=req.web_mode,
        )
        return {
            "answer": result.answer,
            "sources": [_source_payload(doc) for doc in result.chunks_used],
            "web_sources": [_web_source_payload(item) for item in result.web_results],
            "web_triggered_by": result.web_triggered_by,
            "top_rag_score": result.top_rag_score,
            "cache_usage": result.cache_usage,
        }
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e)) from e


@router.get("/video/{vid}/summary")
def video_summary(
    vid: str,
    provider: str = "deepseek",
    model: str = "deepseek-v4-flash",
    force: bool = False,
):
    try:
        if not force:
            cached = load_summary(vid)
            if cached:
                return cached
        llm = make_llm(provider=provider, model=model)
        return generate_summary(vid, llm=llm, force=force, provider=provider, model=model)
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e)) from e
