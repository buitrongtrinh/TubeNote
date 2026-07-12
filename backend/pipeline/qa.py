"""QA Pipeline — RAG + sliding window memory + optional web search hybrid."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain_community.retrievers import BM25Retriever

from ..config import CFG
from ..services.rag.store import get_all_docs, get_vector_store
from ..services.web.searxng import SearXNGError, WebResult, format_for_llm, web_search


ProgressCallback = Callable[[str], None]
WebMode = Literal["off", "auto", "always"]


def _noop(_: str) -> None:
    pass


def _rrf_fuse(rankings: list[list[Document]], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank). ``k=60`` là default của paper."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            docs[key] = doc
    return [docs[key] for key, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi đáp video. Trả lời bằng tiếng Việt.\n\n"
    "NGUỒN THÔNG TIN (theo thứ tự ưu tiên):\n"
    "1. Chunks từ video là nguồn chính.\n"
    "2. Bản tóm tắt video dùng cho câu hỏi tổng quan.\n"
    "3. Lịch sử chat chỉ dùng để hiểu follow-up.\n"
    "4. Kết quả web search chỉ dùng khi được cung cấp.\n\n"
    "QUY TẮC NỘI DUNG:\n"
    "- Câu hỏi cụ thể về video → ưu tiên chunks.\n"
    "- Câu hỏi tổng quan → ưu tiên summary.\n"
    "- Câu hỏi cập nhật / so sánh / ngoài video → dùng [WEB-N].\n"
    "- Cả video lẫn web đều thiếu → 'Tôi không có thông tin về...'. KHÔNG bịa.\n"
    "- Trả lời súc tích, đi thẳng vấn đề.\n\n"
    "QUY TẮC FORMAT BẮT BUỘC:\n"
    "- UI render Markdown, được dùng **bold**, `code` inline, và gạch đầu dòng '- '.\n"
    "- KHÔNG dùng heading ('#'), bảng, code fence (```), vì panel chat hẹp, các khối đó dễ vỡ layout.\n"
    "- Không dùng LaTeX như \\( ... \\); viết công thức dạng plain text, ví dụ f(x) = wx + b.\n"
    "- Không nhúng citation kiểu '(theo video, [02:10-03:35])' trong câu trả lời; UI sẽ hiển thị nguồn riêng.\n"
    "- Nếu cần liệt kê, dùng tối đa 4 gạch đầu dòng ngắn bằng '- '.\n"
    "- Nếu có mốc thời gian quan trọng, thêm dòng cuối: 'Mốc liên quan: 02:10-03:35, ...'."
)


@dataclass
class QAResult:
    video_id: str
    chunks_used: List[Document]
    answer: str
    url: str = ""
    web_results: List[WebResult] = field(default_factory=list)
    top_rag_score: Optional[float] = None
    web_triggered_by: Optional[str] = None      # "always" | "auto" | None
    cache_usage: dict = field(default_factory=dict)


class QAPipeline:
    """RAG QA với sliding window memory + optional web search.

    Yêu cầu video đã được ``ingest()`` trước. Pipeline chỉ làm retrieval + LLM.
    """

    def __init__(self, llm: BaseChatModel, history_window: int = 10):
        self.llm = llm
        self.history_window = history_window

    def run(
        self,
        video_id: str,
        question: str,
        url: str = "",
        history: Optional[Sequence[BaseMessage]] = None,
        video_summary: str = "",
        on_progress: Optional[ProgressCallback] = None,
        web_mode: WebMode = "off",
    ) -> QAResult:
        progress = on_progress or _noop

        # ── RAG retrieval (with scores cho heuristic auto-mode) ──────────────
        progress("Đang search chunks liên quan trong vector store…")
        docs, top_score = self._retrieve_with_scores(video_id, question)
        progress(f"Lấy được {len(docs)} chunks (top score: {top_score:.3f})")

        # ── Web search decision ───────────────────────────────────────────────
        web_results: List[WebResult] = []
        web_triggered: Optional[str] = None
        should_search = (
            CFG.web_search.enabled and (
                web_mode == "always"
                or (web_mode == "auto" and top_score < CFG.web_search.auto_threshold)
            )
        )
        if should_search:
            web_triggered = web_mode
            reason = "luôn bật" if web_mode == "always" else f"top score {top_score:.2f} < {CFG.web_search.auto_threshold}"
            progress(f"Đang search web (SearXNG) — lý do: {reason}…")
            try:
                web_results = web_search(question)
                progress(f"Web search OK ({len(web_results)} kết quả)")
            except SearXNGError as e:
                progress(f"Web search fail: {e} — tiếp tục chỉ với RAG")

        # ── Build messages ───────────────────────────────────────────────────
        messages = self._build_messages(question, docs, web_results, history, video_summary)

        progress("Đang gọi LLM trả lời…")
        response = self.llm.invoke(messages)
        answer = self._clean_answer(self._extract_text(response))
        cache_usage = self._extract_cache_usage(response)

        return QAResult(
            url=url,
            video_id=video_id,
            chunks_used=docs,
            answer=answer,
            web_results=web_results,
            top_rag_score=top_score,
            web_triggered_by=web_triggered,
            cache_usage=cache_usage,
        )

    # ── Internals ───────────────────────────────────────────────────────────

    def _retrieve_with_scores(self, video_id: str, question: str) -> tuple[List[Document], float]:
        """Hybrid retrieval: dense (Chroma) + sparse (BM25), fuse bằng RRF.

        Trả docs đã fuse + ``top_score`` từ DENSE (cho web auto-trigger heuristic).
        BM25 chỉ match keyword nên không phản ánh "câu hỏi nằm ngoài scope video".
        """
        cfg = CFG.rag
        store = get_vector_store(video_id)
        # Fetch pool rộng (fetch_k) để fusion có cơ hội "cứu" chunks rank thấp ở 1 retriever
        scored = store.similarity_search_with_relevance_scores(question, k=cfg.fetch_k)

        if not scored:
            return [], 0.0

        top_score = scored[0][1]

        # Dense: lọc theo threshold; fallback top-N nếu rỗng
        dense_docs = [d for d, s in scored if s >= cfg.similarity_threshold]
        if not dense_docs:
            dense_docs = [d for d, _ in scored[: cfg.fallback_k]]

        # Sparse: BM25 build in-memory trên toàn corpus (cheap, ~ chục chunks)
        all_docs = get_all_docs(store)
        sparse_docs: List[Document] = []
        if all_docs:
            bm25 = BM25Retriever.from_documents(all_docs)
            bm25.k = cfg.fetch_k
            sparse_docs = bm25.invoke(question)

        # Fuse 2 ranking, cap final_k (< fetch_k) gửi LLM
        fused = _rrf_fuse([dense_docs, sparse_docs])[: cfg.final_k]
        return fused, top_score

    def _build_messages(
        self,
        question: str,
        docs: List[Document],
        web_results: List[WebResult],
        history: Optional[Sequence[BaseMessage]],
        video_summary: str = "",
    ) -> List[BaseMessage]:
        context = "\n\n".join(d.page_content for d in docs) if docs else "(không có chunks)"

        parts: List[str] = [
            "## Video context policy:\n"
            "Các đoạn trích bên dưới là nguồn chính. Mốc thời gian trong ngoặc vuông là vị trí trong video."
        ]
        if video_summary.strip():
            parts.append(f"## Tóm tắt video:\n{video_summary.strip()[:4000]}")
        parts.append(f"## Đoạn trích video (chunks):\n{context}")
        if web_results:
            parts.append(f"## Kết quả web search:\n{format_for_llm(web_results)}")
        history_text = self._format_history(history)
        if history_text:
            parts.append(
                "## Lịch sử hội thoại gần đây:\n"
                "Chỉ dùng phần này để hiểu các câu hỏi follow-up như 'nó', 'phần đó', 'ý trên'.\n"
                f"{history_text}"
            )
        parts.append(f"## Câu hỏi: {question}")

        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(parts)),
        ]

    def _format_history(self, history: Optional[Sequence[BaseMessage]]) -> str:
        if not history:
            return ""
        lines: list[str] = []
        for message in list(history)[-self.history_window:]:
            content = message.content if isinstance(message.content, str) else str(message.content)
            content = re.sub(r"\s+", " ", content).strip()
            if not content:
                continue
            if isinstance(message, HumanMessage):
                role = "User"
                content = content[:800]
            elif isinstance(message, AIMessage):
                role = "Assistant"
                content = content[:1200]
            else:
                role = "Message"
                content = content[:800]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _extract_text(response) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)

    @staticmethod
    def _clean_answer(text: str) -> str:
        """Light cleanup because the current UI renders plain text, not Markdown."""
        text = str(text or "").strip()
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"__(.*?)__", r"\1", text)
        text = re.sub(r"\\\((.*?)\\\)", r"\1", text)
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def _extract_cache_usage(response) -> dict:
        usage = getattr(response, "usage_metadata", None) or {}
        response_metadata = getattr(response, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
        details = token_usage.get("prompt_tokens_details") or {}
        return {
            "prompt_cache_hit_tokens": (
                usage.get("input_token_details", {}).get("cache_read")
                or details.get("cached_tokens")
                or token_usage.get("prompt_cache_hit_tokens")
            ),
            "prompt_cache_miss_tokens": token_usage.get("prompt_cache_miss_tokens"),
        }
