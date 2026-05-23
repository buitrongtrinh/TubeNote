"""QA Pipeline — RAG + sliding window memory + optional web search hybrid."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain_community.retrievers import BM25Retriever

from ..config import CFG
from ..processors.rag.store import get_all_docs, get_vector_store
from ..processors.web.searxng import SearXNGError, WebResult, format_for_llm, web_search


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
    "Bạn là trợ lý Q&A trên video YouTube. Trả lời bằng tiếng Việt.\n\n"
    "NGUỒN THÔNG TIN (theo thứ tự ưu tiên):\n"
    "  1. **Chunks từ video** — nguồn chính cho nội dung video.\n"
    "  2. **Bản tóm tắt video** (AIMessage đầu trong lịch sử chat) — cho câu hỏi tổng quan.\n"
    "  3. **Lịch sử chat** — để hiểu follow-up.\n"
    "  4. **Kết quả web search** (đánh dấu `[WEB-N]`, kèm Domain + URL) — info ngoài video / mới hơn.\n\n"
    "QUY TẮC TRÍCH DẪN (BẮT BUỘC):\n"
    "- Claim từ video → ghi `(theo video)` sau câu.\n"
    "- Claim từ web → ghi `(theo <domain>)` + link Markdown `[<domain>](<URL>)`.\n"
    "- Mỗi đoạn dùng web phải ghi RÕ URL đầy đủ — KHÔNG bịa.\n"
    "- KHÔNG trộn lẫn nguồn mà không ghi rõ.\n\n"
    "QUY TẮC NỘI DUNG:\n"
    "- Câu hỏi cụ thể về video → ưu tiên chunks.\n"
    "- Câu hỏi tổng quan → ưu tiên summary.\n"
    "- Câu hỏi cập nhật / so sánh / ngoài video → dùng [WEB-N].\n"
    "- Cả video lẫn web đều thiếu → 'Tôi không có thông tin về...'. KHÔNG bịa.\n"
    "- Trả lời súc tích, đi thẳng vấn đề."
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
        on_progress: Optional[ProgressCallback] = None,
        web_mode: WebMode = "off",
    ) -> QAResult:
        progress = on_progress or _noop

        # ── RAG retrieval (with scores cho heuristic auto-mode) ──────────────
        progress("🔍 Đang search chunks liên quan trong vector store…")
        docs, top_score = self._retrieve_with_scores(video_id, question)
        progress(f"📚 Lấy được {len(docs)} chunks (top score: {top_score:.3f})")

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
            progress(f"🌐 Đang search web (SearXNG) — lý do: {reason}…")
            try:
                web_results = web_search(question)
                progress(f"✅ Web search OK ({len(web_results)} kết quả)")
            except SearXNGError as e:
                progress(f"⚠️ Web search fail: {e} — tiếp tục chỉ với RAG")

        # ── Build messages ───────────────────────────────────────────────────
        messages = self._build_messages(question, docs, web_results, history)

        progress("🤖 Đang gọi LLM trả lời…")
        response = self.llm.invoke(messages)
        answer = self._extract_text(response)

        return QAResult(
            url=url,
            video_id=video_id,
            chunks_used=docs,
            answer=answer,
            web_results=web_results,
            top_rag_score=top_score,
            web_triggered_by=web_triggered,
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
    ) -> List[BaseMessage]:
        context = "\n\n".join(d.page_content for d in docs) if docs else "(không có chunks)"

        # Summary từ AIMessage đầu của history
        summary_text = ""
        if history:
            for m in history:
                if isinstance(m, AIMessage):
                    summary_text = m.content if isinstance(m.content, str) else str(m.content)
                    break

        parts: List[str] = []
        if summary_text:
            parts.append(f"## Bản tóm tắt video:\n{summary_text}")
        parts.append(f"## Đoạn trích video (chunks):\n{context}")
        if web_results:
            parts.append(f"## Kết quả web search:\n{format_for_llm(web_results)}")
        parts.append(f"## Câu hỏi: {question}")

        messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        if history:
            history_list = list(history)
            # Bỏ AIMessage đầu (đã extract làm summary)
            if history_list and isinstance(history_list[0], AIMessage):
                history_list = history_list[1:]
            messages.extend(history_list[-self.history_window:])
        messages.append(HumanMessage(content="\n\n".join(parts)))
        return messages

    @staticmethod
    def _extract_text(response) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)
