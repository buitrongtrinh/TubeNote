from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from tubenote.config import CFG
from tubenote.processors.youtube.transcript import fetch_transcript
from tubenote.processors.youtube.transcript_yt_dlp import fetch_metadata
from tubenote.processors.youtube.utils import load_json

ProgressCallback = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def _extract_text(content: Any) -> str:
    """Gemini trả content dạng list-of-dict → join lại thành string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if t := item.get("text"):
                    parts.append(t)
            else:
                if t := getattr(item, "text", None):
                    parts.append(t)
        return "\n".join(parts)
    return str(content)


SUMMARIZE_PROMPT = """\
FORMAT OUTPUT:
📺 **{title}**

👤 Kênh: [{channel}]({channel_url})

![thumbnail]({thumbnail})

---

**Chủ đề**: 1 câu mô tả nội dung chính.

**Ý chính**:
- bullet 1
- bullet 2
- bullet 3–5

**Chi tiết nổi bật**: con số / ví dụ / thuật ngữ quan trọng (1–3 dòng).

**Kết luận**: 1–2 câu.

GHI CHÚ:
- Luôn trả lời bằng Tiếng Việt, kể cả khi transcript gốc là tiếng Anh.
- Thay {{title}}, {{channel}}, {{channel_url}}, {{thumbnail}} bằng giá trị thật từ metadata.
- Transcript từ TTS có thể thiếu dấu câu / dấu tiếng Việt: hãy diễn giải linh hoạt, KHÔNG chép nguyên văn sai chính tả.
"""


@dataclass
class SummaryResult:
    url: str
    metadata: dict
    transcript: str
    summary: str


class SummarizerPipeline:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(self, url: str, on_progress: Optional[ProgressCallback] = None) -> SummaryResult:
        progress = on_progress or _noop
        languages = CFG.transcript.default_languages

        progress("📺 Đang lấy metadata video…")
        metadata_path = fetch_metadata(url=url)
        metadata = load_json(metadata_path)
        progress("📺 Lấy metadata video thành công!")


        transcript = self._fetch_transcript(url, languages, progress)

        progress("🤖 Đang gọi LLM tóm tắt…")
        summary = self._summarize(metadata, transcript)
        return SummaryResult(
            url=url,
            metadata=metadata,
            transcript=transcript,
            summary=summary,
        )

    def _fetch_transcript(
        self,
        url: str,
        languages: list[str],
        progress: ProgressCallback = _noop,
    ) -> str:
        trans = fetch_transcript(url, languages=languages, on_progress=progress)
        if trans is None:
            raise RuntimeError("Không lấy được transcript bằng cả yt-dlp lẫn Whisper.")
        progress(f"📝 Transcript OK ({len(trans.segments)} segments)")
        return trans.full_text

    def _summarize(self, metadata: dict, transcript: str) -> str:
        system = SystemMessage(content=SUMMARIZE_PROMPT)
        user = HumanMessage(content=(
            f"Metadata:\n{metadata}\n\n"
            f"Transcript:\n{transcript}"
        ))
        response = self.llm.invoke([system, user])
        return _extract_text(response.content)
