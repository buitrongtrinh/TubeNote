"""Summarizer agent: YouTube link -> tóm tắt. LLM + MCP tools đều pluggable."""
from __future__ import annotations

from typing import Any, Callable, Optional

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

EventCallback = Callable[[dict], None]

from ..config import CFG
from ..llm import make_llm
from ..mcp.client import MCPClient
from ..prompts import SUMMARIZER_SYSTEM


_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
)


def _contains_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for lo, hi in _CJK_RANGES:
            if lo <= cp <= hi:
                return True
    return False


async def _repair_to_vietnamese(llm: BaseChatModel, text: str) -> str:
    """Dùng chính LLM để dịch đoạn có lẫn chữ Hán về tiếng Việt thuần."""
    prompt = (
        "Viết lại văn bản sau HOÀN TOÀN bằng tiếng Việt. "
        "Dịch mọi chữ Hán / tiếng Trung sang tiếng Việt. "
        "GIỮ NGUYÊN format Markdown (heading **, bullet -). "
        "Chỉ xuất văn bản đã dịch, không giải thích, không thêm preamble.\n\n"
        f"{text}"
    )
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    return _extract_text(resp.content)


def _extract_text(content: Any) -> str:
    """Chuẩn hoá content từ BaseMessage.

    Ollama/OpenAI/Anthropic thường trả str.
    Gemini trả list content-block: [{'type': 'text', 'text': '...', 'extras': {...}}].
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _emit(callback: Optional[EventCallback], event: dict) -> None:
    if callback is not None:
        try:
            callback(event)
        except Exception:
            pass  # callback lỗi không được ngắt agent


async def summarize(
    url: str,
    instruction: str = "",
    *,
    llm: Optional[BaseChatModel] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    on_event: Optional[EventCallback] = None,
) -> str:
    """Run the summarizer agent.

    Precedence: explicit `llm` arg > (provider/model/api_key overrides) > config.yaml.

    on_event(event_dict) được gọi mỗi khi có sự kiện. Event types:
      - {"type": "agent_start"}
      - {"type": "tool_call", "name": str, "args": dict}
      - {"type": "tool_result", "name": str, "content": str}
      - {"type": "llm_message", "content": str}            # AI message giữa các vòng
      - {"type": "repair", "reason": "cjk"}                # bắt đầu pass dịch CJK
      - {"type": "done"}
    """
    if llm is None:
        llm = make_llm(provider=provider, model=model, api_key=api_key)

    async with MCPClient.default() as client:
        tools = await client.langchain_tools()
        agent = create_agent(llm, tools, system_prompt=SUMMARIZER_SYSTEM)

        content = f"{url}\n{instruction}".strip() if instruction else url
        _emit(on_event, {"type": "agent_start"})

        last_ai: Optional[AIMessage] = None
        seen_calls: set = set()    # tool_call_id đã emit "tool_call"
        seen_results: set = set()  # tool_call_id đã emit "tool_result"

        async for chunk in agent.astream(
            {"messages": [HumanMessage(content=content)]},
            stream_mode="values",
            config={"recursion_limit": CFG.agent.max_iterations * 2},
        ):
            messages = chunk.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage):
                    for tc in msg.tool_calls or []:
                        tc_id = tc.get("id") or f"{tc['name']}-{len(seen_calls)}"
                        if tc_id in seen_calls:
                            continue
                        seen_calls.add(tc_id)
                        _emit(on_event, {
                            "type": "tool_call",
                            "name": tc["name"],
                            "args": tc.get("args", {}),
                        })
                    last_ai = msg
                elif isinstance(msg, ToolMessage):
                    if msg.tool_call_id in seen_results:
                        continue
                    seen_results.add(msg.tool_call_id)
                    _emit(on_event, {
                        "type": "tool_result",
                        "name": msg.name or "<tool>",
                        "content": _extract_text(msg.content),
                    })

        if last_ai is None:
            raise RuntimeError("Agent không trả về AIMessage nào.")

        summary = _extract_text(last_ai.content)
        if _contains_cjk(summary):
            _emit(on_event, {"type": "repair", "reason": "cjk"})
            summary = await _repair_to_vietnamese(llm, summary)

        _emit(on_event, {"type": "done"})
        return summary
