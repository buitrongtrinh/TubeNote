"""Streamlit UI. Run: streamlit run apps/streamlit_app.py"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from tubenote.config import CFG, PROJECT_ROOT
from tubenote.llm.providers import make_llm
from tubenote.pipeline.summarizer import SummarizerPipeline
from tubenote.pipeline.qa import QAPipeline
from tubenote.processors.youtube.cookies import list_cookie_files
from tubenote import config as _cfg


def _to_history(messages: list[dict]) -> list[BaseMessage]:
    """Convert st.session_state['messages'] → BaseMessage list cho LLM."""
    out: list[BaseMessage] = []
    for m in messages:
        if m["role"] == "user":
            out.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            out.append(AIMessage(content=m["content"]))
    return out


# ── Error helpers ────────────────────────────────────────────────────────────

def _collect_error_text(e: BaseException, _seen: set | None = None) -> str:
    if _seen is None:
        _seen = set()
    if id(e) in _seen:
        return ""
    _seen.add(id(e))
    parts = [f"{type(e).__name__}: {e}"]
    for sub in getattr(e, "exceptions", None) or []:
        parts.append(_collect_error_text(sub, _seen))
    if getattr(e, "__cause__", None):
        parts.append(_collect_error_text(e.__cause__, _seen))
    ctx = getattr(e, "__context__", None)
    if ctx and ctx is not getattr(e, "__cause__", None):
        parts.append(_collect_error_text(ctx, _seen))
    return " | ".join(p for p in parts if p)


def _friendly_error(e: Exception) -> tuple[str, str]:
    s = _collect_error_text(e).lower()
    if "api key not valid" in s or "api_key_invalid" in s or "invalid api key" in s:
        return ("API key không hợp lệ", "Kiểm tra lại key. Lấy mới tại https://aistudio.google.com/apikey")
    if "permission_denied" in s or " 401" in s or " 403" in s:
        return ("API key bị từ chối", "Key có thể đã bị revoke hoặc thiếu quyền truy cập model.")
    if "resource_exhausted" in s or " 429" in s or "quota" in s or "rate limit" in s:
        return ("Hết quota / rate limit", "Đợi reset (PT timezone) hoặc đổi sang `gemini-2.5-flash-lite`, hoặc bật billing.")
    if "thiếu" in s and "api" in s:
        return ("Thiếu API key", "Set vào `.env` hoặc nhập ở sidebar bên trái.")
    if "requestblocked" in s.replace(" ", "") or "blocking requests from your ip" in s:
        return ("YouTube đang chặn IP", "Upload `cookies.txt` mới qua sidebar bên trái, hoặc chờ 30–60 phút.")
    if "sign in to confirm" in s or "not a bot" in s:
        return ("YouTube bắt xác thực bot", "Upload `cookies.txt` mới qua sidebar (cookies cũ có thể đã expire).")
    if "connectionrefusederror" in s or "winerror 10061" in s or "connection refused" in s:
        return ("Ollama chưa chạy", "Mở Ollama (Start Menu → Ollama) hoặc chạy `ollama serve` — daemon phải listen ở `localhost:11434`.")
    if "subtitles are disabled" in s or "no transcripts" in s or "could not retrieve a transcript" in s:
        return ("Video không có phụ đề", "Chủ kênh đã tắt phụ đề. Roadmap: hỗ trợ Whisper STT.")
    if "không lấy được video id" in s:
        return ("URL YouTube không hợp lệ", "Kiểm tra lại link, dạng https://youtu.be/... hoặc https://www.youtube.com/watch?v=...")
    if "timeout" in s or "timed out" in s:
        return ("Timeout kết nối", "Mạng chậm hoặc Gemini đang quá tải. Thử lại sau 30s.")
    full = _collect_error_text(e)
    return ("Lỗi không xác định", full[:300] + "…" if len(full) > 300 else full)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title=CFG.ui.title, page_icon=CFG.ui.page_icon, layout="centered")
st.title(f"{CFG.ui.page_icon} {CFG.ui.title}")
st.caption(CFG.ui.caption)


# ── CSS: tool panel styling ───────────────────────────────────────────────────

st.markdown("""
<style>
/* Tool panel card */
.tool-panel {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 16px 0 8px 0;
}
.tool-panel-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(165, 180, 252, 0.7);
    margin-bottom: 14px;
}

/* Spinner animation for processing state */
@keyframes spin {
    to { transform: rotate(360deg); }
}
.tool-spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid rgba(99,102,241,0.3);
    border-top-color: #818cf8;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: middle;
    margin-right: 6px;
}

/* Progress bar pulse */
@keyframes progress-pulse {
    0%   { width: 15%; }
    50%  { width: 85%; }
    100% { width: 15%; }
}
.fake-progress-bar {
    height: 3px;
    border-radius: 99px;
    background: linear-gradient(90deg, #6366f1, #a78bfa, #6366f1);
    animation: progress-pulse 2.5s ease-in-out infinite;
    margin-top: 8px;
}

/* Video result embed */
.video-result-wrap {
    border-radius: 12px;
    overflow: hidden;
    margin-top: 12px;
    border: 1px solid rgba(99, 102, 241, 0.25);
    background: #000;
}
</style>
""", unsafe_allow_html=True)


# ── Auto-start Ollama ─────────────────────────────────────────────────────────

@st.cache_resource
def _bootstrap_ollama() -> bool:
    from tubenote.ollama_runtime import ensure_running, is_running
    if is_running(CFG.embedding.base_url):
        return True
    return ensure_running(CFG.embedding.base_url, wait_seconds=15)


if not _bootstrap_ollama():
    st.error(
        "❌ **Ollama daemon không chạy được.**\n\n"
        "Project cần Ollama cho embedding (`qwen3-embedding:0.6b`). Cách fix:\n"
        "- Cài Ollama: https://ollama.com/download\n"
        "- Hoặc tự start: chạy `ollama serve` trong PowerShell"
    )
    st.stop()


# ── LLM cache ─────────────────────────────────────────────────────────────────

@st.cache_resource
def get_pipeline(provider: str, model: str, api_key: str | None) -> SummarizerPipeline:
    llm = make_llm(provider=provider, model=model, api_key=api_key)
    return SummarizerPipeline(llm)


@st.cache_resource
def get_qa_pipeline(provider: str, model: str, api_key: str | None) -> QAPipeline:
    llm = make_llm(provider=provider, model=model, api_key=api_key)
    return QAPipeline(llm)


# ── Sidebar ───────────────────────────────────────────────────────────────────

PROVIDERS = ["local", "google", "openai", "anthropic"]

with st.sidebar:
    st.subheader("LLM Provider")
    provider = st.selectbox(
        "Provider",
        PROVIDERS,
        index=PROVIDERS.index(CFG.llm.provider) if CFG.llm.provider in PROVIDERS else 0,
    )

    provider_models = CFG.llm.provider_models(provider)
    if not provider_models:
        st.error(f"Provider `{provider}` không có model nào trong config.yaml")
        st.stop()

    model = st.selectbox(
        "Model",
        provider_models,
        index=0,
        key=f"select_model_{provider}",
    )

    if provider == "local":
        ollama_url = os.getenv("OLLAMA_BASE_URL") or CFG.llm.provider_opts("local").get("base_url", "http://localhost:11434")
        st.info(f"🖥️ Ollama @ `{ollama_url}` — không cần API key.")
        api_key = None
    else:
        env_var_name = f"{provider.upper()}_API_KEY"
        env_api_key  = os.getenv(env_var_name)

        if env_api_key:
            masked  = f"{env_api_key[:6]}…{env_api_key[-4:]}" if len(env_api_key) > 12 else "••••"
            st.success(f"✅ Đã có {env_var_name} trong .env\n\n`{masked}`")
            api_key = None
        else:
            api_key = st.text_input(
                f"{provider.upper()} API key",
                type="password",
                help="Lưu trong session, không ghi đĩa.",
            )
            if not api_key:
                st.caption(f"💡 Hoặc set `{env_var_name}` trong `.env`")

    st.divider()
    st.subheader("Session YouTube")

    _flash = st.session_state.pop("_cookie_flash", None)
    if _flash:
        kind, text = _flash
        {"success": st.success, "info": st.info, "error": st.error}[kind](text)

    env_path     = os.getenv("YT_COOKIES_PATH")
    env_dir      = os.getenv("YT_COOKIES_DIR")
    env_resolved: Path | None = None
    env_var_used: str | None  = None

    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            env_resolved, env_var_used = p, "YT_COOKIES_PATH"
    elif env_dir:
        p = Path(env_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_dir() and any(p.glob("*.txt")):
            env_resolved, env_var_used = p, "YT_COOKIES_DIR"

    cookies = list_cookie_files()

    if env_resolved is not None:
        st.success(f"✅ **Có session từ `.env`**\n\n`{env_var_used}={env_resolved}`")
        if len(cookies) > 1:
            st.caption(f"Rotation đang dùng {len(cookies)} file")
    else:
        if env_path or env_dir:
            st.warning(f"⚠️ Path trong `.env` không hợp lệ: `{env_path or env_dir}`")
        elif cookies:
            st.success(f"✅ **Đã có session** ({len(cookies)} file)")
        else:
            st.warning("⚠️ **Chưa có session** — upload bên dưới hoặc set `YT_COOKIES_PATH` trong `.env`.")

        _uploader_n = st.session_state.get("_cookie_uploader_n", 0)
        uploaded = st.file_uploader(
            "📤 Upload cookies.txt",
            type=["txt"],
            help="Export từ Chrome extension 'Get cookies.txt LOCALLY'.",
            key=f"cookie_uploader_{_uploader_n}",
        )

        with st.expander("📖 Cách lấy file cookies.txt"):
            st.markdown("""
**Bước 1.** Cài Chrome extension **[Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)**

**Bước 2.** Mở `https://www.youtube.com` và **đăng nhập** account Google.

**Bước 3.** Click icon extension → **Export** → **Netscape** format → lưu file `.txt`.

**Bước 4.** Upload file vừa lưu lên ô trên ⬆️

---
⚠️ Cookies có TTL **2–8 tuần**. File chứa session login — **không share**.
            """)

        if uploaded is not None:
            import hashlib
            from time import time as _now

            content      = uploaded.getvalue()
            content_hash = hashlib.sha256(content).hexdigest()[:12]
            cookies_dir  = CFG.cookies.dir or (PROJECT_ROOT / "cookies")
            cookies_dir.mkdir(parents=True, exist_ok=True)

            existing_hashes = {
                hashlib.sha256(p.read_bytes()).hexdigest()[:12]: p.name
                for p in cookies_dir.glob("*.txt")
            }
            if content_hash in existing_hashes:
                st.session_state["_cookie_flash"] = (
                    "info",
                    f"ℹ️ File trùng nội dung với `{existing_hashes[content_hash]}` — bỏ qua.",
                )
            else:
                primary_path = cookies_dir / "primary.txt"
                if not primary_path.exists():
                    target = primary_path
                    msg    = "✅ **Đã có session** — saved as `primary.txt`"
                else:
                    target = cookies_dir / f"upload_{int(_now())}.txt"
                    msg    = f"✅ Đã thêm backup `{target.name}` — rotation +1"
                target.write_bytes(content)
                _cfg.CFG.cookies.dir = cookies_dir
                st.session_state["_cookie_flash"] = ("success", msg)

            st.session_state["_cookie_uploader_n"] = _uploader_n + 1
            st.rerun()

    st.divider()
    st.subheader("🌐 Web search")
    web_mode_map = {"Tắt": "off", "Tự động": "auto", "Luôn bật": "always"}
    web_mode_label = st.radio(
        "Mode",
        list(web_mode_map.keys()),
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        help=(
            "Tắt: chỉ dùng video.\n"
            "Tự động: web kích hoạt khi RAG yếu (score thấp).\n"
            "Luôn bật: mỗi câu hỏi đều search web."
        ),
    )
    web_mode = web_mode_map[web_mode_label]
    if web_mode != "off":
        @st.cache_data(ttl=15, show_spinner=False)
        def _searxng_ok() -> bool:
            from tubenote.processors.web.searxng import is_reachable
            return is_reachable()

        if _searxng_ok():
            st.caption(f"✅ SearXNG OK ({CFG.web_search.base_url})")
        else:
            st.warning(
                f"⚠️ SearXNG chưa chạy. Mở terminal ở project root:\n\n"
                f"```\ndocker compose up -d\n```"
            )


# ── Session state defaults ────────────────────────────────────────────────────

st.session_state.setdefault("messages", [])
st.session_state.setdefault("active_url", None)
st.session_state.setdefault("active_video_id", None)
# Tool panel state
st.session_state.setdefault("tool_vietsub_state", "idle")   # idle | running | done | error
st.session_state.setdefault("tool_dubbing_state", "idle")   # idle | running | done | error
st.session_state.setdefault("tool_vietsub_output", None)    # path or URL to output video
st.session_state.setdefault("tool_dubbing_output", None)


def _reset_chat():
    st.session_state["messages"] = []
    st.session_state["active_url"] = None
    st.session_state["active_video_id"] = None
    st.session_state["tool_vietsub_state"] = "idle"
    st.session_state["tool_dubbing_state"] = "idle"
    st.session_state["tool_vietsub_output"] = None
    st.session_state["tool_dubbing_output"] = None


# ── Tool runner stubs (replace with real pipeline calls) ─────────────────────

def _run_vietsub(url: str, video_id: str) -> str | None:
    """Stub: thay bằng pipeline thực tế của bạn."""
    time.sleep(4)
    return "TTS_model_demo/output_dubbed8.mp4"


def _run_dubbing(url: str, video_id: str) -> str | None:
    """Stub: thay bằng pipeline thực tế của bạn."""
    time.sleep(5)
    return None


# ── Tool panel renderer ───────────────────────────────────────────────────────

def _render_tool_panel():
    """Hiển thị panel công cụ xử lý video sau khi summary xong."""
    url      = st.session_state["active_url"]
    video_id = st.session_state["active_video_id"]

    vs_state  = st.session_state["tool_vietsub_state"]
    dub_state = st.session_state["tool_dubbing_state"]

    st.markdown(
        '<div class="tool-panel">'
        '<div class="tool-panel-title">⚡ Công cụ xử lý video</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_vs, col_dub = st.columns(2)

    # ── Vietsub button ────────────────────────────────────────────────────────
    with col_vs:
        if vs_state == "idle":
            if st.button(
                "🎬 Vietsub video",
                key="btn_vietsub",
                use_container_width=True,
                help="Tạo phụ đề tiếng Việt và ghép vào video",
            ):
                st.session_state["tool_vietsub_state"] = "running"
                st.rerun()

        elif vs_state == "running":
            st.markdown(
                '<div style="text-align:center;padding:10px 0;">'
                '<span class="tool-spinner"></span>'
                '<span style="color:#a5b4fc;font-size:0.85rem;">Đang tạo vietsub…</span>'
                '<div class="fake-progress-bar"></div>'
                '</div>',
                unsafe_allow_html=True,
            )
            with st.spinner(""):
                try:
                    output = _run_vietsub(url, video_id)
                    st.session_state["tool_vietsub_output"] = output
                    st.session_state["tool_vietsub_state"] = "done"
                except Exception as e:
                    st.session_state["tool_vietsub_state"] = "error"
                    st.session_state["tool_vietsub_output"] = str(e)
            st.rerun()

        elif vs_state == "done":
            output = st.session_state["tool_vietsub_output"]
            st.success("✅ Vietsub hoàn tất!")
            if output and Path(output).exists():
                with open(output, "rb") as f:
                    st.download_button(
                        "⬇️ Tải video vietsub",
                        data=f,
                        file_name=f"{video_id}_vietsub.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                st.video(output)
            else:
                st.info("Pipeline vietsub chưa được cấu hình — kết nối `_run_vietsub()` với pipeline thực tế.")

        elif vs_state == "error":
            st.error("❌ Vietsub thất bại")
            st.caption(str(st.session_state.get("tool_vietsub_output", "")))
            if st.button("🔄 Thử lại vietsub", key="retry_vietsub", use_container_width=True):
                st.session_state["tool_vietsub_state"] = "idle"
                st.session_state["tool_vietsub_output"] = None
                st.rerun()

    # ── Dubbing button ────────────────────────────────────────────────────────
    with col_dub:
        if dub_state == "idle":
            if st.button(
                "🔊 Lồng tiếng video",
                key="btn_dubbing",
                use_container_width=True,
                help="Dịch và lồng tiếng Việt vào toàn bộ video",
            ):
                st.session_state["tool_dubbing_state"] = "running"
                st.rerun()

        elif dub_state == "running":
            st.markdown(
                '<div style="text-align:center;padding:10px 0;">'
                '<span class="tool-spinner"></span>'
                '<span style="color:#a5b4fc;font-size:0.85rem;">Đang lồng tiếng…</span>'
                '<div class="fake-progress-bar"></div>'
                '</div>',
                unsafe_allow_html=True,
            )
            with st.spinner(""):
                try:
                    output = _run_dubbing(url, video_id)
                    st.session_state["tool_dubbing_output"] = output
                    st.session_state["tool_dubbing_state"] = "done"
                except Exception as e:
                    st.session_state["tool_dubbing_state"] = "error"
                    st.session_state["tool_dubbing_output"] = str(e)
            st.rerun()

        elif dub_state == "done":
            output = st.session_state["tool_dubbing_output"]
            st.success("✅ Lồng tiếng hoàn tất!")
            if output and Path(output).exists():
                with open(output, "rb") as f:
                    st.download_button(
                        "⬇️ Tải video lồng tiếng",
                        data=f,
                        file_name=f"{video_id}_dubbed.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                st.video(output)
            else:
                st.info("Pipeline lồng tiếng chưa được cấu hình — kết nối `_run_dubbing()` với pipeline thực tế.")

        elif dub_state == "error":
            st.error("❌ Lồng tiếng thất bại")
            st.caption(str(st.session_state.get("tool_dubbing_output", "")))
            if st.button("🔄 Thử lại lồng tiếng", key="retry_dubbing", use_container_width=True):
                st.session_state["tool_dubbing_state"] = "idle"
                st.session_state["tool_dubbing_output"] = None
                st.rerun()


# ── URL input + actions ──────────────────────────────────────────────────────
# Dùng st.form để URL CHỈ submit khi user Enter / click ▶ — tránh trigger
# summary khi user blur text_input (vd click sidebar đổi provider).

with st.form("url_form", clear_on_submit=False, border=False):
    url_col, btn_col = st.columns([5, 1])
    with url_col:
        new_url_raw = st.text_input(
            "Link YouTube",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
            key="chat_url_input",
        )
    with btn_col:
        submitted = st.form_submit_button("▶", use_container_width=True, help="Tóm tắt video")

new_url = (new_url_raw or "").strip()  # trim whitespace/newline do paste nhầm

if st.button("🔄 Reset chat", help="Xoá chat, đổi video khác"):
    _reset_chat()
    st.rerun()

# ── Summarize khi user submit URL mới ────────────────────────────────────────

if submitted and new_url and new_url != st.session_state["active_url"]:
    # Reset toàn bộ state — CHƯA commit active_url, chỉ set sau khi pipeline thành công
    st.session_state["messages"] = []
    st.session_state["active_video_id"] = None
    st.session_state["tool_vietsub_state"] = "idle"
    st.session_state["tool_dubbing_state"] = "idle"
    st.session_state["tool_vietsub_output"] = None
    st.session_state["tool_dubbing_output"] = None

    with st.status("Đang xử lý video…", expanded=True) as status:
        progress = lambda msg: status.write(msg)
        try:
            sum_pipe   = get_pipeline(provider, model, api_key or None)
            sum_result = sum_pipe.run(new_url, on_progress=progress)

            from tubenote.processors.rag.pipeline import ingest as _ingest
            info = _ingest(new_url, on_progress=progress)

            # Chỉ commit khi CẢ HAI bước thành công
            st.session_state["active_url"]      = new_url
            st.session_state["active_video_id"] = info["video_id"]

            status.update(label="✅ Sẵn sàng chat", state="complete", expanded=False)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": sum_result.summary,
                "kind": "summary",
            })
        except Exception as e:
            title, hint = _friendly_error(e)
            status.update(label=f"❌ {title}", state="error")
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"❌ **{title}**\n\n{hint}",
            })


# ── Render helpers ────────────────────────────────────────────────────────────

def _render_assistant_message(msg: dict) -> None:
    """Render assistant message + badges + chunks expander + web sources."""
    st.markdown(msg["content"])

    badges = []
    if msg.get("top_rag_score") is not None:
        badges.append(f"RAG top score: `{msg['top_rag_score']:.2f}`")
    if msg.get("web_triggered_by"):
        n_web = len(msg.get("web_results", []))
        badges.append(f"🌐 Web search: **{msg['web_triggered_by']}** ({n_web} kết quả)")
    if badges:
        st.caption(" · ".join(badges))

    chunks = msg.get("chunks") or []
    if chunks:
        with st.expander(f"📚 {len(chunks)} chunks từ video"):
            for i, content in enumerate(chunks, 1):
                st.caption(f"**Chunk {i}**: {content}")

    web_results = msg.get("web_results") or []
    if web_results:
        with st.expander(f"🌐 {len(web_results)} nguồn web"):
            for i, r in enumerate(web_results, 1):
                st.markdown(f"**[WEB-{i}]** [{r['title']}]({r['url']})  \n`{r['domain']}` — {r['snippet']}")


# ── Render chat history ───────────────────────────────────────────────────────

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_assistant_message(msg)
        else:
            st.markdown(msg["content"])
    # Tool panel hiện ngay sau summary, không xuất hiện sau mỗi Q&A
    if msg.get("kind") == "summary" and st.session_state["active_video_id"]:
        _render_tool_panel()


# ── Chat input ────────────────────────────────────────────────────────────────

if st.session_state["active_video_id"]:
    user_input = st.chat_input("Hỏi về video…")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            qa_result = None
            with st.spinner("🤖 Đang tìm câu trả lời…"):
                try:
                    qa_pipe   = get_qa_pipeline(provider, model, api_key or None)
                    history   = _to_history(st.session_state["messages"][:-1])
                    qa_result = qa_pipe.run(
                        video_id=st.session_state["active_video_id"],
                        question=user_input,
                        url=st.session_state["active_url"],
                        history=history,
                        web_mode=web_mode,
                    )
                    answer = qa_result.answer
                except Exception as e:
                    title, hint = _friendly_error(e)
                    answer = f"❌ **{title}**\n\n{hint}"

            assistant_msg: dict = {"role": "assistant", "content": answer}
            if qa_result is not None:
                assistant_msg["chunks"] = [d.page_content for d in qa_result.chunks_used]
                assistant_msg["top_rag_score"] = qa_result.top_rag_score
                assistant_msg["web_triggered_by"] = qa_result.web_triggered_by
                assistant_msg["web_results"] = [
                    {"title": r.title, "url": r.url, "domain": r.domain, "snippet": r.snippet}
                    for r in qa_result.web_results
                ]
            _render_assistant_message(assistant_msg)

        st.session_state["messages"].append(assistant_msg)

elif not new_url:
    st.info("👆 Paste link YouTube ở trên để bắt đầu — tóm tắt sẽ tự động hiện ra, sau đó bạn có thể chat hỏi tiếp.")
