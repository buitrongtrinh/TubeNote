"""Streamlit UI. Run: streamlit run apps/streamlit_app.py"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Streamlit chỉ add folder chứa script (apps/) vào sys.path.
# Thêm project root để import được package `link2slide`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from link2slide.agents.summarizer import summarize
from link2slide.config import CFG
from link2slide.youtube.cookies import list_cookie_files
import os

def _collect_error_text(e: BaseException, _seen: set | None = None) -> str:
    """Walk qua ExceptionGroup + __cause__ + __context__ để lấy hết text."""
    if _seen is None:
        _seen = set()
    if id(e) in _seen:
        return ""
    _seen.add(id(e))

    parts = [f"{type(e).__name__}: {e}"]
    # ExceptionGroup (py3.11+ built-in, py3.10 qua package exceptiongroup)
    subs = getattr(e, "exceptions", None)
    if subs:
        for sub in subs:
            parts.append(_collect_error_text(sub, _seen))
    if getattr(e, "__cause__", None):
        parts.append(_collect_error_text(e.__cause__, _seen))
    ctx = getattr(e, "__context__", None)
    if ctx and ctx is not getattr(e, "__cause__", None):
        parts.append(_collect_error_text(ctx, _seen))
    return " | ".join(p for p in parts if p)


def _friendly_error(e: Exception) -> tuple[str, str]:
    """Phân loại exception → (tiêu đề ngắn, hướng dẫn xử lý). Fallback về message gốc."""
    s = _collect_error_text(e).lower()

    if "api key not valid" in s or "api_key_invalid" in s or "invalid api key" in s:
        return ("API key không hợp lệ", "Kiểm tra lại key. Lấy mới tại https://aistudio.google.com/apikey")
    if "permission_denied" in s or " 401" in s or " 403" in s:
        return ("API key bị từ chối", "Key có thể đã bị revoke hoặc thiếu quyền truy cập model.")
    if "resource_exhausted" in s or " 429" in s or "quota" in s or "rate limit" in s:
        return (
            "Hết quota / rate limit",
            "Đợi reset (PT timezone) hoặc đổi sang `gemini-2.5-flash-lite` (1000 RPD), hoặc bật billing.",
        )
    if "thiếu" in s and "api" in s:
        return ("Thiếu API key", f"Set vào `.env` hoặc nhập ở sidebar bên trái.")
    if "requestblocked" in s.replace(" ", "") or "blocking requests from your ip" in s:
        return (
            "YouTube đang chặn IP",
            "Upload `cookies.txt` mới qua **sidebar bên trái** (mục 📤 Upload cookies.txt), "
            "hoặc chờ 30–60 phút cho IP unblock.",
        )
    if "subtitles are disabled" in s or "no transcripts" in s or "could not retrieve a transcript" in s:
        return (
            "Video không có phụ đề",
            "Chủ kênh đã tắt phụ đề. Roadmap: hỗ trợ Whisper STT cho case này.",
        )
    if "không lấy được video id" in s:
        return ("URL YouTube không hợp lệ", "Kiểm tra lại link, đảm bảo dạng https://youtu.be/... hoặc https://www.youtube.com/watch?v=...")
    if "timeout" in s or "timed out" in s:
        return ("Timeout kết nối", "Mạng chậm hoặc Gemini đang quá tải. Thử lại sau 30s.")

    # Fallback — cắt gọn từ full chain (đã có sub-exception)
    full = _collect_error_text(e)
    if len(full) > 300:
        full = full[:300] + "…"
    return ("Lỗi không xác định", full)


st.set_page_config(page_title=CFG.ui.title, page_icon=CFG.ui.page_icon, layout="centered")

st.title(f"{CFG.ui.page_icon} {CFG.ui.title}")
st.caption(CFG.ui.caption)

# ---------- Sidebar: provider picker ----------
PROVIDERS = ["google", "openai", "anthropic"]
DEFAULT_MODELS = {
    "google": CFG.llm.provider_opts("google").get("model", "gemini-2.5-flash"),
    "openai": CFG.llm.provider_opts("openai").get("model", "gpt-4o-mini"),
    "anthropic": CFG.llm.provider_opts("anthropic").get("model", "claude-haiku-4-5"),
}

with st.sidebar:
    st.subheader("LLM Provider")
    provider = st.selectbox(
        "Provider",
        PROVIDERS,
        index=PROVIDERS.index(CFG.llm.provider) if CFG.llm.provider in PROVIDERS else 0,
    )
    model = st.text_input("Model", value=DEFAULT_MODELS.get(provider, ""))

    env_var_name = f"{provider.upper()}_API_KEY"
    env_api_key = os.getenv(env_var_name)

    if env_api_key:
        masked = f"{env_api_key[:6]}…{env_api_key[-4:]}" if len(env_api_key) > 12 else "••••"
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

    # Flash message từ lần upload trước
    _flash = st.session_state.pop("_cookie_flash", None)
    if _flash:
        kind, text = _flash
        {"success": st.success, "info": st.info, "error": st.error}[kind](text)

    # ----- Detect session source: ENV trước, file system sau -----
    from link2slide.config import PROJECT_ROOT as _ROOT

    env_path = os.getenv("YT_COOKIES_PATH")
    env_dir = os.getenv("YT_COOKIES_DIR")
    env_resolved: Path | None = None
    env_var_used: str | None = None

    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = _ROOT / p
        if p.is_file():
            env_resolved = p
            env_var_used = "YT_COOKIES_PATH"
    elif env_dir:
        p = Path(env_dir)
        if not p.is_absolute():
            p = _ROOT / p
        if p.is_dir() and any(p.glob("*.txt")):
            env_resolved = p
            env_var_used = "YT_COOKIES_DIR"

    cookies = list_cookie_files()

    if env_resolved is not None:
        # Env hợp lệ → KHÔNG hiện upload, chỉ show status
        st.success(f"✅ **Có session từ `.env`**\n\n`{env_var_used}={env_resolved}`")
        if len(cookies) > 1:
            st.caption(f"Rotation đang dùng {len(cookies)} file")
    else:
        # Env không có hoặc path không tồn tại → cho upload
        if env_path or env_dir:
            invalid = env_path or env_dir
            st.warning(f"⚠️ Path trong `.env` không hợp lệ: `{invalid}`")
        elif cookies:
            # Có file local nhưng không qua env (auto-discovery)
            st.success(f"✅ **Đã có session** ({len(cookies)} file)")
        else:
            st.warning("⚠️ **Chưa có session** — upload bên dưới hoặc set `YT_COOKIES_PATH` trong `.env`.")

        # Counter để reset widget sau mỗi upload thành công
        _uploader_n = st.session_state.get("_cookie_uploader_n", 0)
        uploaded = st.file_uploader(
            "📤 Upload cookies.txt",
            type=["txt"],
            help="Export từ Chrome extension 'Get cookies.txt LOCALLY'.",
            key=f"cookie_uploader_{_uploader_n}",
        )

        with st.expander("📖 Cách lấy file cookies.txt"):
            st.markdown(
                """
**Bước 1.** Cài Chrome extension **[Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)**
- Mã nguồn mở, chạy offline.

**Bước 2.** Mở `https://www.youtube.com` và **đăng nhập** account Google.

**Bước 3.** Click icon extension → **Export** → **Netscape** format → lưu file `.txt`.

**Bước 4.** Upload file vừa lưu lên ô trên ⬆️

---

⚠️ **Lưu ý:**
- Cookies có TTL **2–8 tuần** → khi tool fail, export & upload lại.
- Upload nhiều file → rotation tự động khi 1 account bị limit.
- File chứa session login — **không share**.
                """
            )
        # Xử lý upload (chỉ chạy nếu khối else trên hiển thị uploader)
        if uploaded is not None:
            import hashlib
            from time import time as _now
            from link2slide.config import PROJECT_ROOT as _UPLOAD_ROOT
            from link2slide import config as _cfg

            content = uploaded.getvalue()
            content_hash = hashlib.sha256(content).hexdigest()[:12]

            cookies_dir = (CFG.cookies.dir or (_UPLOAD_ROOT / "cookies"))
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
                st.session_state["_cookie_uploader_n"] = _uploader_n + 1
                st.rerun()
            else:
                primary_path = cookies_dir / "primary.txt"
                if not primary_path.exists():
                    target = primary_path
                    msg = "✅ **Đã có session** — saved as `primary.txt`"
                else:
                    target = cookies_dir / f"upload_{int(_now())}.txt"
                    msg = f"✅ Đã thêm backup `{target.name}` — rotation +1"
                target.write_bytes(content)
                _cfg.CFG.cookies.dir = cookies_dir
                st.session_state["_cookie_flash"] = ("success", msg)
                st.session_state["_cookie_uploader_n"] = _uploader_n + 1
                st.rerun()

# ---------- Main: input & run ----------
url = st.text_input("Link YouTube", placeholder="https://www.youtube.com/watch?v=...")
instruction = st.text_input(
    "Chỉ dẫn thêm (tuỳ chọn)",
    placeholder="Vd: Chỉ tóm tắt 5 phút đầu. / Từ phút 3 đến phút 10.",
)

col1, col2 = st.columns([1, 3])
run = col1.button("Tóm tắt", type="primary", use_container_width=True, disabled=not url)
col2.caption(f"Provider: `{provider}` / model: `{model}`")

if run:
    t0 = time.time()
    status = st.status("Đang xử lý…", expanded=True)
    events_log: list = []

    def on_event(event: dict) -> None:
        events_log.append(event)
        with status:
            t = event.get("type")
            if t == "agent_start":
                st.write("● Agent khởi động…")
            elif t == "tool_call":
                args_str = json.dumps(event["args"], ensure_ascii=False, indent=2)
                st.write(f"⚙️ Calling `{event['name']}`")
                st.code(args_str, language="json")
            elif t == "tool_result":
                chars = len(event["content"])
                st.write(f"✅ `{event['name']}` → {chars} chars")
            elif t == "repair":
                st.write("⚠️ Phát hiện chữ Hán → đang dịch lại sang tiếng Việt…")
            elif t == "done":
                st.write("● Hoàn tất.")

    try:
        summary = asyncio.run(
            summarize(
                url,
                instruction,
                provider=provider,
                model=model or None,
                api_key=api_key or None,
                on_event=on_event,
            )
        )
        dt = time.time() - t0
        status.update(label=f"✅ Xong trong {dt:.1f}s", state="complete", expanded=False)

        st.markdown("### Kết quả")
        st.markdown(summary)
        with st.expander("Sao chép / dán"):
            st.code(summary, language="markdown")
    except Exception as e:
        title, hint = _friendly_error(e)
        status.update(label=f"❌ {title}", state="error")
        st.error(f"**{title}**\n\n{hint}")
        with st.expander("Chi tiết lỗi (dev)"):
            st.exception(e)
