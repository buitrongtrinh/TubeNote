# 🎬 Link2Slide

> Agent tóm tắt video YouTube + đọc thành audio. Multi-LLM cloud (Gemini / OpenAI / Anthropic), Multi-tool MCP (transcript, metadata, **TTS**). UI Streamlit.

**Chi phí 0đ** với Gemini free tier + Edge TTS + cookies cá nhân. Test trên Windows 11, Python 3.10.

> ⚠️ **Đang phát triển:** TTS tool (text → speech) đang ở giai đoạn skeleton. Phụ đề + tóm tắt đã hoạt động ổn.

---

## 📺 Demo

[![Link2Slide demo](https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

▶️ **https://youtu.be/dQw4w9WgXcQ**

### Sample output

```markdown
📺 **Jack Ma: You're Supposed to Spend Money on Your People**
👤 Kênh: [World Economic Forum](https://www.youtube.com/@wef)
![thumbnail](https://i.ytimg.com/vi/WsQ7ysVt-0A/hqdefault.jpg)

---

**Chủ đề**: Jack Ma chia sẻ về cuộc gặp với Donald Trump...

**Ý chính**:
- Cuộc gặp "rất hiệu quả" với Tổng thống đắc cử...
- Cam kết tạo 1 triệu việc làm tại Mỹ qua Alibaba...

**Kết luận**: ...

🔊 [audio_summary.mp3]  ← TTS narration (đang triển khai)
```

---

## ✨ Tính năng

| | Status |
|---|---|
| 🤖 Multi-LLM (`google` / `openai` / `anthropic`) | ✅ Done |
| 🔌 MCP stdio server, mỗi tool 1 file, auto-register | ✅ Done |
| 📺 YouTube transcript + time-range + cookie rotation | ✅ Done |
| 📋 Video metadata (title, channel, thumbnail) | ✅ Done |
| 🌐 Fallback ngôn ngữ + auto-translate | ✅ Done |
| 🖥️ Streamlit UI với event streaming | ✅ Done |
| 🔊 **TTS — đọc tóm tắt thành audio** | 🚧 In progress |
| 🎵 Voice picker (vi/en, nam/nữ) | 🚧 In progress |
| 🎤 Whisper STT cho video không phụ đề | 📅 Planned |

---

## 🏗️ Kiến trúc

```
┌──────────────┐
│ Streamlit UI │ ─┐
└──────────────┘  │
┌──────────────┐  │     ┌───────────────┐        ┌──────────────────────────┐
│  CLI         │ ─┼──▶ │ Summarizer     │ ─tools▶│ MCP Server (stdio)       │
└──────────────┘  │     │ Agent          │        │  ├ youtube/              │
                  │     │ (LangGraph)    │        │  │  ├ transcript         │
                  │     └───────┬────────┘        │  │  ├ metadata           │
                  │             │                 │  │  └ languages          │
                  │             │ LLM             │  └ tts/  🚧              │
                  │             ▼                 │     ├ synthesize         │
                  │     ┌───────────────┐        │     └ list_voices        │
                  │     │ make_llm()    │        └─────────────┬────────────┘
                  │     │  • google     │                      │
                  │     │  • openai     │        ┌─────────────▼────────────┐
                  │     │  • anthropic  │        │ Data layer:              │
                  │     └───────────────┘        │  ├ youtube/              │
                  │                              │  │  ├ transcripts.py     │
┌─────────────────┴──┐                           │  │  └ cookies.py         │
│ mcp_cli (debug)    │                           │  └ tts/                  │
└────────────────────┘                           │     ├ synthesize.py 🚧   │
                                                 │     └ voices.py     🚧   │
                                                 └──────────────────────────┘
```

---

## 🚀 Quickstart

```powershell
# 1. Clone + venv
git clone <repo>
cd Link2Slide
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. API key Gemini (free)
copy .env.example .env
notepad .env
# → GOOGLE_API_KEY=AIza... (lấy tại https://aistudio.google.com/apikey)

# 3. Cookies YouTube (khuyến nghị — tránh bị block IP)
# → Cài extension "Get cookies.txt LOCALLY" trên Chrome
# → Login youtube.com, export → lưu vào cookies\primary.txt

# 4. Chạy
streamlit run apps\streamlit_app.py
```

---

## 📁 Cấu trúc project

```
Link2Slide/
├── config.yaml / .env / requirements.txt
├── cookies/                              # cookies/*.txt (rotation)
├── output/audio/                         # 🆕 file MP3 do TTS sinh ra
├── apps/                                 # entrypoints
│   ├── cli.py
│   ├── mcp_cli.py
│   ├── mcp_server.py
│   └── streamlit_app.py
└── link2slide/
    ├── config.py
    ├── prompts.py
    ├── llm/                              # provider factory
    │   ├── __init__.py
    │   ├── google.py / openai.py / anthropic.py
    ├── agents/summarizer.py              # LangGraph ReAct agent
    ├── mcp/
    │   ├── client.py / server.py
    │   └── tools/
    │       ├── __init__.py               # auto-discover domain
    │       ├── youtube/                  # ✅ done
    │       │   ├── __init__.py
    │       │   ├── transcript.py
    │       │   ├── metadata.py
    │       │   └── languages.py
    │       └── tts/                      # 🚧 đang triển khai
    │           ├── __init__.py
    │           ├── synthesize.py         # MCP tool: synthesize_text_to_speech
    │           └── list_voices.py        # MCP tool: list_tts_voices
    ├── youtube/                          # YouTube data layer
    │   ├── cookies.py
    │   └── transcripts.py
    └── tts/                              # 🚧 TTS data layer
        ├── synthesize.py                 # pure logic — gọi edge-tts/gTTS
        └── voices.py                     # list voices
```

---

## 🔧 Triển khai TTS — cấu trúc mong đợi

### File 1: `link2slide/tts/synthesize.py` (data layer, pure)

```python
"""TTS pure logic. Provider khuyến nghị: edge-tts (free, có giọng Việt)."""
from pathlib import Path
import asyncio
import edge_tts                           # pip install edge-tts

DEFAULT_VOICE_VI = "vi-VN-HoaiMyNeural"

async def _synthesize_edge_tts(text: str, voice: str, out_path: Path) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))

def synthesize(text: str, voice: str = None, output_path: Path = None) -> Path:
    """Sinh file MP3 từ text. Trả về path file đã tạo."""
    voice = voice or DEFAULT_VOICE_VI
    out = output_path or _make_default_path()
    asyncio.run(_synthesize_edge_tts(text, voice, out))
    return out
```

### File 2: `link2slide/tts/voices.py`

```python
"""List voices từ provider."""
import asyncio
import edge_tts

def list_voices(language: str = None) -> list[dict]:
    voices = asyncio.run(edge_tts.list_voices())
    if language:
        voices = [v for v in voices if v["Locale"].startswith(language)]
    return voices
```

### File 3: `link2slide/mcp/tools/tts/synthesize.py` (MCP wrapper)

```python
"""MCP tool — wrap pure logic thành tool LLM gọi được."""
from mcp.server.fastmcp import FastMCP
from ....tts.synthesize import synthesize

def synthesize_text_to_speech(text: str, voice: str = None) -> str:
    """Convert text to MP3. Returns absolute file path."""
    return str(synthesize(text=text, voice=voice))

def register(mcp: FastMCP) -> None:
    mcp.tool()(synthesize_text_to_speech)
```

### File 4: `link2slide/mcp/tools/tts/list_voices.py`

```python
"""MCP tool — list voices."""
import json
from mcp.server.fastmcp import FastMCP
from ....tts.voices import list_voices

def list_tts_voices(language: str = None) -> str:
    """Returns JSON list of voices."""
    return json.dumps(list_voices(language), ensure_ascii=False, indent=2)

def register(mcp: FastMCP) -> None:
    mcp.tool()(list_tts_voices)
```

### File 5: `link2slide/mcp/tools/tts/__init__.py`

```python
"""Domain hub — gọi register cho từng tool."""
from . import list_voices, synthesize

def register(mcp):
    synthesize.register(mcp)
    list_voices.register(mcp)
```

### Flow cuối

1. `tools/__init__.py::register_all()` scan folder → thấy domain `tts/`
2. Import `tts/__init__.py` → gọi `tts.register(mcp)`
3. `tts.register()` gọi `synthesize.register(mcp)` + `list_voices.register(mcp)`
4. Mỗi tool đăng ký vào FastMCP → agent thấy thêm 2 tool

**Pattern này đồng nhất với folder `youtube/`** — không sửa file ngoài, chỉ thêm folder mới.

### Update prompt để agent dùng TTS

Trong [link2slide/prompts.py](link2slide/prompts.py), thêm section:

```
"4. Sau khi viết xong tóm tắt: nếu user yêu cầu 'đọc', 'audio', 'voice' "
"→ gọi synthesize_text_to_speech(text=<bản tóm tắt>) → đính kèm path MP3 vào output."
```

---

## ⚙️ Cấu hình

**`.env`** — secrets:
```ini
GOOGLE_API_KEY=AIza...
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
YT_COOKIES_DIR=cookies
```

**`config.yaml`** — defaults:
```yaml
llm:
  provider: google
  google:
    model: gemini-2.5-flash

transcript:
  default_languages: [vi, en]

tts:                              # 🚧 TTS config
  provider: edge_tts              # edge_tts | gtts | openai | elevenlabs
  default_voice: vi-VN-HoaiMyNeural
  output_dir: output/audio

agent:
  max_iterations: 6
```

---

## 💻 Sử dụng

```powershell
# CLI
python -m apps.cli "https://youtu.be/VIDEO_ID"
python -m apps.cli "https://youtu.be/VIDEO_ID" "Chỉ tóm tắt 5 phút đầu"

# Streamlit UI
streamlit run apps\streamlit_app.py

# Debug MCP tool trực tiếp
python -m apps.mcp_cli list
python -m apps.mcp_cli call get_youtube_transcript url="https://youtu.be/VIDEO_ID" end_seconds=300

# (Sau khi triển khai TTS)
python -m apps.mcp_cli call synthesize_text_to_speech text="Xin chào" voice=vi-VN-NamMinhNeural
python -m apps.mcp_cli call list_tts_voices language=vi
```

---

## 🔧 Mở rộng — thêm domain tool mới

Theo pattern domain folder (giống `youtube/`, `tts/`):

```
link2slide/mcp/tools/<domain>/
├── __init__.py          # register tất cả tool con
├── tool_a.py            # def tool_a(...) + def register(mcp)
└── tool_b.py
```

Server tự discover folder mới. Không cần sửa `server.py`.

---

## ⚠️ Hạn chế

- Chỉ chạy với video **có phụ đề**. Roadmap: Whisper + yt-dlp cho video không phụ đề.
- Cookies TTL **2–8 tuần** → export lại định kỳ.
- Free tier Gemini ~10 RPM, **20 RPD** cho `gemini-2.5-flash`. Hết quota → đổi `gemini-2.5-flash-lite` (~1000 RPD) hoặc bật billing.

---

## 🐛 Troubleshooting

| Lỗi | Fix |
|---|---|
| `ModuleNotFoundError: link2slide` (Streamlit) | Chạy từ project root: `streamlit run apps\streamlit_app.py` |
| `RequestBlocked` (transcript) | Set cookies `cookies\primary.txt`, hoặc chờ 30–60p |
| `RESOURCE_EXHAUSTED` (Gemini) | Hết quota free tier — đổi model hoặc đợi reset (PT timezone) |
| `Thiếu GOOGLE_API_KEY` | Set `.env` hoặc nhập trong sidebar Streamlit |
| TTS fail (sau khi triển khai) | `pip install edge-tts`. Test: `python -c "import edge_tts"` |

---

## 📜 License

MIT — dùng tự do, không bảo hành.
