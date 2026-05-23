# 📓 TuBeNote

> Vietnamese-first AI assistant for **chatting with YouTube videos** — summarize + memory-aware Q&A. Multi-LLM (Gemini / OpenAI / Anthropic / Ollama), Hybrid RAG (Dense + BM25), Whisper STT fallback, hybrid web search.

Near-zero cost with Gemini free tier + local Ollama embedding + personal cookies. Tested on Windows 11 + RTX 4050 6GB.

> ⚠️ **Work in progress:** Vietsub burn-in and TTS dubbing are experimental and **not shipped in this release**. See roadmap below.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📺 **Video summarization** | Paste a YouTube link → AI summarizes in Vietnamese (topic, key points, highlights, conclusion) |
| 💬 **Memory-aware Q&A** | Chat about the video with a 10-message sliding-window memory |
| 🔍 **Hybrid RAG** | Dense (Chroma + qwen3-embedding) + Sparse (BM25) + RRF fusion |
| 🌐 **Hybrid web search** | Self-hosted SearXNG; auto-triggers when RAG relevance is low |
| 🤖 **Multi-LLM** | Switch between Gemini / OpenAI / Anthropic / Ollama in the UI |
| 🎙️ **Whisper STT fallback** | Prefer yt-dlp manual subtitles; fall back to Whisper `base.en` / `base` multilingual |
| 🍪 **Cookie rotation** | Upload `cookies.txt` via the UI, or set `YT_COOKIES_PATH` in `.env` |

---

## 🧱 Tech stack

| Layer | Choice |
|---|---|
| **UI** | Streamlit |
| **LLM** | LangChain — Google Gemini / OpenAI / Anthropic / Ollama |
| **Vector store** | Chroma (cosine similarity) |
| **Embedding** | Ollama `qwen3-embedding:0.6b` (1024-dim, multilingual VN + EN) |
| **Sparse retrieval** | `rank_bm25` |
| **Fusion** | Reciprocal Rank Fusion (RRF, k=60) |
| **Transcript** | `yt-dlp` (manual subs) → `openai-whisper` (STT fallback) |
| **Web search** | Self-hosted SearXNG via Docker Compose |

---

## 🚀 Quick start

### 1. Requirements

- **Python 3.10+**
- **[Ollama](https://ollama.com/download)** — required for embedding (the app auto-starts the daemon if not running)
- **Docker** (optional) — only needed for SearXNG web search

### 2. Setup

```bash
# Clone + virtualenv
git clone https://github.com/buitrongtrinh/TubeNote.git TuBeNote
cd TuBeNote
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Pull the Ollama embedding model (one-time)
ollama pull qwen3-embedding:0.6b

# Create .env from template
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY (Gemini free tier: https://aistudio.google.com/apikey)
```

### 3. Run

```bash
streamlit run apps/streamlit_app.py
```

The app opens at `http://localhost:8501`. Paste a YouTube link → summary is generated automatically → chat freely.

### 4. Web search (optional)

```bash
docker compose up -d            # starts SearXNG on port 8888
```

In the Streamlit sidebar, switch the "🌐 Web search" radio to **Auto** or **Always**.

---

## ⚙️ Configuration

### `config.yaml`

All non-secret settings. Three important sections:

```yaml
llm:
  provider: local                          # default: gemini / openai / anthropic / local
  google:
    models: [gemini-2.5-flash-lite, ...]   # priority list — first entry is the default
  local:
    models: [llama3.2:latest]
    base_url: http://localhost:11434

rag:
  similarity_threshold: 0.55               # cosine threshold for Dense
  fetch_k: 30                              # pool size per retriever (RRF input)
  final_k: 10                              # cap on chunks sent to the LLM (RRF output)

embedding:
  model: qwen3-embedding:0.6b              # project-locked — changing requires wiping output/chroma/
```

### `.env`

API keys + paths. See `.env.example` for the supported variables.

---

## 🏗️ Architecture

```
YouTube URL
   ↓
[fetch metadata]  ← yt-dlp
   ↓
[fetch transcript]
   ├─ yt-dlp manual subtitle (preferred)
   └─ Whisper STT fallback (base.en / base multilingual, language auto-detected from title)
   ↓
   ┌────────────────┬────────────────┐
   ↓                ↓                ↓
[summary]      [chunking]      (Streamlit render)
LLM 1 call    RecursiveSplitter
              chunk=500 / overlap=100
                   ↓
              [embed via Ollama qwen]
                   ↓
              [Chroma store]
                   ↓
              (ready for chat)

Q&A:
question → [Dense Chroma top-30] ─┐
        → [BM25 top-30]            ├─→ [RRF fuse] → top-10 → [LLM answer]
        → (optional web search)   ─┘
```

---

## 🗺️ Roadmap

- [ ] **Vietsub burn-in** — generate `.srt` and mux into the video via ffmpeg
- [ ] **Audio dubbing** — Whisper segments → translate → TTS → mux
- [ ] **Multi-video research mode** — synthesize content across multiple videos
- [ ] **Timestamp-aware citations** — link `(per video, 2:30)` → `youtu.be/...?t=150s`
- [ ] **Streaming LLM responses** — token-by-token output instead of waiting for full response
- [ ] **AI Agent mode** — ReAct loop with tool use

---

## 📁 Project structure

```
tubenote/
  config.py                       # YAML + .env loader
  ollama_runtime.py               # auto-start Ollama daemon
  llm/providers/                  # google, openai, anthropic, local builders
  pipeline/
    summarizer.py                 # video → summary
    qa.py                         # RAG + memory + web search
  processors/
    youtube/
      transcript.py               # orchestrator: yt-dlp → Whisper fallback
      transcript_yt_dlp.py        # manual sub fetcher + metadata
      transcript_whisper.py       # Whisper STT (base.en / base multilingual)
      cookies.py, utils.py, types.py
    rag/
      chunker.py                  # 500-char chunks, 100 overlap
      embedder.py                 # Ollama qwen3-embedding client
      store.py                    # Chroma store + get_all_docs (for BM25)
      pipeline.py                 # ingest = fetch → chunk → embed → store
    web/
      searxng.py                  # JSON API client

apps/
  streamlit_app.py                # main UI
  qa.py                           # CLI Q&A
  stop_services.py                # stop Docker + Ollama

searxng_config/                   # SearXNG settings
docker-compose.yml                # SearXNG service definition
```

---

## 📄 License

MIT.
