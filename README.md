# TubeNote

[![YouTube Demo](https://img.shields.io/badge/YouTube-Demo-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=YOUR_GUI_DEMO_VIDEO_ID)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![Hardware](https://img.shields.io/badge/Hardware-CPU--only%20OK%20%7C%20GPU%20optional-2ea44f)](#prerequisites)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TubeNote is a local-first AI video dubbing and video Q&A application. It turns a
YouTube video into a Vietnamese dubbed video, keeps editable subtitles and timing
metadata, and adds a RAG chat panel so users can ask questions about the video.

The project is built as a practical full-stack system around video localization:
subtitle acquisition, ASR fallback, duration-aware translation, TTS generation,
speech/subtitle alignment, background audio preservation, and hybrid retrieval
over processed transcripts.

**The entire pipeline runs on a CPU-only machine** (faster-whisper `small.en`
int8 + Supertonic TTS). An NVIDIA GPU is optional and unlocks the
higher-quality path: OmniVoice TTS with voice cloning and `medium.en` ASR.

![TubeNote detailed architecture](docs/assets/tubenote-architecture.webp)

## Demo

Click a thumbnail to watch on YouTube:

| GUI walkthrough | Dubbed output sample |
| :---: | :---: |
| [![GUI walkthrough](https://img.youtube.com/vi/YOUR_GUI_DEMO_VIDEO_ID/hqdefault.jpg)](https://www.youtube.com/watch?v=YOUR_GUI_DEMO_VIDEO_ID) | [![Dubbed output sample](https://img.youtube.com/vi/YOUR_DUBBED_OUTPUT_VIDEO_ID/hqdefault.jpg)](https://www.youtube.com/watch?v=YOUR_DUBBED_OUTPUT_VIDEO_ID) |

<!-- Replace YOUR_GUI_DEMO_VIDEO_ID / YOUR_DUBBED_OUTPUT_VIDEO_ID (3 places:
     the badge on top and the two thumbnails) with real YouTube video ids
     before publishing. -->

## What TubeNote Does

1. Loads a YouTube video and metadata with `yt-dlp`.
2. Gets English subtitles from YouTube when available.
3. Falls back to faster-whisper ASR when subtitles are unavailable.
4. Generates duration-aware translation prompts from subtitle segments.
5. Lets the user translate manually, or translate batches through an LLM API.
6. Validates translated batches before TTS.
7. Generates Vietnamese speech with Supertonic CPU or OmniVoice GPU.
8. Aligns generated speech and Vietnamese subtitles to the video timeline.
9. Optionally separates and preserves background audio with Demucs.
10. Serves the final MP4 in a Vidstack player with subtitle controls.
11. Builds a hybrid RAG index over local subtitles for video Q&A.

## Highlights

- Local-first workflow: generated media, subtitles, Chroma indexes, summaries,
  logs, cookies, and voice samples stay outside git under ignored runtime paths.
- Two ASR presets: CPU-friendly `small.en` and GPU `medium.en`, both through
  faster-whisper/CTranslate2.
- Two translation modes:
  - Manual: copy prompts to ChatGPT and paste validated results back.
  - API: choose provider/model and translate batches directly from TubeNote.
- Two TTS engines:
  - Supertonic CPU: fast default path, fitted to timing slots after generation.
  - OmniVoice GPU: higher quality path with duration-conditioned generation and
    voice reference support.
- Per-segment regeneration after dubbing, including pronunciation mapping.
- Vietnamese subtitles are aligned to generated TTS timing, not only raw source
  subtitles.
- Optional Demucs background preservation for videos with music or ambience.
- Hybrid RAG: Chroma dense retrieval with `BAAI/bge-m3`, BM25 sparse retrieval,
  and Reciprocal Rank Fusion.
- RAG chat starts by creating/loading one cached video summary, then uses the
  selected LLM provider/model for the chat session.

## Tech Stack

| Area | Stack |
| --- | --- |
| Frontend | Next.js 14, React, Vidstack |
| Backend | FastAPI, Python 3.11 |
| Video ingest | `yt-dlp`, YouTube subtitles |
| ASR | faster-whisper, CTranslate2 |
| Translation | Prompt workflow plus DeepSeek/OpenAI/Gemini/Anthropic via LangChain |
| TTS | Supertonic, OmniVoice |
| Audio/video | ffmpeg, imageio-ffmpeg, SoundFile, Demucs, torch/torchaudio |
| RAG | LangChain, Chroma, `BAAI/bge-m3`, BM25, RRF |
| Runtime jobs | FastAPI background tasks with in-memory job status |

## Repository Layout

```text
backend/
  api/                 FastAPI routers for dubbing, video, and RAG
  llm/providers/       Provider adapters for DeepSeek, OpenAI, Google, Anthropic
  pipeline/            High-level dubbing and Q&A orchestration
  services/
    dubbing/           TTS validation, timing fit, background separation, logs
    rag/               Chunking, embedding, Chroma store, summary cache
    video/             WebVTT and timing helpers
    youtube/           yt-dlp metadata/subtitle fetch and Whisper fallback
  workers/             Lightweight background job registry

frontend/
  app/                 Next.js app routes: library, drafts, add, video detail
  components/          Vidstack player and transcript UI
  lib/api.js           Frontend API client through Next rewrites

docs/
  architecture.md      System boundaries and data flow
  dubbing.md           Dubbing pipeline details
  rag.md               RAG indexing and Q&A details
```

## Documentation

Implementation details live in `docs/`:

- [Architecture](docs/architecture.md)
- [Dubbing Pipeline](docs/dubbing.md)
- [RAG Pipeline](docs/rag.md)

The README is intentionally kept as the project entry point. Detailed design
notes and pipeline behavior are documented in the files above.

## Prerequisites

- Python 3.11.
- Node.js 20 recommended.
- `ffmpeg` installed on PATH is recommended. Some operations also use
  `imageio-ffmpeg` as a fallback.
- CPU-only machines can run the default Supertonic + Whisper CPU path, but ASR
  and TTS will be slower.
- NVIDIA GPU is recommended for OmniVoice and faster Whisper GPU mode.
- Docker is optional and only needed if you enable local SearXNG web search.

## Setup

### Backend

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your machine needs a specific PyTorch CUDA/MPS/XPU build, install the matching
`torch` and `torchaudio` wheels first, then install `requirements.txt`.

For faster-whisper GPU mode on Linux, `requirements.txt` includes CUDA 12
`nvidia-cublas-cu12` and `nvidia-cudnn-cu12` wheels. `run.sh` adds installed
NVIDIA wheel library directories to `LD_LIBRARY_PATH`.

### Frontend

```bash
cd frontend
npm install
cd ..
```

### Environment

```bash
cp .env.example .env
```

Fill only the providers/features you use. Current defaults:

- Translation API UI defaults to DeepSeek Flash when API mode is selected.
- RAG UI defaults to DeepSeek unless changed in `backend/config.yaml`.
- TTS defaults to Supertonic CPU.
- ASR defaults to the CPU preset in `backend/config.yaml`.

Minimal API setup for DeepSeek:

```text
DEEPSEEK_API_KEY=...
```

Optional alternative providers:

```text
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
```

## YouTube Cookies

Cookies are optional, but useful when YouTube blocks unauthenticated subtitle or
media requests.

```text
YT_COOKIES_DIR=cookies
# or
YT_COOKIES_PATH=cookies/primary.txt
```

To create a cookies file:

1. Install a browser extension that exports cookies in Netscape `cookies.txt`
   format, for example "Get cookies.txt LOCALLY".
2. Log in to YouTube in that browser.
3. Export cookies for `youtube.com`.
4. Save the file as `cookies/primary.txt`.

The `cookies/` directory is ignored by git. Do not commit exported cookies.

Alternatively, let `yt-dlp` read cookies from your browser profile:

```text
YT_COOKIES_BROWSER=chrome
```

## Optional Web Search

Video-local RAG works without web search. If you want to experiment with web
search fallback, run SearXNG separately and configure the URL:

```bash
docker compose up -d searxng
```

```text
SEARXNG_URL=http://localhost:8888
```

## Run

```bash
./run.sh
```

Default URLs:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8010`
- Backend docs: `http://localhost:8010/docs`

`run.sh` detects `.venv/bin/python`, then falls back to
`~/miniconda3/envs/tubenote`, then to `python3`. Override when needed:

```bash
PYTHON_BIN=/path/to/python BACKEND_PORT=8010 ./run.sh
```

Manual run:

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010
cd frontend
NEXT_PUBLIC_API=http://localhost:8010 npm run dev
```

## Typical Workflow

1. Open `Tạo lồng tiếng`.
2. Choose ASR model. CPU is the default compatibility path.
3. Load a YouTube URL.
4. Choose TTS engine in the TTS panel.
5. Choose translation mode:
   - Manual: copy prompts, translate externally, paste back.
   - API: choose provider/model and let TubeNote translate batches.
6. Validate translated batches.
7. Optionally adjust voice, quality, and background preservation.
8. Start dubbing.
9. Review the output in the player.
10. Regenerate individual segments when pronunciation or delivery needs fixing.
11. Use the Q&A tab to generate/load a summary and ask questions over the video.

## Runtime Data

Generated files are ignored by git:

- `data/` (including runtime voice references in `data/voice_clones/`)
- `cookies/`
- `.env`

`voice_clones/` at the repository root is committed on purpose: it contains the
built-in voice preset references for the two TTS engines, which the voice
dropdown depends on.

Important runtime paths:

```text
data/metadata/          YouTube metadata JSON
data/sub_raw/           Raw subtitles from YouTube or Whisper
data/sub_vi_super/      Supertonic translated subtitle/timing JSON
data/sub_vi_omni/       OmniVoice translated subtitle/timing JSON
data/audio_dub/         Generated dubbed speech
data/video_dub/         Final MP4 outputs
data/chroma/            RAG vector stores
data/rag_summary/       Cached video summaries
data/logs/              CSV timing/performance logs
data/voice_clones/      Runtime voice references generated from source videos
```

## Checks

```bash
python -m compileall -q backend
python -m unittest discover -s backend/tests -t .
cd frontend
npm run build
```

Use the project environment Python when dependencies are installed outside the
default shell Python, for example:

```bash
~/miniconda3/envs/tubenote/bin/python -m unittest discover -s backend/tests -t .
```

## Troubleshooting

### `Port 8010 đang bận`

An old backend process is still using the backend port. Stop it before rerunning:

```bash
pkill -f 'uvicorn backend.main'
```

### The UI first shows `Internal Server Error`, then works after refresh

This usually means Next.js opened before FastAPI finished startup. The first
frontend request to `/api/library` can fail if the proxy target is not ready
yet. Wait a few seconds and refresh, or start backend manually before frontend.

### `libcublas.so.12` or cuDNN errors

faster-whisper GPU mode needs CUDA runtime libraries. Install the CUDA 12
NVIDIA wheels from `requirements.txt`, or switch ASR to CPU mode.

### Demucs downloads weights or fails on first run

Demucs may download model weights on first background-separation use. If
background preservation is not needed, disable it in the dubbing UI.

### RAG embedding model changes

Changing `embedding.model` changes vector dimensions and embedding space. Delete
`data/chroma/` before rebuilding indexes.

## Project Limitations

- Runtime jobs are stored in memory. A backend restart loses active job status.
- Local browser draft state is used during dubbing, not a multi-user database.
- TTS quality depends on subtitle timing, translation length, model behavior,
  and segment duration.
- Background preservation is source-dependent; music-heavy videos usually work
  better than noisy speech with overlapping vocals.
- Generated media and downloaded YouTube content are intentionally excluded from
  the public repository.

## CV Summary

TubeNote - Personal AI Video Dubbing and RAG Assistant.

Built a full-stack application that extracts YouTube transcripts, translates
them into Vietnamese, generates dubbed speech with multiple TTS engines, aligns
audio/subtitles, preserves background audio, and provides hybrid RAG-based Q&A
over processed video content.

Tech: Next.js, React, Vidstack, FastAPI, faster-whisper, CTranslate2, Chroma,
LangChain, BM25, BAAI/bge-m3, DeepSeek/OpenAI/Gemini/Anthropic APIs,
Supertonic, OmniVoice, Demucs, ffmpeg.

## License

MIT.
