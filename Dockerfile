# TubeNote backend (FastAPI + Whisper/TTS pipeline).
#
# Build mặc định là CPU-only (torch bản CPU, nhẹ hơn nhiều so với bản CUDA):
#   docker compose build backend
# Build có GPU (torch CUDA + cuDNN/cuBLAS wheels cho faster-whisper):
#   docker compose -f docker-compose.yml -f docker-compose.gpu.yml build backend
FROM python:3.11-slim

ARG TORCH_FLAVOR=cpu

# - ffmpeg: ffmpeg-python (merge_video_audio) gọi binary "ffmpeg" từ PATH.
# - nodejs: yt-dlp cần JS runtime để giải n-challenge khi dùng cookie đăng
#   nhập (cookies.py bật js_runtimes {deno,node}); Debian bookworm có node 18,
#   đủ cho yt-dlp EJS.
# - build-essential (chỉ image GPU): Triton JIT cần C compiler ở runtime để
#   build launcher cho các CUDA kernel mà OmniVoice/PyTorch sử dụng.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg nodejs ca-certificates \
    && if [ "$TORCH_FLAVOR" != "cpu" ]; then \
        apt-get install -y --no-install-recommends build-essential; \
    fi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# cpu (mặc định): cài torch/torchaudio bản CPU TRƯỚC để requirements.txt không
# kéo bản CUDA (~5GB) về một cách vô ích. cuda: bỏ qua bước pin, để pip lấy
# torch mặc định (kèm CUDA) + cài requirements-gpu.txt.
COPY requirements.txt requirements-gpu.txt ./
RUN if [ "$TORCH_FLAVOR" = "cpu" ]; then \
        pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi \
    && pip install --no-cache-dir -r requirements.txt \
    && if [ "$TORCH_FLAVOR" != "cpu" ]; then \
        pip install --no-cache-dir -r requirements-gpu.txt; \
    fi

COPY backend ./backend
# Reference audio cho voice cloning OmniVoice (config.yaml trỏ tới).
COPY voice_clones ./voice_clones
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1

EXPOSE 8010
# 0.0.0.0 bắt buộc trong container: bind 127.0.0.1 thì port mapping của Docker
# không chạm được vào app.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8010"]
