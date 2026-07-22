#!/usr/bin/env bash
# Chạy backend FastAPI + frontend Next.js, tự mở browser vào đúng port.
# Ctrl+C dừng cả hai. Chạy từ repo root: ./run.sh
cd "$(dirname "$0")"

if [ -n "${PYTHON_BIN:-}" ]; then
  PY="$PYTHON_BIN"
elif [ -x ".venv/bin/python" ]; then
  PY="$(pwd)/.venv/bin/python"
elif [ -x "$HOME/miniconda3/envs/tubenote/bin/python" ]; then
  PY="$HOME/miniconda3/envs/tubenote/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo "Không tìm thấy Python. Tạo .venv hoặc set PYTHON_BIN=/path/to/python"
  exit 1
fi

ENV_BIN="$(dirname "$PY")"
NPM_BIN="${NPM_BIN:-npm}"
CHROMIUM="${BROWSER_BIN:-}"
if [ -z "$CHROMIUM" ]; then
  for candidate in chromium-browser chromium google-chrome google-chrome-stable; do
    if command -v "$candidate" >/dev/null 2>&1; then
      CHROMIUM="$(command -v "$candidate")"
      break
    fi
  done
fi
FE_LOG="$(mktemp)"
BACKEND_PORT="${BACKEND_PORT:-8010}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_API="${NEXT_PUBLIC_API:-http://localhost:${BACKEND_PORT}}"

CUDA_LIB_DIRS=$("$PY" - <<'PY'
from pathlib import Path
import site

roots = []
for getter in (site.getsitepackages,):
    try:
        roots.extend(Path(p) / "nvidia" for p in getter())
    except Exception:
        pass
try:
    roots.append(Path(site.getusersitepackages()) / "nvidia")
except Exception:
    pass

dirs = []
for root in roots:
    if not root.exists():
        continue
    for pattern in ("**/libcublas.so*", "**/libcudnn.so*"):
        for lib in root.glob(pattern):
            path = str(lib.parent)
            if path not in dirs:
                dirs.append(path)

print(":".join(dirs))
PY
)
if [ -n "$CUDA_LIB_DIRS" ]; then
  export LD_LIBRARY_PATH="$CUDA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if ss -ltn 2>/dev/null | grep -q "${BACKEND_HOST}:${BACKEND_PORT}"; then
  echo "⚠ Port ${BACKEND_PORT} đang bận — backend cũ? Dừng: pkill -f 'uvicorn backend.main'"
fi

cleanup() {
  trap - INT TERM EXIT
  [ -n "${BE_PID:-}" ] && kill "$BE_PID" 2>/dev/null
  [ -n "${FE_PID:-}" ] && { kill "$FE_PID" 2>/dev/null; pkill -P "$FE_PID" 2>/dev/null; }
  [ -n "${TAIL_PID:-}" ] && kill "$TAIL_PID" 2>/dev/null
  [ -n "${OPEN_PID:-}" ] && kill "$OPEN_PID" 2>/dev/null
  pkill -f "uvicorn backend.main" 2>/dev/null
  rm -f "$FE_LOG"
}
trap cleanup INT TERM EXIT

# Backend — chạy từ repo root, entry backend.main:app
"$PY" -m uvicorn backend.main:app --reload --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
BE_PID=$!

# Frontend
( cd frontend && PATH="$ENV_BIN:$PATH" NEXT_PUBLIC_API="$BACKEND_API" exec "$NPM_BIN" run dev > "$FE_LOG" 2>&1 ) &
FE_PID=$!
tail -f "$FE_LOG" &
TAIL_PID=$!

# Mở Chromium khi Next báo "Local: http://localhost:PORT"
(
  for _ in $(seq 1 60); do
    [ -f "$FE_LOG" ] || break                         # log bị xóa → thoát, không spam
    url=$(grep -oE 'http://localhost:[0-9]+' "$FE_LOG" 2>/dev/null | head -1)
    if [ -n "$url" ] && curl -s -o /dev/null "$url"; then
      # setsid: đẩy trình duyệt sang process group RIÊNG. Không có nó, tiến
      # trình nền vẫn thừa hưởng process group của script (job control tắt
      # trong script), mà Ctrl+C thì gửi SIGINT cho CẢ nhóm foreground -> tắt
      # server kéo theo tắt luôn cửa sổ trình duyệt vừa mở. Chỉ lộ ra khi máy
      # chưa mở Chrome sẵn; có sẵn rồi thì lệnh này chỉ đẩy URL sang instance
      # cũ (khác nhóm) rồi tự thoát, nên trước đây lúc gặp lúc không.
      if [ -n "$CHROMIUM" ] && [ -x "$CHROMIUM" ]; then
        if command -v setsid >/dev/null 2>&1; then
          setsid "$CHROMIUM" "$url" >/dev/null 2>&1 &
        else
          "$CHROMIUM" "$url" >/dev/null 2>&1 &
        fi
      fi
      echo "▶ Đã mở $url"
      break
    fi
    sleep 1
  done
) &
OPEN_PID=$!

wait "$BE_PID" "$FE_PID"
