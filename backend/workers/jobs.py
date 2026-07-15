"""Job runner tổng quát (Asynchronous Request-Reply pattern).

Dùng chung cho mọi tác vụ chạy lâu (load/Whisper, dub/TTS). Bản tối giản:
dict trong RAM + BackgroundTasks. Sau này nhiều user thì nâng lên Celery/Redis.
"""
from __future__ import annotations

import uuid
from typing import Callable

# job_id -> {status, stage, progress, error, result}
JOBS: dict[str, dict] = {}

# Giữ RAM bounded: mỗi job 'done'/'error' còn ôm nguyên result (metadata +
# prompts). Khi vượt ngưỡng, evict các job đã kết thúc cũ nhất (dict giữ thứ tự
# chèn từ Python 3.7). Job đang 'running' không bao giờ bị xoá.
MAX_JOBS = 200


def _evict() -> None:
    overflow = len(JOBS) - MAX_JOBS
    if overflow <= 0:
        return
    for job_id in list(JOBS.keys()):
        if overflow <= 0:
            break
        if JOBS[job_id].get("status") in ("done", "error"):
            JOBS.pop(job_id, None)
            overflow -= 1


def create() -> str:
    """Tạo job mới ở trạng thái 'running', trả job_id."""
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "running", "stage": "Bắt đầu", "progress": 0,
        "error": None, "result": None, "cancel_requested": False,
    }
    _evict()
    return job_id


def request_cancel(job_id: str) -> bool:
    """Yêu cầu hủy job đang chạy (hủy HỢP TÁC — không kill được thread giữa
    chừng). Đặt cờ để ``fn`` tự kiểm tra ở checkpoint an toàn rồi dừng.

    Trả True nếu đặt được cờ (job tồn tại & đang chạy), False nếu job đã kết
    thúc/không tồn tại (không có gì để hủy)."""
    job = JOBS.get(job_id)
    if not job or job.get("status") != "running":
        return False
    job["cancel_requested"] = True
    return True


def is_cancelled(job_id: str) -> bool:
    """``fn`` gọi để biết có yêu cầu hủy hay chưa (kiểm tra ở checkpoint)."""
    job = JOBS.get(job_id)
    return bool(job and job.get("cancel_requested"))


def run(job_id: str, fn: Callable[[Callable], object]) -> None:
    """Chạy ``fn(update)`` ở nền. ``update(**fields)`` để báo tiến độ.

    Kết quả trả về của ``fn`` được lưu vào ``result``.
    """
    def update(**fields):
        if job_id in JOBS:
            JOBS[job_id].update(**fields)

    try:
        result = fn(update)
        update(status="done", progress=100, stage="Hoàn tất", result=result)
    except Exception as e:  # noqa: BLE001
        # fn ném exception SAU khi đã có yêu cầu hủy = hủy thành công (fn dừng
        # tại checkpoint), không phải lỗi thật -> trạng thái "cancelled".
        if is_cancelled(job_id):
            update(status="cancelled", stage="Đã hủy", error=None)
        else:
            update(status="error", error=str(e))


def status(job_id: str) -> dict | None:
    return JOBS.get(job_id)
