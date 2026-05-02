"""Text-to-speech synthesis. Pure function, không liên quan MCP/agent.

Provider khuyến nghị: `edge-tts` (free, Microsoft Edge voices, hỗ trợ tiếng Việt).
Cài: pip install edge-tts

Khi dùng provider khác (gTTS / OpenAI TTS / ElevenLabs / Coqui):
- Đổi function `_synthesize_*` bên dưới
- Hoặc thêm provider mới rồi switch trong `synthesize()`
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import CFG, PROJECT_ROOT


DEFAULT_VOICE_VI = "vi-VN-HoaiMyNeural"     # nữ, miền Bắc
DEFAULT_VOICE_VI_MALE = "vi-VN-NamMinhNeural"  # nam


def _output_dir() -> Path:
    """Thư mục lưu file audio. Tạo nếu chưa có."""
    # TODO: lấy từ CFG.tts.output_dir khi đã wire vào config.py
    out = PROJECT_ROOT / "output" / "audio"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _make_filename(prefix: str = "tts") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.mp3"


async def _synthesize_edge_tts(text: str, voice: str, out_path: Path) -> None:
    """Dùng Microsoft Edge TTS (free, no API key)."""
    # TODO: import edge_tts
    # import edge_tts
    # communicate = edge_tts.Communicate(text, voice)
    # await communicate.save(str(out_path))
    raise NotImplementedError("Cài edge-tts: pip install edge-tts, rồi mở comment ở trên.")


def synthesize(
    text: str,
    voice: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Synthesize text → file audio MP3. Trả về absolute path đến file đã tạo.

    Args:
        text: Văn bản cần đọc.
        voice: Tên voice (mặc định vi-VN-HoaiMyNeural).
        output_path: Đường dẫn output (mặc định output/audio/tts_<timestamp>.mp3).

    Returns:
        Path đến file MP3 đã tạo.
    """
    voice = voice or DEFAULT_VOICE_VI
    out = output_path or (_output_dir() / _make_filename())
    asyncio.run(_synthesize_edge_tts(text, voice, out))
    return out
