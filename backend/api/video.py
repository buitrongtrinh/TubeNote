"""Route cho video: thư viện, metadata, stream phát video."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from backend.pipeline import dubbing
from backend.services.youtube.utils import load_json
from backend.services.video.vtt import build_vtt
from backend.services.video.timing import display_range

router = APIRouter(prefix="/api", tags=["video"])

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


@router.get("/library")
def library():
    """Danh sách video đã dub."""
    return dubbing.list_library()


@router.get("/drafts")
def drafts():
    """Danh sách video đã load transcript nhưng chưa dub."""
    return dubbing.list_drafts()


@router.delete("/video/{vid}")
def delete_video(vid: str):
    """Xóa video khỏi thư viện cùng các file cache/dub liên quan."""
    try:
        return dubbing.delete_library_video(vid)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/video/{vid}/meta")
def video_meta(vid: str):
    """Metadata đầy đủ của 1 video."""
    m = dubbing.load_metadata(vid)
    if not m:
        raise HTTPException(404, "Không tìm thấy metadata")
    m["dubbed"] = dubbing.is_dubbed(vid)
    return m


def _seg_text(seg: dict, lang: str) -> str:
    """Chữ để HIỂN THỊ:
    - en: text gốc
    - vi: ưu tiên text_vi (sạch), fallback text_tts nếu thiếu.
    """
    if lang == "en":
        return (seg.get("text") or "").replace("\n", " ").strip()
    if seg.get("text_vi"):
        return seg["text_vi"].strip()
    return (seg.get("text_tts") or "").strip()


@router.get("/video/{vid}/subtitles/{lang}")
def subtitles(vid: str, lang: str):
    """Phụ đề WebVTT. lang='en' (gốc) hoặc 'vi' (dịch, bản sạch để đọc)."""
    if lang not in ("en", "vi"):
        raise HTTPException(404, "lang phải là 'en' hoặc 'vi'")
    p = dubbing.subtitles_path(vid)
    if not p.exists():
        raise HTTPException(404, "Không có phụ đề")
    segs = load_json(str(p))
    # gắn 'disp' = chữ hiển thị, rồi dựng VTT theo field đó
    for s in segs:
        s["disp"] = _seg_text(s, lang)
    return Response(
        build_vtt(segs, "disp", prefer_tts_timing=(lang == "vi")),
        media_type="text/vtt",
        headers=NO_CACHE_HEADERS,
    )


@router.get("/video/{vid}/transcript")
def transcript(vid: str):
    """Segments cho bảng phụ đề chạy theo video: [{start, duration, en, vi}]."""
    p = dubbing.subtitles_path(vid)
    if not p.exists():
        raise HTTPException(404, "Không có phụ đề")
    segs = load_json(str(p))
    out = []
    for i, s in enumerate(segs):
        tts = s.get("tts") if isinstance(s.get("tts"), dict) else {}
        tts_engine = tts.get("engine")
        start, end = display_range(segs, i, prefer_tts_timing=True)
        if end <= start:
            end = start + max(s.get("duration", 0), 0.5)
        out.append({
            "index": i,
            "start": start,
            "duration": s["duration"],
            "end": end,
            "en": _seg_text(s, "en"),
            "vi": _seg_text(s, "vi"),
            "tts_text": (s.get("text_tts") or s.get("text_vi") or _seg_text(s, "vi")).strip(),
            "pronunciation_map": s.get("pronunciation_map") or {},
            "tts_engine": tts_engine,
            "num_step": int(tts.get("num_step") or (48 if tts_engine == "omnivoice" else 8)),
            "can_regenerate": tts_engine in {"omnivoice", "supertonic"},
        })
    return out


@router.get("/stream/{vid}")
def stream(vid: str):
    """Phát file mp4 (FileResponse tự hỗ trợ Range → tua/đổi tốc độ)."""
    p = dubbing.video_dub_path(vid)
    if not p.exists():
        raise HTTPException(404, "Video chưa được dub")
    return FileResponse(str(p), media_type="video/mp4", headers=NO_CACHE_HEADERS)
