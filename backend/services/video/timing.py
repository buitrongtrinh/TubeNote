"""Timing helpers for subtitle and transcript playback.

New dubbed videos store generated timing under:

    segment["tts"]      -> TTS/audio-generation timeline
    segment["playback"] -> final output-video timeline

Legacy videos used flat fields such as ``tts_start`` and
``tts_playback_start``.  Keep all reads backward compatible here so API routes
do not need to know both schemas.
"""
from __future__ import annotations


LEGACY_TIMING_KEYS = (
    "tts_start",
    "tts_end",
    "tts_duration",
    "output_speed",
    "playback_start",
    "playback_end",
    "playback_duration",
    "tts_playback_start",
    "tts_playback_end",
    "tts_playback_duration",
)


def clear_generated_timing(seg: dict) -> None:
    """Remove generated timing metadata before a fresh dubbing run."""
    seg.pop("tts", None)
    seg.pop("playback", None)
    for key in LEGACY_TIMING_KEYS:
        seg.pop(key, None)


def source_range(segments: list[dict], index: int) -> tuple[float, float]:
    """Return the original transcript range for a segment."""
    seg = segments[index]
    start = float(seg.get("start", 0.0))
    if index + 1 < len(segments):
        end = float(segments[index + 1].get("start", start + float(seg.get("duration", 0.0))))
    else:
        end = start + float(seg.get("duration", 0.0))
    if end <= start:
        end = start + max(float(seg.get("duration", 0.0)), 0.5)
    return start, end


def tts_range(seg: dict, playback: bool = False) -> tuple[float, float] | None:
    """Return TTS timing for source or output-video timeline.

    ``playback=True`` returns final-video time after retiming, when available.
    """
    if playback:
        pb = seg.get("playback")
        if isinstance(pb, dict) and pb.get("tts_start") is not None and pb.get("tts_end") is not None:
            return float(pb["tts_start"]), float(pb["tts_end"])
        if seg.get("tts_playback_start") is not None and seg.get("tts_playback_end") is not None:
            return float(seg["tts_playback_start"]), float(seg["tts_playback_end"])

    tts = seg.get("tts")
    if isinstance(tts, dict) and tts.get("start") is not None and tts.get("end") is not None:
        return float(tts["start"]), float(tts["end"])
    if seg.get("tts_start") is not None and seg.get("tts_end") is not None:
        return float(seg["tts_start"]), float(seg["tts_end"])
    return None


def playback_range(segments: list[dict], index: int) -> tuple[float, float] | None:
    """Return output-video segment timing, not necessarily TTS-active timing."""
    seg = segments[index]
    pb = seg.get("playback")
    if isinstance(pb, dict) and pb.get("start") is not None and pb.get("end") is not None:
        return float(pb["start"]), float(pb["end"])
    if seg.get("playback_start") is not None and seg.get("playback_end") is not None:
        return float(seg["playback_start"]), float(seg["playback_end"])
    return None


def display_range(
    segments: list[dict],
    index: int,
    prefer_tts_timing: bool = False,
) -> tuple[float, float]:
    """Return the timing range to use in player-facing APIs."""
    seg = segments[index]

    if prefer_tts_timing:
        rng = tts_range(seg, playback=True) or tts_range(seg, playback=False)
        if rng is not None:
            start, end = rng
            if end > start:
                return start, end

    rng = playback_range(segments, index)
    if rng is not None:
        start, end = rng
        if end > start:
            return start, end

    return source_range(segments, index)
