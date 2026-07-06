import re
import os
import inspect
import json

_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/|youtube\.com/live/)"
    r"([A-Za-z0-9_-]{11})"
)

def extract_video_id(url_or_id: str) -> str:
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    m = _VIDEO_ID_RE.search(s)
    if m:
        return m.group(1)
    raise ValueError(f"Không lấy được video ID từ: {url_or_id!r}")

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def skip_if_exists(func):
    sig = inspect.signature(func)
    defaults = {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    def wrapper(*args, **kwargs):
        output_path = kwargs.get("output_path")

        if output_path is None:
            url = kwargs.get("url") or args[0]
            output_dir = kwargs.get("output_dir") or defaults.get("output_dir", "data")
            ext = kwargs.get("ext") or defaults.get("ext", "m4a")
            video_id = extract_video_id(url)  # dùng thẳng, không import lại
            output_path = os.path.join(output_dir, f"{video_id}.{ext}")

        if os.path.exists(output_path):
            # Silent skip — nếu cần debug, set env TUBENOTE_VERBOSE=1
            if os.getenv("TUBENOTE_VERBOSE"):
                print(f"⏭️  Đã tồn tại, bỏ qua: {output_path}")
            return output_path

        return func(*args, **kwargs)
    return wrapper