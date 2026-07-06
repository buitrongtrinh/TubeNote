"use client";

import { useEffect, useMemo, useRef, useState } from "react";

/**
 * Bảng phụ đề chạy theo video — hiện cả câu gốc (EN) và câu dịch (VI),
 * sáng dòng đang phát, tự cuộn tới, click để tua.
 *
 * Props:
 * - segments: [{start, duration, end, en, vi}]
 * - time: thời gian hiện tại của video (giây)
 * - onSeek: (start) => tua video tới giây đó
 */
export default function Transcript({ segments, time, onSeek, onRegenerate, regenerating }) {
  const [show, setShow] = useState({ en: true, vi: true });
  const bodyRef = useRef(null);
  const activeRef = useRef(null);

  // Tìm dòng đang phát: start <= time < end. Nếu backend cũ chưa có end,
  // fallback về start + duration.
  const activeIdx = useMemo(() => {
    let idx = -1;
    for (let i = 0; i < segments.length; i++) {
      const end = segments[i].end ?? (segments[i].start + segments[i].duration);
      if (segments[i].start <= time && time < end) idx = i;
      else if (segments[i].start <= time) idx = i;
      else break;
    }
    return idx;
  }, [segments, time]);

  // Cuộn dòng đang phát vào giữa — CHỈ trong khung transcript, KHÔNG cuộn cả trang
  // (tránh đẩy video ra ngoài tầm nhìn).
  useEffect(() => {
    const body = bodyRef.current, el = activeRef.current;
    if (!body || !el) return;
    const bodyRect = body.getBoundingClientRect();
    const elRect = el.getBoundingClientRect();
    const offset = (elRect.top - bodyRect.top) - (body.clientHeight / 2 - el.clientHeight / 2);
    body.scrollBy({ top: offset, behavior: "smooth" });
  }, [activeIdx]);

  return (
    <div className="transcript">
      <div className="transcript-head">
        <span>Phụ đề</span>
        <label><input type="checkbox" checked={show.en}
          onChange={(e) => setShow((s) => ({ ...s, en: e.target.checked }))} /> Gốc</label>
        <label><input type="checkbox" checked={show.vi}
          onChange={(e) => setShow((s) => ({ ...s, vi: e.target.checked }))} /> Dịch</label>
      </div>
      <div className="transcript-body" ref={bodyRef}>
        {segments.map((seg, i) => (
          <div
            key={i}
            ref={i === activeIdx ? activeRef : null}
            className={"tline" + (i === activeIdx ? " active" : "") + (seg.can_regenerate ? " has-action" : "")}
            onClick={() => onSeek?.(seg.start)}
          >
            <span className="tline-time">{fmtTime(seg.start)}</span>
            <span className="tline-text">
              {show.en && seg.en && <span className="t-en">{seg.en}</span>}
              {show.vi && seg.vi && <span className="t-vi">{seg.vi}</span>}
            </span>
            {seg.can_regenerate && (
              <button
                type="button"
                className="tline-regenerate"
                title="Tạo lại giọng đọc cho đoạn này"
                disabled={regenerating}
                onClick={(event) => {
                  event.stopPropagation();
                  onRegenerate?.(seg);
                }}
              >Tạo lại</button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function fmtTime(t) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
