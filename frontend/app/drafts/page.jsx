"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { clearDubbingDraftForVideo } from "@/lib/draft";

function fmtDuration(s) {
  if (typeof s !== "number" || s <= 0) return null;
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  const mm = h ? String(m).padStart(2, "0") : String(m);
  return `${h ? h + ":" : ""}${mm}:${String(sec).padStart(2, "0")}`;
}

function draftHref(item) {
  const url = item.webpage_url || `https://www.youtube.com/watch?v=${item.video_id}`;
  return `/add?url=${encodeURIComponent(url)}&autoload=1`;
}

export default function DraftsPage() {
  const [items, setItems] = useState(null);
  const [err, setErr] = useState(null);
  const [deletingIds, setDeletingIds] = useState({});

  useEffect(() => {
    api.drafts().then(setItems).catch((e) => setErr(String(e)));
  }, []);

  async function deleteDraft(item) {
    const ok = window.confirm(`Xóa bản nháp "${item.title}"? Thao tác này sẽ xóa metadata, audio/video cache và phụ đề đã load.`);
    if (!ok) return;
    setErr(null);
    setDeletingIds((prev) => ({ ...prev, [item.video_id]: true }));
    try {
      await api.deleteVideo(item.video_id);
      clearDubbingDraftForVideo(item.video_id);
      setItems((prev) => (prev || []).filter((video) => video.video_id !== item.video_id));
    } catch (e) {
      setErr(String(e));
    } finally {
      setDeletingIds((prev) => {
        const next = { ...prev };
        delete next[item.video_id];
        return next;
      });
    }
  }

  return (
    <main className="page-content">
      <div className="feed-header">
        <div>
          <h1>Bản nháp</h1>
          <p>{items?.length ? `${items.length} video đã load, chưa dubbing` : "Video đã load nhưng chưa xuất lồng tiếng"}</p>
        </div>
      </div>

      {err && <div className="tag-fail">{err}</div>}
      {items === null && !err && (
        <div className="empty-state">
          <span className="eq-loader" aria-hidden="true" />
          Đang tải bản nháp…
        </div>
      )}

      {items && items.length === 0 && (
        <div className="empty-state">
          <div className="empty-icon">▶</div>
          <h2>Chưa có bản nháp nào</h2>
          <p>Video sau khi nhấn Load và chưa dubbing sẽ xuất hiện ở đây.</p>
        </div>
      )}

      {items && items.length > 0 && (
        <div className="video-grid">
          {items.map((it) => {
            const dur = fmtDuration(it.duration);
            const deleting = Boolean(deletingIds[it.video_id]);
            return (
              <div key={it.video_id} className={deleting ? "video-card deleting" : "video-card"}>
                <Link href={draftHref(it)} className="video-card-link" aria-label={`Mở bản nháp ${it.title}`}>
                  <div className="thumb-wrap">
                    {it.thumbnail ? (
                      <img src={it.thumbnail} alt={it.title} />
                    ) : (
                      <div className="draft-thumb-placeholder">Draft</div>
                    )}
                    {dur && <div className="thumb-dur">{dur}</div>}
                    <div className="draft-badge">Bản nháp</div>
                  </div>
                  <div className="video-card-body">
                    <div className="channel-avatar">
                      {it.channel_avatar
                        ? <img src={it.channel_avatar} alt="" loading="lazy" referrerPolicy="no-referrer" />
                        : (it.channel?.slice(0, 1)?.toUpperCase() || "D")}
                    </div>
                    <div className="video-card-text">
                      <div className="video-title">{it.title}</div>
                      <div className="video-meta">{it.channel}</div>
                      <div className="video-meta">Đã load transcript, chưa dubbing</div>
                    </div>
                  </div>
                </Link>
                <button
                  type="button"
                  className="video-delete-button"
                  onClick={() => deleteDraft(it)}
                  disabled={deleting}
                  title="Xóa bản nháp"
                  aria-label={`Xóa bản nháp ${it.title}`}
                >
                  {deleting ? "Đang xóa" : "Xóa"}
                </button>
              </div>
            );
          })}
        </div>
      )}
    </main>
  );
}
