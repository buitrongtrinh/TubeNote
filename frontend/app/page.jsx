"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { clearDubbingDraftForVideo } from "@/lib/draft";

function fmtCount(n) {
  if (typeof n !== "number") return null;
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(".0", "") + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1).replace(".0", "") + "K";
  return String(n);
}

function fmtDuration(s) {
  if (typeof s !== "number" || s <= 0) return null;
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  const mm = h ? String(m).padStart(2, "0") : String(m);
  return `${h ? h + ":" : ""}${mm}:${String(sec).padStart(2, "0")}`;
}

export default function LibraryPage() {
  const [items, setItems] = useState(null);
  const [err, setErr] = useState(null);
  const [deletingIds, setDeletingIds] = useState({});

  useEffect(() => {
    api.library().then(setItems).catch((e) => setErr(String(e)));
  }, []);

  async function deleteVideo(item) {
    const ok = window.confirm(`Xóa "${item.title}" khỏi thư viện? Thao tác này sẽ xóa cả file video, audio và phụ đề đã tạo.`);
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
          <h1>Thư viện</h1>
          <p>{items?.length ? `${items.length} video đã lồng tiếng` : "Video đã lồng tiếng của bạn"}</p>
        </div>
      </div>

      {err && <div className="tag-fail">{err}</div>}
      {items === null && !err && (
        <div className="empty-state">
          <span className="eq-loader" aria-hidden="true" />
          Đang tải video…
        </div>
      )}

      {items && items.length === 0 && (
        <div className="empty-state">
          <div className="empty-icon">▶</div>
          <h2>Chưa có video nào</h2>
          <p>Dán một link YouTube để tạo bản lồng tiếng đầu tiên.</p>
        </div>
      )}

      {items && items.length > 0 && (
        <div className="video-grid">
          {items.map((it) => {
            const dur = fmtDuration(it.duration);
            const deleting = Boolean(deletingIds[it.video_id]);
            return (
              <div key={it.video_id} className={deleting ? "video-card deleting" : "video-card"}>
                <Link href={`/video/${it.video_id}`} className="video-card-link" aria-label={`Mở ${it.title}`}>
                  <div className="thumb-wrap">
                    {it.thumbnail && <img src={it.thumbnail} alt={it.title} />}
                    {dur && <div className="thumb-dur">{dur}</div>}
                  </div>
                  <div className="video-card-body">
                    <div className="channel-avatar">
                      {it.channel_avatar
                        ? <img src={it.channel_avatar} alt="" loading="lazy" referrerPolicy="no-referrer" />
                        : (it.channel?.slice(0, 1)?.toUpperCase() || "T")}
                    </div>
                    <div className="video-card-text">
                      <div className="video-title">{it.title}</div>
                      <div className="video-meta">{it.channel}</div>
                      <div className="video-meta">
                        {fmtCount(it.view_count) ? `${fmtCount(it.view_count)} lượt xem` : "Video đã xử lý"}
                      </div>
                    </div>
                  </div>
                </Link>
                <button
                  type="button"
                  className="video-delete-button"
                  onClick={() => deleteVideo(it)}
                  disabled={deleting}
                  title="Xóa video"
                  aria-label={`Xóa ${it.title}`}
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
