export const DUBBING_DRAFT_KEY = "tubenote:dubbing-draft";

export function clearDubbingDraftForVideo(videoId) {
  if (typeof window === "undefined" || !videoId) return;
  try {
    const saved = JSON.parse(localStorage.getItem(DUBBING_DRAFT_KEY) || "null");
    const savedVideoId = saved?.meta?.video_id;
    const savedUrl = String(saved?.url || "");
    if (savedVideoId === videoId || savedUrl.includes(videoId)) {
      localStorage.removeItem(DUBBING_DRAFT_KEY);
    }
  } catch {
    localStorage.removeItem(DUBBING_DRAFT_KEY);
  }
}
