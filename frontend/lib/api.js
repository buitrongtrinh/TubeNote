// Lớp gọi API tới FastAPI. Dùng đường dẫn tương đối "/api" — đã proxy ở
// next.config.mjs nên không lo CORS lúc dev.

async function jget(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function jpost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function jdelete(path) {
  const r = await fetch(path, { method: "DELETE" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export const api = {
  hardware: () => jget("/api/hardware"),
  hardwareRecommend: (ramGb = 0, vramGb = 0) =>
    jget(`/api/hardware/recommend?ram_gb=${encodeURIComponent(ramGb)}&vram_gb=${encodeURIComponent(vramGb)}`),
  library: () => jget("/api/library"),
  drafts: () => jget("/api/drafts"),
  deleteVideo: (vid) => jdelete(`/api/video/${encodeURIComponent(vid)}`),
  meta: (vid) => jget(`/api/video/${vid}/meta`),
  load: (url, engine = "supertonic", speech_preset = "cpu", manual_batch_size = null, api_batch_size = null, sentence_split_mode = null) =>
    jpost("/api/load", { url, engine, speech_preset, manual_batch_size, api_batch_size, sentence_split_mode }),
  loadStatus: (jobId) => jget(`/api/load/${jobId}`),
  validate: (prompt_index, response, expected = 0, engine = "supertonic", budgets = []) =>
    jpost("/api/validate", { prompt_index, response, expected, engine, budgets }),
  translationModels: () => jget("/api/translation/models"),
  translatePrompt: (prompt_index, prompt, provider = "deepseek", model = "deepseek-v4-flash") =>
    jpost("/api/translate", { prompt_index, prompt, provider, model }),
  translateStatus: (jobId) => jget(`/api/translate/${jobId}`),
  ttsModels: () => jget("/api/tts/models"),
  dub: (url, segments, tts, chapter_titles = null) => jpost("/api/dub", {
    url, segments, tts, chapter_titles,
  }),
  dubStatus: (jobId) => jget(`/api/dub/${jobId}`),
  cancelDub: (jobId) => jpost(`/api/dub/${jobId}/cancel`, {}),
  cancelLoad: (jobId) => jpost(`/api/load/${jobId}/cancel`, {}),
  regenerateSegment: (vid, segmentIndex, text_vi, pronunciation_map = {}, num_step = 48) =>
    jpost(`/api/video/${vid}/segments/${segmentIndex}/regenerate`, {
      text_vi, pronunciation_map, num_step,
    }),
  regenerateVideo: (vid) => jpost(`/api/video/${encodeURIComponent(vid)}/regenerate`, {}),
  regenerateStatus: (jobId) => jget(`/api/regenerate/${jobId}`),
  ragModels: () => jget("/api/rag/models"),
  askRag: (vid, question, history = [], summary = "", provider = "deepseek", model = "deepseek-v4-flash", webMode = "off") =>
    jpost(`/api/rag/video/${encodeURIComponent(vid)}/ask`, { question, history, summary, provider, model, web_mode: webMode }),
  ragSummary: (vid, provider = "deepseek", model = "deepseek-v4-flash") =>
    jget(`/api/rag/video/${encodeURIComponent(vid)}/summary?provider=${encodeURIComponent(provider)}&model=${encodeURIComponent(model)}`),
  streamUrl: (vid) => `/api/stream/${vid}`,
  transcript: (vid) => jget(`/api/video/${vid}/transcript`),
  subtitleUrl: (vid, lang) => `/api/video/${vid}/subtitles/${lang}`,
  chapterUrl: (vid) => `/api/video/${vid}/chapters`,
  chapterTranslationPrompt: (vid) => jget(`/api/video/${encodeURIComponent(vid)}/chapters/prompt`),
  validateChapters: (vid, response) => jpost(`/api/video/${encodeURIComponent(vid)}/chapters/validate`, { response }),
};
