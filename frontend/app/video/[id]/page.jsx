"use client";

import { useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import ReactMarkdown from "react-markdown";
import { api } from "@/lib/api";
import Transcript from "@/components/Transcript";
import VideoPlayer from "@/components/VideoPlayer";

// Prompt yêu cầu LLM chỉ dùng bold/list, không heading/bảng/code-fence (panel
// chat hẹp, dễ vỡ layout) — chặn thêm ở đây phòng khi model không tuân thủ.
const RAG_MARKDOWN_DISALLOWED = ["h1", "h2", "h3", "h4", "h5", "h6", "table", "pre", "img"];

function RagMarkdown({ text }) {
  return (
    <ReactMarkdown disallowedElements={RAG_MARKDOWN_DISALLOWED} unwrapDisallowed>
      {text}
    </ReactMarkdown>
  );
}

function fmtCount(n) {
  if (typeof n !== "number") return null;
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(".0", "") + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1).replace(".0", "") + "K";
  return String(n);
}

function fmtDate(s) {
  if (!s || String(s).length !== 8) return null;
  const t = String(s);
  return `${t.slice(6, 8)}/${t.slice(4, 6)}/${t.slice(0, 4)}`;
}

function fmtTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const secs = Math.floor(value % 60);
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function fmtDateTime(value) {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString("vi-VN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function fmtSeconds(value) {
  if (value === null || value === undefined || value === "NaN") return null;
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  return `${number.toFixed(1)}s`;
}

function compactJoin(parts, separator = " · ") {
  return parts.filter(Boolean).join(separator);
}

function buildDubbingRows(meta, segments) {
  const dubbing = meta?.dubbing || {};
  const tts = dubbing.tts || {};
  const asr = dubbing.asr || {};
  const translation = dubbing.translation || {};
  const background = dubbing.background || {};
  const timing = dubbing.timing || {};
  const run = dubbing.run || {};
  const fallbackSegment = (segments || []).find((segment) => segment.tts_engine);
  const engine = tts.engine || fallbackSegment?.tts_engine;
  const engineLabel = tts.engine_label || (
    engine === "omnivoice" ? "OmniVoice - GPU" :
    engine === "supertonic" ? "Supertonic - CPU" :
    engine
  );
  const translationLabel = translation.mode === "api"
    ? compactJoin([translation.provider, translation.model])
    : (translation.mode === "manual" ? (translation.model || "ChatGPT") : null);
  const timingLabel = compactJoin([
    timing.wsola_limit !== null && timing.wsola_limit !== undefined ? `WSOLA ${timing.wsola_limit}x` : null,
    timing.generation_duration_delta !== null && timing.generation_duration_delta !== undefined
      ? `delta ${timing.generation_duration_delta}s`
      : null,
    timing.merge_max_chars === 0 ? "không merge" : null,
  ]);
  const rows = [
    ["TTS engine", engineLabel],
    ["Model", tts.model],
    ["Giọng", tts.voice_label || tts.voice_id],
    ["Chất lượng", tts.num_step || fallbackSegment?.num_step ? `${tts.num_step || fallbackSegment?.num_step} steps` : null],
    ["Batch TTS", tts.batch_size ? String(tts.batch_size) : null],
    // Video có phụ đề người đăng tự làm -> Whisper không chạy, hiện đúng
    // nguồn thay vì model ASR đã chọn (nhưng không dùng).
    asr.source === "manual_sub"
      ? ["Phụ đề nguồn", "Phụ đề gốc của video (manual sub)"]
      : ["ASR", compactJoin([asr.preset, asr.engine])],
    ["Dịch", translationLabel],
    ["Nhạc nền", typeof background.enabled === "boolean" ? (background.enabled ? `Bật${background.source ? ` · ${background.source}` : ""}` : "Tắt") : null],
    ["Timing", timingLabel],
    ["Thời gian xử lý", compactJoin([fmtSeconds(run.tts_time_sec) && `TTS ${fmtSeconds(run.tts_time_sec)}`, fmtSeconds(run.total_time_sec) && `Tổng ${fmtSeconds(run.total_time_sec)}`])],
    ["Tạo lúc", fmtDateTime(dubbing.generated_at)],
    ["Run ID", dubbing.run_id || meta?.latest_run_id],
  ];
  return rows.filter(([, value]) => value !== null && value !== undefined && String(value).trim());
}

function cleanError(error) {
  const raw = String(error?.message || error || "");
  try {
    const parsed = JSON.parse(raw);
    return parsed.detail || raw;
  } catch {
    return raw;
  }
}

function pronunciationRows(mapping) {
  const rows = Object.entries(mapping || {}).map(([source, spoken]) => ({ source, spoken }));
  return rows.length ? rows : [{ source: "", spoken: "" }];
}

function parsePronunciationMap(rows) {
  const mapping = {};
  rows.forEach((row, index) => {
    const source = row.source.trim();
    const spoken = row.spoken.trim();
    if (!source && !spoken) return;
    if (!source || !spoken) throw new Error(`Mapping dòng ${index + 1} chưa điền đủ hai cột`);
    mapping[source] = spoken;
  });
  return mapping;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function applyPronunciationPreview(text, rows) {
  let next = text || "";
  rows.forEach((row) => {
    const source = row.source.trim();
    const spoken = row.spoken.trim();
    if (!source || !spoken) return;
    try {
      const pattern = new RegExp(`(?<![\\p{L}\\p{N}_])${escapeRegExp(source)}(?![\\p{L}\\p{N}_])`, "giu");
      next = next.replace(pattern, spoken);
    } catch {
      next = next.split(source).join(spoken);
    }
  });
  return next;
}

function buildRagHistory(messages) {
  const history = [];
  messages
    .filter((message) => !message.pending && !message.error && message.answer)
    .slice(-3)
    .forEach((message) => {
      history.push({ role: "user", content: message.question || "" });
      history.push({ role: "assistant", content: String(message.answer || "").slice(0, 1200) });
    });
  return history.filter((item) => item.content.trim());
}

const REGEN_QUALITY_OPTIONS = {
  supertonic: [
    { value: 5, label: "Nhanh · 5 steps" },
    { value: 8, label: "Cân bằng · 8 steps" },
    { value: 12, label: "Cao · 12 steps" },
  ],
  omnivoice: [
    { value: 16, label: "Nhanh · 16 steps" },
    { value: 24, label: "Cân bằng · 24 steps" },
    { value: 32, label: "Cao · 32 steps" },
    { value: 48, label: "Tối đa · 48 steps" },
  ],
};

const FALLBACK_RAG_MODELS = {
  default_provider: "deepseek",
  providers: [
    {
      id: "deepseek",
      label: "DeepSeek",
      models: ["deepseek-v4-flash", "deepseek-v4-pro"],
      default_model: "deepseek-v4-flash",
    },
  ],
};

const RAG_WEB_MODES = [
  { id: "off", label: "Tắt", hint: "Chỉ dùng nội dung video, không tìm web." },
  { id: "auto", label: "Tự động", hint: "Chỉ tìm web khi nội dung video không đủ trả lời." },
  { id: "always", label: "Luôn tìm", hint: "Luôn tìm web kèm nội dung video cho mỗi câu hỏi." },
];

function defaultRegenStep(engine) {
  return engine === "supertonic" ? 8 : 48;
}

export default function VideoDetail() {
  const { id } = useParams();
  const [meta, setMeta] = useState(null);
  const [err, setErr] = useState(null);
  const [segments, setSegments] = useState([]);
  const [time, setTime] = useState(0);
  const [revision, setRevision] = useState(0);
  const [regenSegment, setRegenSegment] = useState(null);
  const [regenText, setRegenText] = useState("");
  const [regenMappings, setRegenMappings] = useState([{ source: "", spoken: "" }]);
  const [regenNumStep, setRegenNumStep] = useState(48);
  const [regenState, setRegenState] = useState(null);
  const [redubState, setRedubState] = useState(null);
  const [sideTab, setSideTab] = useState("transcript");
  const [ragQuestion, setRagQuestion] = useState("");
  const [ragMessages, setRagMessages] = useState([]);
  const [ragState, setRagState] = useState(null);
  const [ragSummary, setRagSummary] = useState(null);
  const [ragSummaryState, setRagSummaryState] = useState(null);
  const [ragModelConfig, setRagModelConfig] = useState(FALLBACK_RAG_MODELS);
  const [ragProvider, setRagProvider] = useState(FALLBACK_RAG_MODELS.default_provider);
  const [ragModel, setRagModel] = useState(FALLBACK_RAG_MODELS.providers[0].default_model);
  const [ragSessionModel, setRagSessionModel] = useState(null);
  const [ragWebMode, setRagWebMode] = useState("off");
  const playerRef = useRef(null);
  const pendingSeekRef = useRef(null);
  const summaryLoadingRef = useRef(null);

  useEffect(() => {
    api.ragModels()
      .then((config) => {
        const providers = Array.isArray(config?.providers) && config.providers.length
          ? config.providers
          : FALLBACK_RAG_MODELS.providers;
        const nextConfig = {
          default_provider: config?.default_provider || providers[0].id,
          providers,
        };
        const defaultProvider = providers.find((item) => item.id === nextConfig.default_provider) || providers[0];
        setRagModelConfig(nextConfig);
        setRagProvider(defaultProvider.id);
        setRagModel(defaultProvider.default_model || defaultProvider.models?.[0] || "");
      })
      .catch(() => {
        setRagModelConfig(FALLBACK_RAG_MODELS);
      });
  }, []);

  useEffect(() => {
    api.meta(id).then(setMeta).catch((e) => setErr(String(e)));
    api.transcript(id).then(setSegments).catch(() => setSegments([]));
    setRagQuestion("");
    setRagMessages([]);
    setRagState(null);
    setRagSummary(null);
    setRagSummaryState(null);
    setRagSessionModel(null);
    setRedubState(null);
    summaryLoadingRef.current = null;
  }, [id]);

  useEffect(() => {
    if (!revision || pendingSeekRef.current === null) return undefined;
    const timer = setTimeout(() => {
      if (playerRef.current) playerRef.current.currentTime = pendingSeekRef.current;
      pendingSeekRef.current = null;
    }, 150);
    return () => clearTimeout(timer);
  }, [revision]);

  function seek(t) {
    if (playerRef.current) playerRef.current.currentTime = t;
  }

  async function refreshVideo(pendingSeek = null) {
    const [nextMeta, nextSegments] = await Promise.all([
      api.meta(id),
      api.transcript(id),
    ]);
    setMeta(nextMeta);
    setSegments(nextSegments);
    pendingSeekRef.current = pendingSeek;
    setRevision(Date.now());
  }

  function openRegenerate(segment) {
    setRegenSegment(segment);
    setRegenText(segment.vi || "");
    setRegenMappings(pronunciationRows(segment.pronunciation_map));
    setRegenNumStep(segment.num_step || defaultRegenStep(segment.tts_engine));
    setRegenState(null);
  }

  function selectedRagProvider() {
    return ragModelConfig.providers.find((provider) => provider.id === ragProvider)
      || ragModelConfig.providers[0]
      || FALLBACK_RAG_MODELS.providers[0];
  }

  function changeRagProvider(providerId) {
    if (ragSessionModel || ragState?.status === "running" || ragSummaryState?.status === "running") return;
    const provider = ragModelConfig.providers.find((item) => item.id === providerId)
      || ragModelConfig.providers[0]
      || FALLBACK_RAG_MODELS.providers[0];
    setRagProvider(provider.id);
    setRagModel(provider.default_model || provider.models?.[0] || "");
    setRagSummary(null);
    setRagSummaryState(null);
    summaryLoadingRef.current = null;
  }

  function resetRagSession() {
    if (ragState?.status === "running" || ragSummaryState?.status === "running") return;
    setRagQuestion("");
    setRagMessages([]);
    setRagState(null);
    setRagSummary(null);
    setRagSummaryState(null);
    setRagSessionModel(null);
    summaryLoadingRef.current = null;
  }

  async function startRagSession() {
    if (!ragProvider || !ragModel || ragState?.status === "running" || ragSummaryState?.status === "running") return;
    const session = { provider: ragProvider, model: ragModel };
    const summaryKey = `${id}:${session.provider}:${session.model}`;
    summaryLoadingRef.current = summaryKey;
    setRagQuestion("");
    setRagMessages([]);
    setRagState(null);
    setRagSummary(null);
    setRagSessionModel(null);
    setRagSummaryState({ status: "running" });
    try {
      const result = await api.ragSummary(id, session.provider, session.model);
      if (summaryLoadingRef.current !== summaryKey) return;
      setRagSummary(result);
      setRagSessionModel(session);
      setRagSummaryState(null);
      summaryLoadingRef.current = null;
    } catch (error) {
      if (summaryLoadingRef.current !== summaryKey) return;
      setRagSummaryState({ status: "error", error: cleanError(error) });
      summaryLoadingRef.current = null;
    }
  }

  async function askRag(event) {
    event.preventDefault();
    const question = ragQuestion.trim();
    if (!question || !ragSessionModel || !ragSummary?.summary || ragState?.status === "running" || ragSummaryState?.status === "running") return;
    const messageId = `${Date.now()}`;
    const history = buildRagHistory(ragMessages);
    setRagQuestion("");
    setRagMessages((current) => [
      ...current,
      {
        id: messageId,
        question,
        answer: "",
        sources: [],
        webSources: [],
        pending: true,
      },
    ]);
    setRagState({ status: "running" });
    try {
      const result = await api.askRag(
        id,
        question,
        history,
        ragSummary.summary,
        ragSessionModel.provider,
        ragSessionModel.model,
        ragWebMode,
      );
      setRagMessages((current) => current.map((message) => (
        message.id === messageId
          ? {
              ...message,
              answer: result.answer || "",
              sources: result.sources || [],
              webSources: result.web_sources || [],
              pending: false,
            }
          : message
      )));
      setRagState(null);
    } catch (error) {
      setRagMessages((current) => current.map((message) => (
        message.id === messageId
          ? {
              ...message,
              answer: "",
              sources: [],
              webSources: [],
              error: cleanError(error),
              pending: false,
            }
          : message
      )));
      setRagState(null);
    }
  }

  function handleRagKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  async function regenerate() {
    if (!regenSegment || !regenText.trim() || regenState?.status === "running") return;
    try {
      const pronunciationMap = parsePronunciationMap(regenMappings);
      const { job_id } = await api.regenerateSegment(
        id,
        regenSegment.index,
        regenText.trim(),
        pronunciationMap,
        regenNumStep,
      );
      setRegenState({ status: "running", progress: 0, stage: "Bắt đầu" });
      while (true) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        const job = await api.regenerateStatus(job_id);
        setRegenState(job);
        if (job.status === "error") throw new Error(job.error || "Không thể tạo lại đoạn");
        if (job.status === "done") break;
      }
      await refreshVideo(regenSegment.start);
      setRegenSegment(null);
      setRegenState(null);
    } catch (error) {
      setRegenState({ status: "error", error: String(error.message || error) });
    }
  }

  async function regenerateFullDubbing() {
    if (redubState?.status === "running" || regenState?.status === "running") return;
    try {
      const { job_id } = await api.regenerateVideo(id);
      setRedubState({ status: "running", progress: 0, stage: "Bắt đầu" });
      const currentTime = time;
      while (true) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        const job = await api.dubStatus(job_id);
        setRedubState(job);
        if (job.status === "error") throw new Error(job.error || "Không thể tạo lại toàn bộ audio");
        if (job.status === "done") break;
      }
      await refreshVideo(currentTime);
      setRedubState(null);
    } catch (error) {
      setRedubState({ status: "error", error: cleanError(error) });
    }
  }

  if (err) return <main className="page-content"><div className="tag-fail">{err}</div></main>;
  if (!meta) return (
    <main className="page-content">
      <div className="empty-state">
        <span className="eq-loader" aria-hidden="true" />
        Đang tải video…
      </div>
    </main>
  );

  const stats = [
    meta.channel,
    fmtCount(meta.view_count) && `${fmtCount(meta.view_count)} lượt xem`,
    fmtCount(meta.like_count) && `${fmtCount(meta.like_count)} thích`,
    fmtDate(meta.upload_date),
  ].filter(Boolean).join("  ·  ");
  const regenQualityOptions = REGEN_QUALITY_OPTIONS[regenSegment?.tts_engine] || REGEN_QUALITY_OPTIONS.omnivoice;
  const regenTtsText = applyPronunciationPreview(regenText, regenMappings);
  const currentRagProvider = selectedRagProvider();
  const currentRagModels = currentRagProvider?.models || [];
  const ragBusy = ragState?.status === "running" || ragSummaryState?.status === "running";
  const ragReady = Boolean(ragSessionModel && ragSummary?.summary);
  const ragSelectionLocked = ragBusy || Boolean(ragSessionModel);
  const dubbingRows = buildDubbingRows(meta, segments);
  const ragActionLabel = ragSessionModel
    ? "Đổi model"
    : (ragSummaryState?.status === "error" ? "Thử lại" : "Bắt đầu hỏi đáp");
  const redubRunning = redubState?.status === "running";
  const hasTranslatedChapters = Array.isArray(meta.chapters) && meta.chapters.length > 0 && (
    meta.chapters.every((chapter) => String(chapter?.title_vi || "").trim())
  );

  return (
    <main className="page-content watch-content">
      <div className="watch-grid">
        <section className="watch-main">
          <VideoPlayer
            key={revision}
            src={`${api.streamUrl(id)}?v=${revision}`}
            title={meta.title}
            poster={meta.thumbnail}
            playerRef={playerRef}
            onTime={setTime}
            subtitles={{
              vi: `${api.subtitleUrl(id, "vi")}?v=${revision}`,
              en: `${api.subtitleUrl(id, "en")}?v=${revision}`,
            }}
            chapters={hasTranslatedChapters ? `${api.chapterUrl(id)}?v=${revision}` : null}
          />

          <h1 className="watch-title">{meta.title}</h1>
          <div className="watch-meta-row">
            <div className="channel-block">
              <div className="channel-avatar large">
                {meta.channel_avatar
                  ? <img src={meta.channel_avatar} alt="" referrerPolicy="no-referrer" />
                  : (meta.channel?.slice(0, 1)?.toUpperCase() || "T")}
              </div>
              <div>
                <div className="channel-name">{meta.channel}</div>
                <div className="video-meta">{stats}</div>
              </div>
            </div>
            <div className="action-row">
              <button>Thích</button>
              <a href={api.streamUrl(id)} download={`${id}_dubbed.mp4`}>
                <button>Tải MP4</button>
              </a>
            </div>
          </div>

          {dubbingRows.length > 0 && (
            <section className="dubbing-info-box" aria-label="Thông tin lồng tiếng">
              <div className="dubbing-info-head">
                <h2>Thông tin lồng tiếng</h2>
                <div className="dubbing-info-actions">
                  <span>{meta.video_id || id}</span>
                  <button
                    type="button"
                    disabled={redubRunning || regenState?.status === "running"}
                    onClick={regenerateFullDubbing}
                  >
                    {redubRunning ? "Đang tạo lại..." : "Tạo lại toàn bộ audio"}
                  </button>
                </div>
              </div>
              {redubRunning && (
                <div className="dub-progress">
                  <div className="dub-progress-bar">
                    <div className="dub-progress-fill" style={{ width: `${redubState.progress || 0}%` }} />
                  </div>
                  <div className="dub-progress-label">
                    <span className="dub-progress-stage">
                      <span className="eq-loader" aria-hidden="true" />
                      {redubState.stage}
                    </span>
                    <span>{redubState.progress || 0}%</span>
                  </div>
                </div>
              )}
              {redubState?.status === "error" && (
                <div className="tag-fail regen-error">{redubState.error}</div>
              )}
              <div className="dubbing-info-grid">
                {dubbingRows.map(([label, value]) => (
                  <div className="dubbing-info-row" key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                  </div>
                ))}
              </div>
            </section>
          )}

        </section>

        <aside className="watch-side" aria-label="Khu vực tính năng bổ sung">
          <div className="side-tabs" role="tablist" aria-label="Công cụ video">
            <button
              type="button"
              className={sideTab === "transcript" ? "active" : ""}
              onClick={() => setSideTab("transcript")}
              role="tab"
              aria-selected={sideTab === "transcript"}
            >
              Phụ đề
            </button>
            <button
              type="button"
              className={sideTab === "rag" ? "active" : ""}
              onClick={() => setSideTab("rag")}
              role="tab"
              aria-selected={sideTab === "rag"}
            >
              Hỏi đáp
            </button>
          </div>
          <div className="side-panel">
            {sideTab === "transcript" && (
              segments.length > 0 ? (
                <Transcript
                  segments={segments}
                  time={time}
                  onSeek={seek}
                  onRegenerate={openRegenerate}
                  regenerating={regenState?.status === "running"}
                />
              ) : (
                <div className="empty-state compact">Chưa có transcript.</div>
              )
            )}
            {sideTab === "rag" && (
              <div className="rag-panel">
                <div className="rag-header">
                  <div className="rag-header-title">
                    <span>Hỏi đáp</span>
                    <strong>{meta.title}</strong>
                  </div>
                  <div className="rag-model-controls" aria-label="Chọn mô hình hỏi đáp">
                    <label className="rag-model-field">
                      <span>Provider</span>
                      <select
                        value={ragProvider}
                        disabled={ragSelectionLocked}
                        onChange={(event) => changeRagProvider(event.target.value)}
                      >
                        {ragModelConfig.providers.map((provider) => (
                          <option key={provider.id} value={provider.id}>
                            {provider.label || provider.id}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="rag-model-field">
                      <span>Model</span>
                      <select
                        value={ragModel}
                        disabled={ragSelectionLocked || currentRagModels.length <= 1}
                        onChange={(event) => {
                          setRagModel(event.target.value);
                          setRagSummary(null);
                          setRagSummaryState(null);
                          summaryLoadingRef.current = null;
                        }}
                      >
                        {currentRagModels.map((model) => (
                          <option key={model} value={model}>
                            {model}
                          </option>
                        ))}
                      </select>
                    </label>
                    <button
                      type="button"
                      className="rag-model-action"
                      disabled={ragBusy || !ragProvider || !ragModel}
                      onClick={ragSessionModel ? resetRagSession : startRagSession}
                    >
                      {ragActionLabel}
                    </button>
                  </div>
                </div>
                <div className="rag-thread">
                  {ragSummaryState?.status === "running" && !ragSummary && (
                    <div className="rag-summary-card rag-summary-loading">
                      <span />
                      <p>Đang tạo tóm tắt video...</p>
                    </div>
                  )}
                  {ragSummary?.summary && (
                    <article className="rag-summary-card">
                      <div className="rag-summary-title">
                        <span>Tóm tắt video</span>
                        {ragSessionModel && <em>{ragSessionModel.provider} · {ragSessionModel.model}</em>}
                      </div>
                      <RagMarkdown text={ragSummary.summary} />
                    </article>
                  )}
                  {ragSummaryState?.status === "error" && (
                    <div className="tag-fail rag-error">{ragSummaryState.error}</div>
                  )}
                  {ragMessages.length === 0 && !ragSummary && ragSummaryState?.status !== "running" && (
                    <div className="rag-empty">
                      <div className="rag-empty-icon">?</div>
                      <strong>Bắt đầu hỏi đáp</strong>
                      <span>Chọn model rồi bấm Bắt đầu hỏi đáp để tạo tóm tắt nền cho video.</span>
                    </div>
                  )}
                  {ragMessages.map((message) => (
                    <article className="rag-message" key={message.id}>
                      <div className="rag-question">
                        <p>{message.question}</p>
                      </div>
                      {message.pending ? null : message.error ? (
                        <div className="tag-fail rag-error">{message.error}</div>
                      ) : (
                        <>
                          <div className="rag-answer">
                            <RagMarkdown text={message.answer} />
                          </div>
                          {message.sources.length > 0 && (
                            <div className="rag-sources">
                              <div className="rag-sources-title">Nguồn trong video</div>
                              {message.sources.slice(0, 5).map((source, index) => (
                                <button
                                  type="button"
                                  key={`${message.id}-${index}`}
                                  className="rag-source"
                                  onClick={() => seek(source.start || 0)}
                                >
                                  <span className="rag-source-time">{fmtTime(source.start)} - {fmtTime(source.end)}</span>
                                  <small>{source.text}</small>
                                </button>
                              ))}
                            </div>
                          )}
                          {message.webSources?.length > 0 && (
                            <div className="rag-sources">
                              <div className="rag-sources-title">Nguồn web</div>
                              {message.webSources.map((source, index) => (
                                <a
                                  key={`${message.id}-web-${index}`}
                                  className="rag-source"
                                  href={source.url}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  <span className="rag-source-time">{source.domain}</span>
                                  <small>{source.title || source.snippet}</small>
                                </a>
                              ))}
                            </div>
                          )}
                        </>
                      )}
                    </article>
                  ))}
                  {ragState?.status === "running" && (
                    <div className="rag-loading">
                      <span />
                      <p>Đang trả lời...</p>
                    </div>
                  )}
                </div>
                <div className="rag-web-mode" role="group" aria-label="Chế độ tìm web">
                  {RAG_WEB_MODES.map((mode) => (
                    <button
                      key={mode.id}
                      type="button"
                      className={ragWebMode === mode.id ? "active" : ""}
                      disabled={ragBusy}
                      onClick={() => setRagWebMode(mode.id)}
                      title={mode.hint}
                    >
                      {mode.label}
                    </button>
                  ))}
                </div>
                <form className="rag-input-row" onSubmit={askRag}>
                  <textarea
                    rows={1}
                    value={ragQuestion}
                    placeholder={ragSummaryState?.status === "running" ? "Đang tạo tóm tắt trước khi hỏi..." : (ragReady ? "Đặt câu hỏi về video..." : "Bấm Bắt đầu hỏi đáp trước khi đặt câu hỏi...")}
                    disabled={ragBusy || !ragReady}
                    onChange={(event) => setRagQuestion(event.target.value)}
                    onKeyDown={handleRagKeyDown}
                  />
                  <button
                    type="submit"
                    disabled={!ragQuestion.trim() || ragBusy || !ragReady}
                  >
                    Gửi
                  </button>
                </form>
              </div>
            )}
          </div>
        </aside>
      </div>
      {regenSegment && (
        <div className="modal-overlay" onClick={() => regenState?.status !== "running" && setRegenSegment(null)}>
          <div className="modal regen-modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-head">
              <span>Tạo lại đoạn {regenSegment.index + 1}</span>
              <button
                className="modal-x"
                title="Đóng"
                disabled={regenState?.status === "running"}
                onClick={() => setRegenSegment(null)}
              >x</button>
            </div>
            <label className="regen-field">
              <span>Nội dung tiếng Việt</span>
              <textarea
                rows={4}
                value={regenText}
                disabled={regenState?.status === "running"}
                onChange={(event) => setRegenText(event.target.value)}
              />
            </label>
            <label className="regen-field">
              <span>Chất lượng tạo lại</span>
              <select
                value={regenNumStep}
                disabled={regenState?.status === "running"}
                onChange={(event) => setRegenNumStep(Number(event.target.value))}
              >
                {regenQualityOptions.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <div className="regen-field">
              <span>Mapping phát âm</span>
              <div className="pronunciation-map-head" aria-hidden="true">
                <span>Từ gốc</span>
                <span>Cách đọc</span>
              </div>
              <div className="pronunciation-map-rows">
                {regenMappings.map((row, index) => (
                  <div className="pronunciation-map-row" key={index}>
                    <input
                      type="text"
                      value={row.source}
                      disabled={regenState?.status === "running"}
                      placeholder="RAG"
                      aria-label={`Từ gốc ${index + 1}`}
                      onChange={(event) => setRegenMappings((current) => current.map((item, itemIndex) => (
                        itemIndex === index ? { ...item, source: event.target.value } : item
                      )))}
                    />
                    <input
                      type="text"
                      value={row.spoken}
                      disabled={regenState?.status === "running"}
                      placeholder="Rác"
                      aria-label={`Cách đọc ${index + 1}`}
                      onChange={(event) => setRegenMappings((current) => current.map((item, itemIndex) => (
                        itemIndex === index ? { ...item, spoken: event.target.value } : item
                      )))}
                    />
                    <button
                      type="button"
                      className="pronunciation-map-remove"
                      title="Xóa mapping"
                      aria-label={`Xóa mapping ${index + 1}`}
                      disabled={regenState?.status === "running"}
                      onClick={() => setRegenMappings((current) => (
                        current.length === 1
                          ? [{ source: "", spoken: "" }]
                          : current.filter((_, itemIndex) => itemIndex !== index)
                      ))}
                    >x</button>
                  </div>
                ))}
              </div>
              <button
                type="button"
                className="pronunciation-map-add"
                title="Thêm mapping"
                aria-label="Thêm mapping phát âm"
                disabled={regenState?.status === "running"}
                onClick={() => setRegenMappings((current) => [...current, { source: "", spoken: "" }])}
              >+</button>
            </div>
            <details className="regen-tts-preview">
              <summary>Văn bản gửi vào TTS</summary>
              <div className="regen-tts-preview-body">
                <div>
                  <span>Đang lưu</span>
                  <p>{regenSegment.tts_text || regenSegment.vi || ""}</p>
                </div>
                <div>
                  <span>Sẽ gửi khi tạo lại</span>
                  <code>{regenTtsText}</code>
                </div>
              </div>
            </details>
            {regenState?.status === "running" && (
              <div className="dub-progress">
                <div className="dub-progress-bar">
                  <div className="dub-progress-fill" style={{ width: `${regenState.progress || 0}%` }} />
                </div>
                <div className="dub-progress-label">
                  <span className="dub-progress-stage">
                    <span className="eq-loader" aria-hidden="true" />
                    {regenState.stage}
                  </span>
                  <span>{regenState.progress || 0}%</span>
                </div>
              </div>
            )}
            {regenState?.status === "error" && <div className="tag-fail regen-error">{regenState.error}</div>}
            <div className="modal-foot">
              <button disabled={regenState?.status === "running"} onClick={() => setRegenSegment(null)}>Hủy</button>
              <button
                className="primary"
                disabled={!regenText.trim() || regenState?.status === "running"}
                onClick={regenerate}
              >{regenState?.status === "running" ? "Đang tạo" : "Tạo lại giọng đọc"}</button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
