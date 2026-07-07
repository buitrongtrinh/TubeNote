"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

const DRAFT_KEY = "tubenote:dubbing-draft";
const DEFAULT_TTS_CONFIG = {
  engine: "supertonic",
  model: "M5",
  device: "auto",
  voice_preset_id: "",
  voice_mode: "default",
  voice_id: "",
  reference_audio_id: "",
  reference_text: "",
  instruction: "",
  instruction_tags: [],
  num_step: 8,
  keep_background: true,
  batch_size: 0, // 0 = auto theo VRAM detect; >0 = từ VRAM người dùng nhập
};

const TTS_QUALITY_OPTIONS = {
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

const OMNIVOICE_BATCH_OPTIONS = [0, 1, 2, 3, 4, 5, 6, 8, 12, 16];

const TRANSLATION_PROVIDERS = [
  { id: "chatgpt", label: "ChatGPT", url: "https://chatgpt.com/" },
];

const FALLBACK_TRANSLATION_MODELS = {
  default_provider: "deepseek",
  providers: [
    {
      id: "deepseek",
      label: "DeepSeek",
      models: ["deepseek-v4-flash"],
      default_model: "deepseek-v4-flash",
    },
  ],
};

const API_RETRY_MIN_BATCH_SIZE = 5;
const API_RETRY_MAX_DEPTH = 3;
const DEFAULT_API_CONCURRENCY = 8;
const API_STATUS_TIMEOUT_MS = 15000;
const MIN_API_JOB_TIMEOUT_SEC = 30;
const DEFAULT_TRANSLATION_BATCHING = {
  manual_batch_size: 50,
  api_batch_size: 25,
  api_min_batch_size: API_RETRY_MIN_BATCH_SIZE,
  api_max_chars_per_batch: 4000,
  api_concurrency: DEFAULT_API_CONCURRENCY,
  api_job_timeout_sec: 300,
};

function asrModelLabel(preset) {
  const model = preset?.model || preset?.label || preset?.id || "ASR";
  const engine = preset?.engine ? `${preset.engine}:` : "";
  const device = [preset?.device, preset?.compute_type].filter(Boolean).join("/");
  return `${engine}${model}${device ? ` · ${device}` : ""}`;
}

function hardwareOptionStatus(recommendation, group, id) {
  return recommendation?.availability?.[group]?.[id] || null;
}

function isHardwareOptionAvailable(recommendation, group, id) {
  const status = hardwareOptionStatus(recommendation, group, id);
  return !status || status.available !== false;
}

function hardwareOptionReason(recommendation, group, id) {
  const status = hardwareOptionStatus(recommendation, group, id);
  return status?.available === false ? (status.reason || "Không đủ phần cứng") : "";
}

// Thông số máy người dùng nhập — sống lâu hơn draft (không bị xoá khi
// "Xóa bản nháp"), mọi lần vào trang đều dùng lại.
const HW_STORAGE_KEY = "tubenote-hardware";

function readStoredHardware() {
  try {
    const saved = JSON.parse(localStorage.getItem(HW_STORAGE_KEY) || "null");
    if (saved && typeof saved === "object") {
      return {
        ram_gb: Number(saved.ram_gb) > 0 ? Number(saved.ram_gb) : null,
        vram_gb: Number(saved.vram_gb) > 0 ? Number(saved.vram_gb) : 0,
      };
    }
  } catch {}
  return null;
}

function ttsModelLabel(engine) {
  return engine?.label || engine?.id || "TTS";
}

function defaultNumStepForEngine(engine) {
  return engine === "omnivoice" ? 32 : 8;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function withTimeout(promise, timeoutMs, message) {
  let timeoutId;
  const timeout = new Promise((_, reject) => {
    timeoutId = setTimeout(() => reject(new Error(message)), timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timeoutId));
}

function hasObjectData(value) {
  return value && typeof value === "object" && Object.keys(value).length > 0;
}

function hasDraftWork({ responses, validated, dubJobId, dubbing }) {
  return (
    hasObjectData(responses) ||
    hasObjectData(validated) ||
    Boolean(dubJobId) ||
    Boolean(dubbing)
  );
}

const DEFAULT_OMNI_BUDGET_POLICY = {
  source_units_per_sec: 6.0,
  min_units_per_sec: 3.2,
  max_units_per_sec: 5.2,
  tolerance_ratio: 0.4,
  tolerance_min: 3,
};

function omniBudgetLabel(budget, policy = DEFAULT_OMNI_BUDGET_POLICY) {
  const sourceRate = Number(policy.source_units_per_sec) || DEFAULT_OMNI_BUDGET_POLICY.source_units_per_sec;
  const minRate = Number(policy.min_units_per_sec) || DEFAULT_OMNI_BUDGET_POLICY.min_units_per_sec;
  const maxRate = Number(policy.max_units_per_sec) || DEFAULT_OMNI_BUDGET_POLICY.max_units_per_sec;
  const toleranceRatio = Number(policy.tolerance_ratio) || DEFAULT_OMNI_BUDGET_POLICY.tolerance_ratio;
  const toleranceMin = Number(policy.tolerance_min) || DEFAULT_OMNI_BUDGET_POLICY.tolerance_min;
  const duration = Math.max(0.1, Number(budget || 0) / sourceRate);
  const min = Math.max(1, Math.floor(duration * minRate));
  const baseMax = Math.max(2, Math.floor(duration * maxRate));
  const tolerance = Math.max(toleranceMin, Math.ceil(baseMax * toleranceRatio));
  const max = baseMax + tolerance;
  return `[${min}-${max} tiếng]`;
}

function promptForEngine(prompt, engine, policy) {
  return prompt
    .replace(/\[≤\s*(\d+)\s*tiếng\]/g, (_, budget) => omniBudgetLabel(Number(budget), policy))
    .replace(
      "- Không chép ký hiệu \"[≤N tiếng]\" vào bản dịch.\n",
      "- Không chép marker \"[A-B tiếng]\" vào bản dịch.\n",
    )
    .replace(
      /2\. VỪA THỜI LƯỢNG[\s\S]*?(?=3\. DỊCH TỰ NHIÊN VÀ NHẤT QUÁN)/,
      "2. VỪA THỜI LƯỢNG CHO TTS\n- \"[A-B tiếng]\" là khoảng độ dài nên dùng cho dòng đó; B là giới hạn tối đa bắt buộc.\n- Một \"tiếng\" là một cụm được tách bằng khoảng trắng trong bản dịch tiếng Việt.\n- Sau khi dịch mỗi dòng, tự đếm số tiếng. Nếu vượt B, phải tự rút gọn dòng đó trước khi trả lời.\n- Cố gắng nằm trong khoảng A-B nếu vẫn đủ ý; với câu rất ngắn, được thấp hơn A nếu tự nhiên hơn.\n- Với slot rất ngắn, dùng cụm cực ngắn, có thể giữ thuật ngữ tiếng Anh nếu ngắn hơn bản Việt.\n- Nếu câu nguồn dài, bỏ từ đệm và ý phụ; không diễn giải thêm, không thêm ví dụ, không thêm chủ ngữ nếu không cần.\n\n",
    )
    .replace(
      "mỗi câu không vượt ngân sách.",
      "mỗi câu không vượt giới hạn B trong marker [A-B tiếng].",
    );
}

function promptLines(prompt) {
  const marker = "DỮ LIỆU NGUỒN";
  const markerIndex = prompt.lastIndexOf(marker);
  const source = markerIndex >= 0 ? prompt.slice(markerIndex + marker.length) : prompt;
  return source
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => /^\d+\s*[\.\)\:\-]\s+/.test(line));
}

function renumberPromptLine(line, index) {
  return line.replace(/^\s*\d+\s*([\.\)\:\-])\s*/, `${index + 1}. `);
}

function splitPromptForRetry(prompt, depth = 0, minBatchSize = API_RETRY_MIN_BATCH_SIZE) {
  const lines = promptLines(prompt);
  if (lines.length <= minBatchSize) return null;
  const marker = "DỮ LIỆU NGUỒN";
  const markerIndex = prompt.lastIndexOf(marker);
  if (markerIndex < 0) return null;
  const originalHeader = prompt.match(/\[batch_[^\]]+\]/)?.[0] || `[batch_retry_${depth + 1}]`;
  const prefix = prompt.slice(0, markerIndex + marker.length);
  const half = Math.ceil(lines.length / 2);
  return [lines.slice(0, half), lines.slice(half)].map((part, partIndex) => {
    const header = originalHeader.replace(/\]$/, `_r${depth + 1}_${partIndex + 1}]`);
    const body = part.map(renumberPromptLine).join("\n");
    return `${prefix.replaceAll(originalHeader, header)}\n${header}\n\n${body}`;
  });
}

function lineCountError(validation) {
  return /Thiếu\/thừa dòng|thiếu|thừa/i.test(validation?.error || "");
}

function responseFromSegments(promptIndex, segments) {
  return [
    `[batch_${promptIndex + 1}]`,
    ...segments.map((segment, index) => `${index + 1}. ${segment.vi || segment.tts || ""}`),
  ].join("\n");
}

function normalizeTtsConfig(value, fallbackModel = "M5", fallbackEngine = "supertonic") {
  const cfg = value && typeof value === "object" ? value : {};
  const engine = cfg.engine || fallbackEngine;
  const qualityOptions = TTS_QUALITY_OPTIONS[engine] || TTS_QUALITY_OPTIONS.supertonic;
  const allowedSteps = qualityOptions.map((option) => option.value);
  const numStep = allowedSteps.includes(Number(cfg.num_step))
    ? Number(cfg.num_step)
    : defaultNumStepForEngine(engine);
  return {
    engine,
    model: cfg.model || fallbackModel,
    device: cfg.device || "auto",
    voice_preset_id: cfg.voice_preset_id || "",
    voice_mode: cfg.voice_mode || "default",
    voice_id: cfg.voice_id || "",
    reference_audio_id: cfg.reference_audio_id || "",
    reference_text: cfg.reference_text || "",
    instruction: cfg.instruction || "",
    instruction_tags: Array.isArray(cfg.instruction_tags) ? cfg.instruction_tags : [],
    num_step: numStep,
    keep_background: cfg.keep_background !== false,
    batch_size: Number(cfg.batch_size) > 0 ? Math.floor(Number(cfg.batch_size)) : 0,
  };
}

function ttsConfigForEngine(engine, fallbackModels = ["M5"]) {
  const models = Array.isArray(engine?.models) && engine.models.length ? engine.models : fallbackModels;
  const model = engine?.default_model || models[0] || "M5";
  const engineId = engine?.id || "supertonic";
  return normalizeTtsConfig({
    engine: engineId,
    model,
    device: engineId === "omnivoice" ? "cuda" : "cpu",
    voice_preset_id: "",
    voice_mode: engine?.supports_clone ? "clone" : "default",
    voice_id: engine?.default_voice_id || "",
    reference_audio_id: "",
    reference_text: "",
    instruction: "",
    instruction_tags: [],
    num_step: defaultNumStepForEngine(engineId),
    keep_background: true,
  }, model, engineId);
}

function isWhisperLoadStage(stage) {
  return /Whisper|nhận diện giọng nói/i.test(stage || "");
}

export default function AddPage() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadStage, setLoadStage] = useState("");
  const [loadProgress, setLoadProgress] = useState(0);
  const [meta, setMeta] = useState(null);
  const [prompts, setPrompts] = useState([]);
  const [apiPrompts, setApiPrompts] = useState([]);
  const [responses, setResponses] = useState({});
  const [validated, setValidated] = useState({}); // i -> {ok, error, segments}
  const [dubbing, setDubbing] = useState(false);
  const [stage, setStage] = useState("");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [copiedIdx, setCopiedIdx] = useState(null);
  const [modalIdx, setModalIdx] = useState(null);
  const [expandedResponses, setExpandedResponses] = useState({});
  const [dubJobId, setDubJobId] = useState(null);
  const [restoredDraft, setRestoredDraft] = useState(false);
  const [ttsModels, setTtsModels] = useState([]);
  const [ttsEngines, setTtsEngines] = useState([]);
  const [ttsConfig, setTtsConfig] = useState(DEFAULT_TTS_CONFIG);
  const [speechPresets, setSpeechPresets] = useState([]);
  const [speechPreset, setSpeechPreset] = useState("cpu");
  const [hardware, setHardware] = useState(null);
  const [hwRam, setHwRam] = useState("");
  const [hwVram, setHwVram] = useState("");
  const [hwRec, setHwRec] = useState(null);
  const [hwDirty, setHwDirty] = useState(false);
  const [hwDetecting, setHwDetecting] = useState(false);
  const [hwApplying, setHwApplying] = useState(false);
  const [translationMode, setTranslationMode] = useState("manual");
  const [translationModelConfig, setTranslationModelConfig] = useState(FALLBACK_TRANSLATION_MODELS);
  const [translationProvider, setTranslationProvider] = useState(FALLBACK_TRANSLATION_MODELS.default_provider);
  const [translationModel, setTranslationModel] = useState(FALLBACK_TRANSLATION_MODELS.providers[0].default_model);
  const [translationBatching, setTranslationBatching] = useState(DEFAULT_TRANSLATION_BATCHING);
  const [translatingPrompts, setTranslatingPrompts] = useState({});
  const providerWindowsRef = useRef({});
  const mountedRef = useRef(false);
  const pollingJobRef = useRef(null);
  const suppressDraftRef = useRef(false);
  const autoLoadRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;

    try {
	      const saved = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
	      const params = new URLSearchParams(window.location.search);
	      const queryJob = params.get("job");
	      const queryUrl = params.get("url");
	      autoLoadRef.current = params.get("autoload") === "1";
	      if (saved) {
	        setUrl(saved.url || "");
        setMeta(saved.meta || null);
        setPrompts(Array.isArray(saved.prompts) ? saved.prompts : []);
        setApiPrompts(Array.isArray(saved.apiPrompts) ? saved.apiPrompts : []);
        setResponses(saved.responses || {});
        setValidated(saved.validated || {});
        setStage(saved.stage || "");
        setProgress(Number(saved.progress) || 0);
        setExpandedResponses(saved.expandedResponses || {});
	        const savedTts = saved.ttsConfig || saved.tts || (saved.ttsModel ? { model: saved.ttsModel } : null);
	        if (savedTts) setTtsConfig(normalizeTtsConfig(savedTts));
	        if (saved.speechPreset) setSpeechPreset(saved.speechPreset);
	        if (saved.translationMode === "api" || saved.translationMode === "manual") {
	          setTranslationMode(saved.translationMode);
	        }
	        if (saved.translationProvider) setTranslationProvider(saved.translationProvider);
        if (saved.translationModel) setTranslationModel(saved.translationModel);
        if (saved.translationBatching && typeof saved.translationBatching === "object") {
          setTranslationBatching({ ...DEFAULT_TRANSLATION_BATCHING, ...saved.translationBatching });
        }
	      }
	      if (queryUrl) setUrl(queryUrl);

      const jobId = queryJob || saved?.dubJobId || null;
      if (jobId) {
        setDubJobId(jobId);
        setDubbing(true);
        setStage(saved?.stage || "Đang khôi phục tiến trình dubbing");
        setJobParam(jobId);
        pollDubJob(jobId);
      }
    } catch {
      localStorage.removeItem(DRAFT_KEY);
    } finally {
      setRestoredDraft(true);
    }

    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Gộp các fetch vào 1 effect và tính default MỘT LẦN từ kết quả tất cả —
  // tách nhiều effect sẽ dính race lúc dev (StrictMode mount đôi): lần resolve
  // thứ hai của ttsModels ghi đè engine mà đề xuất phần cứng vừa áp.
  // Ưu tiên: lựa chọn đã lưu trong draft > đề xuất theo RAM/VRAM (đã lưu ở
  // HW_STORAGE_KEY nếu người dùng từng nhập, không thì theo máy detect) >
  // default backend.
  useEffect(() => {
    const storedHw = readStoredHardware();
    Promise.allSettled([
      api.ttsModels(),
      api.hardware(),
      storedHw
        ? api.hardwareRecommend(storedHw.ram_gb || 0, storedHw.vram_gb || 0)
        : Promise.resolve(null),
    ]).then(([modelsRes, hwRes, storedRecRes]) => {
      const hw = hwRes.status === "fulfilled" ? hwRes.value : null;
      setHardware(hw);
      const detected = hw?.detected || {};
      const ramValue = storedHw?.ram_gb ?? detected.ram_gb ?? null;
      const vramValue = storedHw ? storedHw.vram_gb : (detected.gpu?.vram_gb ?? 0);
      setHwRam(ramValue ? String(ramValue) : "");
      setHwVram(vramValue ? String(vramValue) : "");
      const storedRec = storedRecRes?.status === "fulfilled" ? storedRecRes.value : null;
      const recommendation = storedRec || hw?.recommendation || null;
      setHwRec(recommendation);
      if (modelsRes.status !== "fulfilled") {
        setTtsModels(["M5", "F5"]);
        setTtsEngines([]);
        setSpeechPresets([]);
        setSpeechPreset("cpu");
        setTtsConfig(ttsConfigForEngine({
          id: "supertonic",
          label: "Supertonic - CPU",
          default_model: "M5",
          models: ["M5", "F5"],
          supports_clone: false,
        }, ["M5", "F5"]));
        return;
      }
      const data = modelsRes.value;
      const models = Array.isArray(data.models) ? data.models : [];
      const engines = Array.isArray(data.engines) ? data.engines : [];
      const presets = Array.isArray(data.speech_presets) ? data.speech_presets : [];
      let saved = null;
      try {
        saved = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
      } catch {}
      const savedTts = saved?.ttsConfig || saved?.tts || (saved?.ttsModel ? { model: saved.ttsModel } : null);
      const recommendedPreset =
        recommendation && presets.some((preset) => preset.id === recommendation.asr_preset)
          ? recommendation.asr_preset
          : null;
      const recommendedEngineId =
        recommendation && engines.some((item) => item.id === recommendation.tts_engine)
          ? recommendation.tts_engine
          : null;
      const recommendedBatch =
        recommendation?.tts_engine === "omnivoice" ? (recommendation.omnivoice_batch_size || 0) : 0;
      const defaultPreset = presets.find((preset) => preset.id === data.default_speech_preset) || presets[0];
      const fallbackPreset =
        [defaultPreset, ...presets].find((preset) => (
          preset && isHardwareOptionAvailable(recommendation, "asr", preset.id)
        )) || presets[0];
      const savedPreset = presets.find((preset) => (
        preset.id === saved?.speechPreset &&
        isHardwareOptionAvailable(recommendation, "asr", preset.id)
      ));
      const nextSpeechPreset = savedPreset?.id || recommendedPreset || fallbackPreset?.id || "cpu";
      const defaultEngineId = recommendedEngineId || data.default_engine || engines[0]?.id || "supertonic";
      const fallbackEngine =
        [engines.find((item) => item.id === defaultEngineId), ...engines].find((item) => (
          item && isHardwareOptionAvailable(recommendation, "tts", item.id)
        )) || engines[0];
      const savedEngine = engines.find((item) => (
        item.id === savedTts?.engine &&
        isHardwareOptionAvailable(recommendation, "tts", item.id)
      ));
      const engine = savedEngine || fallbackEngine;
      setTtsModels(models);
      setTtsEngines(engines);
      setSpeechPresets(presets);
      setSpeechPreset(nextSpeechPreset);
      const normalizedSaved = savedTts
        ? normalizeTtsConfig(savedTts, engine?.default_model || models[0] || "M5", engine?.id || defaultEngineId)
        : null;
      setTtsConfig(
        normalizedSaved?.engine === (engine?.id || defaultEngineId)
          ? normalizedSaved
          : normalizeTtsConfig(
              {
                ...ttsConfigForEngine(engine, models.length ? models : ["M5"]),
                batch_size: engine?.id === recommendedEngineId ? recommendedBatch : 0,
              },
              engine?.default_model || models[0] || "M5",
              engine?.id || defaultEngineId,
            )
      );
    });
  }, []);

  useEffect(() => {
    api.translationModels()
      .then((config) => {
        const providers = Array.isArray(config?.providers) && config.providers.length
          ? config.providers
          : FALLBACK_TRANSLATION_MODELS.providers;
        let saved = null;
        try {
          saved = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
        } catch {}
        const nextConfig = {
          default_provider: config?.default_provider || providers[0].id,
          providers,
        };
        const defaultProvider = providers.find((item) => item.id === nextConfig.default_provider) || providers[0];
        const savedProvider = providers.find((item) => item.id === saved?.translationProvider);
        const provider = savedProvider || defaultProvider;
        setTranslationModelConfig(nextConfig);
        setTranslationProvider(provider.id);
        setTranslationModel(
          provider.models?.includes(saved?.translationModel)
            ? saved.translationModel
            : (provider.default_model || provider.models?.[0] || "")
        );
      })
      .catch(() => {
        setTranslationModelConfig(FALLBACK_TRANSLATION_MODELS);
      });
  }, []);

  useEffect(() => {
    if (!restoredDraft || !meta?.video_id) return;
    api.meta(meta.video_id)
      .then((m) => {
        if (m?.dubbed && mountedRef.current) clearDraft();
      })
      .catch(() => {});
  }, [restoredDraft, meta?.video_id]);

  useEffect(() => {
    if (!restoredDraft) return;
    if (suppressDraftRef.current) {
      localStorage.removeItem(DRAFT_KEY);
      return;
    }
    const hasDraft = hasDraftWork({ responses, validated, dubJobId, dubbing });
    if (!hasDraft) {
      localStorage.removeItem(DRAFT_KEY);
      return;
    }

    localStorage.setItem(DRAFT_KEY, JSON.stringify({
      url,
      meta,
      prompts,
      apiPrompts,
      responses,
      validated,
      dubbing,
      stage,
      progress,
      dubJobId,
      ttsConfig,
      speechPreset,
      translationMode,
      translationProvider,
      translationModel,
      translationBatching,
      expandedResponses,
      updatedAt: Date.now(),
    }));
  }, [
    restoredDraft,
    url,
    meta,
    prompts,
    apiPrompts,
    responses,
    validated,
    dubbing,
    stage,
    progress,
    dubJobId,
    ttsConfig,
    speechPreset,
    translationMode,
    translationProvider,
    translationModel,
    translationBatching,
    expandedResponses,
  ]);

  const hasSavedDraft = hasDraftWork({ responses, validated, dubJobId, dubbing });

  function setJobParam(jobId) {
    if (typeof window === "undefined") return;
    const next = new URL(window.location.href);
    if (jobId) next.searchParams.set("job", jobId);
    else next.searchParams.delete("job");
    window.history.replaceState(null, "", `${next.pathname}${next.search}${next.hash}`);
  }

  function resetDraftState() {
    setUrl("");
    setLoading(false);
    setLoadStage("");
    setLoadProgress(0);
    setMeta(null);
    setPrompts([]);
    setApiPrompts([]);
    setResponses({});
    setValidated({});
    setDubbing(false);
    setStage("");
    setProgress(0);
    setError("");
    setCopiedIdx(null);
    setModalIdx(null);
    setExpandedResponses({});
    setDubJobId(null);
    setTranslatingPrompts({});
    setTranslationBatching(DEFAULT_TRANSLATION_BATCHING);
    setSpeechPreset(speechPresets.find((preset) => preset.id === "cpu")?.id || speechPresets[0]?.id || "cpu");
    const defaultEngine = ttsEngines.find((engine) => engine.id === DEFAULT_TTS_CONFIG.engine) || ttsEngines[0];
    if (defaultEngine) {
      setTtsConfig(ttsConfigForEngine(defaultEngine, ttsModels.length ? ttsModels : ["M5"]));
    }
    pollingJobRef.current = null;
  }

  function clearDraft() {
    suppressDraftRef.current = true;
    localStorage.removeItem(DRAFT_KEY);
    setJobParam(null);
    resetDraftState();
  }

  async function onLoad() {
    if (hasSavedDraft) {
      const ok = window.confirm("Bạn đang có bản dịch chưa xuất video. Load video mới sẽ xóa bản nháp hiện tại. Tiếp tục?");
      if (!ok) return;
      localStorage.removeItem(DRAFT_KEY);
      setResponses({});
      setValidated({});
      setExpandedResponses({});
      setTranslatingPrompts({});
      setDubJobId(null);
      setStage("");
      setProgress(0);
      setJobParam(null);
    }
    suppressDraftRef.current = false;
    setError("");
    setLoading(true);
    setLoadStage("Bắt đầu");
    setLoadProgress(0);
    setDubbing(false);
    setDubJobId(null);
    setJobParam(null);
    try {
        const { job_id } = await api.load(
          url,
          selectedTtsEngine.id,
          selectedSpeechPreset.id,
        );
      // Polling vì bước Whisper có thể vài phút (tránh timeout).
      while (true) {
        await wait(1500);
        const s = await api.loadStatus(job_id);
        const nextStage = s.stage || "";
        const nextProgress = Number(s.progress) || 0;
        setLoadStage(nextStage);
        if (isWhisperLoadStage(nextStage) || nextProgress >= 100) {
          setLoadProgress(nextProgress);
        }
        if (s.status === "done") {
          setLoadProgress(100);
          const data = s.result;
          if (data.already_dubbed) {
            clearDraft();
            router.push(`/video/${data.video_id}`);
            return;
          }
          setMeta(data.metadata);
          setPrompts(Array.isArray(data.prompts) ? data.prompts : []);
          setApiPrompts(Array.isArray(data.api_prompts) && data.api_prompts.length ? data.api_prompts : (Array.isArray(data.prompts) ? data.prompts : []));
          setTranslationBatching({ ...DEFAULT_TRANSLATION_BATCHING, ...(data.translation_batching || {}) });
          setResponses({});
          setValidated({});
          setExpandedResponses({});
          setTranslatingPrompts({});
          return;
        }
        if (s.status === "error") {
          setError(s.error || "Lỗi không xác định");
          return;
        }
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!restoredDraft || !autoLoadRef.current || !url || loading || meta || prompts.length) return;
    autoLoadRef.current = false;
    onLoad();
  }, [restoredDraft, url, loading, meta, prompts.length]);

  const apiMinBatchSize = Math.max(
    1,
    Number(translationBatching.api_min_batch_size) || API_RETRY_MIN_BATCH_SIZE,
  );
  const apiConcurrency = Math.max(
    1,
    Math.min(64, Number(translationBatching.api_concurrency) || DEFAULT_API_CONCURRENCY),
  );
  const apiJobTimeoutMs = Math.max(
    MIN_API_JOB_TIMEOUT_SEC,
    Number(translationBatching.api_job_timeout_sec) || DEFAULT_TRANSLATION_BATCHING.api_job_timeout_sec,
  ) * 1000;

  function promptBudgets(prompt) {
    return Array.from(
      prompt.matchAll(/\[(?:≤\s*)?(\d+)(?:\s*-\s*(\d+))?\s*tiếng\]/g),
      (match) => Number(match[2] || match[1]),
    );
  }

  function selectedTranslationProvider() {
    return translationModelConfig.providers.find((provider) => provider.id === translationProvider)
      || translationModelConfig.providers[0]
      || FALLBACK_TRANSLATION_MODELS.providers[0];
  }

  function changeTranslationProvider(providerId) {
    const provider = translationModelConfig.providers.find((item) => item.id === providerId)
      || translationModelConfig.providers[0]
      || FALLBACK_TRANSLATION_MODELS.providers[0];
    setTranslationProvider(provider.id);
    setTranslationModel(provider.default_model || provider.models?.[0] || "");
  }

  function changeTranslationMode(mode) {
    setTranslationMode(mode);
    setResponses({});
    setValidated({});
    setExpandedResponses({});
    setCopiedIdx(null);
    setModalIdx(null);
    setTranslatingPrompts({});
  }

  function translatedPromptText(prompt) {
    return promptForEngine(
      prompt,
      selectedTtsEngine.id,
      selectedTtsEngine.budget_policy,
    );
  }

  async function validatePromptResponseText(i, text, prompt, engine = selectedTtsEngine.id) {
    const budgets = promptBudgets(prompt);
    const expected = promptLines(prompt).length || budgets.length;
    return api.validate(
      i,
      text,
      expected,
      engine,
      budgets,
    );
  }

  async function validateResponseText(i, text, engine = selectedTtsEngine.id) {
    return validatePromptResponseText(i, text, activePrompts[i], engine);
  }

  async function validateBatch(i, engine = selectedTtsEngine.id) {
    const budgets = promptBudgets(activePrompts[i]);
    const expected = promptLines(activePrompts[i]).length || budgets.length;
    return api.validate(
      i,
      responses[i] ?? "",
      expected,
      engine,
      budgets,
    );
  }

  async function onValidate(i) {
    const res = await validateBatch(i);
    setValidated((v) => ({ ...v, [i]: res }));
    if (!res.ok && res.segments?.length) {
      setExpandedResponses((cur) => ({ ...cur, [`content-${i}`]: true }));
    }
  }

  async function requestApiTranslation(i, promptText, deadlineMs) {
    const remainingForCreate = deadlineMs - Date.now();
    if (remainingForCreate <= 0) {
      throw new Error(`Prompt ${i + 1} dịch quá thời gian cho phép.`);
    }
    const { job_id } = await withTimeout(
      api.translatePrompt(
        i,
        promptText,
        translationProvider,
        translationModel,
      ),
      Math.min(API_STATUS_TIMEOUT_MS, remainingForCreate),
      `Không tạo được job dịch prompt ${i + 1} trong thời gian cho phép.`,
    );
    while (true) {
      const remaining = deadlineMs - Date.now();
      if (remaining <= 0) {
        throw new Error(`Prompt ${i + 1} dịch quá thời gian cho phép.`);
      }
      await wait(Math.min(1200, remaining));
      const statusRemaining = deadlineMs - Date.now();
      if (statusRemaining <= 0) {
        throw new Error(`Prompt ${i + 1} dịch quá thời gian cho phép.`);
      }
      const status = await withTimeout(
        api.translateStatus(job_id),
        Math.min(API_STATUS_TIMEOUT_MS, statusRemaining),
        `Không lấy được trạng thái dịch prompt ${i + 1}.`,
      );
      if (status.status === "done") {
        return status.result?.response || "";
      }
      if (status.status === "error") {
        throw new Error(status.error || "Không thể dịch prompt bằng API");
      }
    }
  }

  async function translatePromptTextWithRetry(i, promptText, depth = 0, deadlineMs = Date.now() + apiJobTimeoutMs) {
    const text = await requestApiTranslation(i, promptText, deadlineMs);
    const validation = await validatePromptResponseText(i, text, promptText);
    if (validation.ok || depth >= API_RETRY_MAX_DEPTH || !lineCountError(validation)) {
      return { response: text, validation };
    }

    const parts = splitPromptForRetry(promptText, depth, apiMinBatchSize);
    if (!parts) return { response: text, validation };

    const combinedSegments = [];
    for (const part of parts) {
      const sub = await translatePromptTextWithRetry(i, part, depth + 1, deadlineMs);
      if (!sub.validation?.ok) {
        return {
          response: sub.response,
          validation: {
            ...sub.validation,
            ok: false,
            error: `API dịch thiếu/thừa dòng sau khi chia nhỏ batch. ${sub.validation?.error || ""}`.trim(),
          },
        };
      }
      combinedSegments.push(...(sub.validation.segments || []));
    }

    const combinedResponse = responseFromSegments(i, combinedSegments);
    const combinedValidation = await validateResponseText(i, combinedResponse);
    return { response: combinedResponse, validation: combinedValidation };
  }

  async function translatePromptByApi(i) {
    if (!activePrompts[i] || translatingPrompts[i] || dubbing || loading) return null;
    setError("");
    setTranslatingPrompts((cur) => ({ ...cur, [i]: true }));
    try {
      const result = await translatePromptTextWithRetry(i, translatedPromptText(activePrompts[i]));
      const text = result.response || "";
      setResponses((cur) => ({ ...cur, [i]: text }));
      const validation = result.validation || await validateResponseText(i, text);
      setValidated((cur) => ({ ...cur, [i]: validation }));
      if (!validation.ok || validation.warnings?.length) {
        setExpandedResponses((cur) => ({ ...cur, [`content-${i}`]: true }));
      }
      return validation;
    } catch (e) {
      setError(String(e.message || e));
      return null;
    } finally {
      setTranslatingPrompts((cur) => {
        const next = { ...cur };
        delete next[i];
        return next;
      });
    }
  }

  async function translateAllByApi() {
    if (!activePrompts.length || dubbing || loading) return;
    setError("");
    const indexes = activePrompts
      .map((_, i) => i)
      .filter((i) => !validated[i]?.ok);  // batch có cảnh báo (ok) coi như đã dịch xong, không dịch lại
    if (!indexes.length) return;

    let cursor = 0;
    let failed = 0;
    const limit = Math.min(apiConcurrency, indexes.length);
    const worker = async () => {
      while (cursor < indexes.length) {
        const i = indexes[cursor];
        cursor += 1;
        const result = await translatePromptByApi(i);
        if (!result?.ok) {  // chỉ lỗi cấu trúc mới tính là fail; cảnh báo độ dài thì bỏ qua
          failed += 1;
        }
      }
    };

    await Promise.all(Array.from({ length: limit }, worker));

    if (failed > 0) {
      setError(`${failed}/${indexes.length} prompt API chưa đạt xác nhận. Mở các dòng đỏ để xem lỗi hoặc bấm dịch lại.`);
    }
  }

  function copyPrompt(text, i) {
    navigator.clipboard?.writeText(promptForEngine(
      text,
      selectedTtsEngine.id,
      selectedTtsEngine.budget_policy,
    ));
    setCopiedIdx(i);
    setTimeout(() => setCopiedIdx((cur) => (cur === i ? null : cur)), 1500);
  }

  function openTranslationProvider(provider) {
    const existing = providerWindowsRef.current[provider.id];
    if (existing && !existing.closed) {
      existing.focus();
    } else {
      providerWindowsRef.current[provider.id] = window.open(provider.url, `tubenote-${provider.id}`);
    }
  }

  function updateResponse(i, value) {
    setResponses((r) => ({ ...r, [i]: value }));
    setValidated((v) => {
      if (!v[i]) return v;
      const next = { ...v };
      delete next[i];
      return next;
    });
  }

  const apiTranslationMode = translationMode === "api";
  const activePrompts = apiTranslationMode ? (apiPrompts.length ? apiPrompts : prompts) : prompts;
  // Cảnh báo độ dài (warnings) KHÔNG chặn dub — chỉ cần cấu trúc hợp lệ (ok).
  const allValid = activePrompts.length > 0 && activePrompts.every((_, i) => (
    validated[i]?.ok
  ));

  async function pollDubJob(jobId) {
    pollingJobRef.current = jobId;
    setDubbing(true);
    while (mountedRef.current && pollingJobRef.current === jobId) {
      try {
        const s = await api.dubStatus(jobId);
        setStage(s.stage || "");
        setProgress(s.progress || 0);
        if (s.status === "done") {
          clearDraft();
          router.push(`/video/${s.result}`);
          return;
        }
        if (s.status === "error") {
          setError(s.error || "Lỗi không xác định");
          setDubbing(false);
          setDubJobId(null);
          setJobParam(null);
          return;
        }
      } catch (e) {
        if (meta?.video_id) {
          try {
            const m = await api.meta(meta.video_id);
            if (m?.dubbed) {
              clearDraft();
              return;
            }
          } catch {}
        }
        setError(`Không thể khôi phục tiến trình dubbing: ${String(e)}`);
        setDubbing(false);
        setDubJobId(null);
        setJobParam(null);
        return;
      }

      await wait(1500);
    }
  }

  async function onDub() {
    suppressDraftRef.current = false;
    setDubbing(true);
    setProgress(0);
    setStage("Bắt đầu");
    setError("");
    try {
      const segments = activePrompts.flatMap((_, i) => validated[i].segments);
      const modeEngine = selectedTtsEngine;
      const baseDubTtsConfig = ttsConfig.engine === modeEngine.id
        ? ttsConfig
        : ttsConfigForEngine(modeEngine, currentTtsModels);
      const dubTtsConfig = {
        ...baseDubTtsConfig,
        asr_preset: selectedSpeechPreset?.id || speechPreset,
        translation: {
          mode: translationMode,
          provider: translationMode === "api" ? translationProvider : "manual",
          model: translationMode === "api" ? translationModel : "ChatGPT",
        },
      };
      const { job_id } = await api.dub(url, segments, dubTtsConfig);
      setDubJobId(job_id);
      setJobParam(job_id);
      await pollDubJob(job_id);
    } catch (e) {
      setError(String(e));
      setDubbing(false);
      setDubJobId(null);
      setJobParam(null);
    }
  }

  const fallbackTtsEngine = {
    id: "supertonic",
    label: "Supertonic - CPU",
    description: "Nhanh, chạy CPU",
    models: ttsModels.length ? ttsModels : ["M5", "F5"],
    default_model: ttsModels[0] || "M5",
    devices: ["cpu"],
    supports_clone: false,
    supports_instruction: false,
  };
  const fallbackOmniEngine = {
    id: "omnivoice",
    label: "OmniVoice - GPU",
    description: "Chất lượng tự nhiên hơn, cần GPU",
    models: ["k2-fsa/OmniVoice"],
    default_model: "k2-fsa/OmniVoice",
    devices: ["cuda"],
    supports_clone: true,
    supports_instruction: true,
    default_voice_id: "academic_male",
    voices: [
      { id: "academic_male", label: "Giọng nam" },
      { id: "academic_female", label: "Giọng nữ" },
      { id: "source_video", label: "Giọng gốc video" },
    ],
  };
  const fallbackSpeechPresets = [
    {
      id: "cpu",
      label: "Whisper - CPU",
      description: "Tương thích máy không có GPU",
    },
    {
      id: "gpu",
      label: "Whisper - GPU",
      description: "Chất lượng cao, dùng GPU",
    },
  ];
  const availableSpeechPresets = speechPresets.length ? speechPresets : fallbackSpeechPresets;
  const showWhisperProgress = isWhisperLoadStage(loadStage);
  const availableTtsEngines = ttsEngines.length ? ttsEngines : [fallbackTtsEngine, fallbackOmniEngine];
  const currentAsrOptions = availableSpeechPresets;
  const selectedSpeechPreset =
    currentAsrOptions.find((preset) => preset.id === speechPreset) ||
    currentAsrOptions[0] ||
    availableSpeechPresets[0];
  const currentTtsEngineOptions = availableTtsEngines;
  const selectedTtsEngine =
    currentTtsEngineOptions.find((engine) => engine.id === ttsConfig.engine) ||
    currentTtsEngineOptions[0] ||
    availableTtsEngines[0];
  const baseTtsModels =
    Array.isArray(selectedTtsEngine.models) && selectedTtsEngine.models.length
      ? selectedTtsEngine.models
      : (ttsModels.length ? ttsModels : [ttsConfig.model || "M5"]);
  const currentTtsModels =
    ttsConfig.model && !baseTtsModels.includes(ttsConfig.model)
      ? [ttsConfig.model, ...baseTtsModels]
      : baseTtsModels;
  const currentVoicePresets =
    Array.isArray(selectedTtsEngine.voices) && selectedTtsEngine.voices.length
      ? selectedTtsEngine.voices
      : [];
  const defaultVoiceId = selectedTtsEngine.default_voice_id || currentVoicePresets[0]?.id || "";
  const supertonicVoiceLabels = {
    M5: "Giọng nam",
    F5: "Giọng nữ",
  };
  const supertonicVoiceOptions = currentTtsModels.map((model) => ({
    id: model,
    label: supertonicVoiceLabels[model] || model,
  }));
  const currentVoiceOptions = selectedTtsEngine.id === "supertonic"
    ? supertonicVoiceOptions
    : currentVoicePresets;
  const ttsVoiceChoice = selectedTtsEngine.id === "supertonic"
    ? ttsConfig.model
    : `preset:${ttsConfig.voice_id || defaultVoiceId}`;
  const selectedTtsVoiceLabel =
    currentVoiceOptions.find((voice) => (
      selectedTtsEngine.id === "supertonic"
        ? voice.id === ttsVoiceChoice
        : `preset:${voice.id}` === ttsVoiceChoice
    ))?.label || ttsVoiceChoice;
  const ttsQualityOptions = TTS_QUALITY_OPTIONS[selectedTtsEngine.id] || TTS_QUALITY_OPTIONS.supertonic;
  const recommendedOmniBatch = hwRec?.tts_engine === "omnivoice"
    ? Number(hwRec.omnivoice_batch_size) || 0
    : 0;
  const ttsBatchOptions = Array.from(new Set([
    ...OMNIVOICE_BATCH_OPTIONS,
    recommendedOmniBatch,
    Number(ttsConfig.batch_size) || 0,
  ])).filter((value) => value >= 0).sort((a, b) => a - b);
  const currentTranslationProvider = selectedTranslationProvider();
  const currentTranslationModels = currentTranslationProvider?.models || [];
  const translatingCount = Object.keys(translatingPrompts).length;
  const translatingAny = translatingCount > 0;

  function updateTtsConfig(patch) {
    setTtsConfig((cur) => normalizeTtsConfig({
      ...cur,
      ...patch,
    }, currentTtsModels[0] || "M5", selectedTtsEngine.id || "supertonic"));
  }

  async function changeTtsEngine(engineId, recommendation = hwRec) {
    if (!isHardwareOptionAvailable(recommendation, "tts", engineId)) return;
    const engine = availableTtsEngines.find((item) => item.id === engineId) || fallbackTtsEngine;
    setTtsConfig(ttsConfigForEngine(engine, currentTtsModels));
    setValidated({});
    const entries = await Promise.all(activePrompts.map(async (_, index) => {
      if (!(responses[index] || "").trim()) return null;
      return [index, await validateBatch(index, engine.id)];
    }));
    setValidated(Object.fromEntries(entries.filter(Boolean)));
  }

  function changeAsrModel(presetId) {
    if (!isHardwareOptionAvailable(hwRec, "asr", presetId)) return;
    setSpeechPreset(presetId);
  }

  const detectedHardware = hardware?.detected;
  const hardwareHint = detectedHardware
    ? `Phát hiện: ${detectedHardware.gpu
        ? `${detectedHardware.gpu.name} · ${detectedHardware.gpu.vram_gb}GB VRAM`
        : "không có GPU NVIDIA"}${detectedHardware.ram_gb ? ` · RAM ${detectedHardware.ram_gb}GB` : ""}${detectedHardware.cpu_cores ? ` · ${detectedHardware.cpu_cores} cores` : ""}`
    : "Không đọc được thông tin máy — nhập theo phần cứng của bạn.";
  const hwRecPresetLabel = hwRec
    ? (availableSpeechPresets.find((preset) => preset.id === hwRec.asr_preset)?.label || hwRec.asr_preset)
    : "";
  const hwRecSummary = hwRec
    ? `${hwRecPresetLabel} · ${hwRec.tts_engine === "omnivoice" ? "OmniVoice - GPU" : "Supertonic - CPU"}` +
      (hwRec.tts_engine === "omnivoice" && hwRec.omnivoice_batch_size
        ? ` · batch ${hwRec.omnivoice_batch_size}`
        : "") +
      ` · ${hwRec.whisper_cpu_threads} threads`
    : "";

  async function applyHardwareRecommendation(rec) {
    if (!rec) return;
    if (availableSpeechPresets.some((preset) => preset.id === rec.asr_preset)) {
      setSpeechPreset(rec.asr_preset);
    }
    const batch = rec.tts_engine === "omnivoice" ? (rec.omnivoice_batch_size || 0) : 0;
    if (selectedTtsEngine.id !== rec.tts_engine) {
      await changeTtsEngine(rec.tts_engine, rec);
      setTtsConfig((cur) => normalizeTtsConfig(
        { ...cur, batch_size: batch }, cur.model || "M5", cur.engine || rec.tts_engine,
      ));
    } else {
      updateTtsConfig({ batch_size: batch });
    }
  }

  function onHardwareInputChange(field, value) {
    if (field === "ram") setHwRam(value);
    else setHwVram(value);
    setHwDirty(true);
  }

  async function detectHardwareInput() {
    setHwDetecting(true);
    setError("");
    try {
      const hw = await api.hardware();
      setHardware(hw);
      const detected = hw?.detected || {};
      setHwRam(detected.ram_gb ? String(detected.ram_gb) : "");
      setHwVram(detected.gpu?.vram_gb ? String(detected.gpu.vram_gb) : "");
      setHwDirty(true);
    } catch (e) {
      setError(String(e));
    } finally {
      setHwDetecting(false);
    }
  }

  async function applyHardwareInputs() {
    const ram = Number(hwRam) || 0;
    const vram = Number(hwVram) || 0;
    setHwApplying(true);
    setError("");
    try {
      const rec = await api.hardwareRecommend(ram, vram);
      setHwRec(rec);
      try {
        localStorage.setItem(HW_STORAGE_KEY, JSON.stringify({ ram_gb: ram, vram_gb: vram }));
      } catch {}
      await applyHardwareRecommendation(rec);
      setHwDirty(false);
    } catch (e) {
      setError(String(e));
    } finally {
      setHwApplying(false);
    }
  }

  async function changeTtsModel(engineId) {
    await changeTtsEngine(engineId);
  }

  function changeTtsVoiceChoice(value) {
    if (selectedTtsEngine.id === "supertonic") {
      updateTtsConfig({ model: value });
      return;
    }
    updateTtsConfig({
      voice_id: value.replace(/^preset:/, ""),
      voice_preset_id: "",
      voice_mode: "clone",
      reference_audio_id: "",
      reference_text: "",
      instruction_tags: [],
    });
  }

  return (
    <main className="page-content">
      <div className="create-layout">
        <aside className="create-steps" aria-label="Quy trình tạo lồng tiếng">
          <div className={"step done"}>
            <span>1</span>
            <div><b>Nạp video</b><p>Lấy metadata, audio và transcript.</p></div>
          </div>
          <div className={activePrompts.length ? "step done" : "step"}>
            <span>2</span>
            <div><b>Dịch lời thoại</b><p>Kiểm tra số dòng trước khi ghép.</p></div>
          </div>
          <div className={allValid ? "step done" : "step"}>
            <span>3</span>
            <div><b>Dubbing</b><p>Tạo giọng đọc và xuất video.</p></div>
          </div>
        </aside>

        <section className="create-panel">
          <div className="feed-header compact">
            <div>
              <h1>Tạo video lồng tiếng</h1>
              <p>Dán link YouTube, dịch lời thoại và xuất bản MP4 đã lồng tiếng.</p>
            </div>
            {hasSavedDraft && (
              <button
                type="button"
                onClick={clearDraft}
                disabled={loading || dubbing}
                title="Xóa bản dịch đang làm dở"
              >
                Xóa bản nháp
              </button>
            )}
          </div>

            <div className="url-box">
            <label htmlFor="youtube-url">URL YouTube</label>
            <div className="url-row">
              <input
                id="youtube-url"
                type="text"
                placeholder="https://youtube.com/watch?v=..."
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && url && !loading && !dubbing && onLoad()}
              />
              <button className="primary" onClick={onLoad} disabled={!url || loading || dubbing}>
                {loading ? "Đang tải" : "Load"}
              </button>
            </div>
            <div className="load-options hw-options">
              <label className="tts-select">
                <span>RAM (GB)</span>
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={hwRam}
                  onChange={(event) => onHardwareInputChange("ram", event.target.value)}
                  disabled={loading || dubbing}
                />
              </label>
              <label className="tts-select">
                <span>VRAM GPU (GB)</span>
                <input
                  type="number"
                  min="0"
                  step="0.5"
                  placeholder="0 = không có"
                  value={hwVram}
                  onChange={(event) => onHardwareInputChange("vram", event.target.value)}
                  disabled={loading || dubbing}
                />
              </label>
              <em>{hardwareHint}</em>
              <div className="hw-actions">
                <button
                  type="button"
                  onClick={detectHardwareInput}
                  disabled={loading || dubbing || hwDetecting || hwApplying}
                >
                  {hwDetecting ? "Đang phát hiện" : "Tự phát hiện"}
                </button>
                <button
                  type="button"
                  className="primary"
                  onClick={applyHardwareInputs}
                  disabled={loading || dubbing || hwDetecting || hwApplying || Boolean(!hwDirty && hwRec)}
                >
                  {hwApplying ? "Đang áp dụng" : "Áp dụng cấu hình"}
                </button>
              </div>
            </div>
            {(hwRecSummary || hwDirty) && (
              <div className={`hw-recommend${hwDirty ? " pending" : ""}`}>
                {hwRecSummary ? (
                  <>
                    {hwDirty ? "Đang dùng cấu hình đã áp dụng: " : "Đề xuất đang áp dụng: "}
                    <b>{hwRecSummary}</b>
                  </>
                ) : (
                  "Chưa có cấu hình phần cứng đã áp dụng."
                )}
                {hwDirty && (
                  <span> Thông số RAM/VRAM đã đổi, nhấn áp dụng cấu hình để cập nhật lựa chọn.</span>
                )}
              </div>
            )}
            <div className="load-options">
              <label className="tts-select">
                <span>Mô hình ASR</span>
                <select
                  value={selectedSpeechPreset.id}
                  onChange={(event) => changeAsrModel(event.target.value)}
                  disabled={loading || dubbing || currentAsrOptions.length <= 1}
                >
                  {currentAsrOptions.map((preset) => {
                    const reason = hardwareOptionReason(hwRec, "asr", preset.id);
                    return (
                      <option key={preset.id} value={preset.id} disabled={Boolean(reason)}>
                        {asrModelLabel(preset)}{reason ? ` (${reason})` : ""}
                      </option>
                    );
                  })}
                </select>
              </label>
              {selectedSpeechPreset.description && (
                <em>{selectedSpeechPreset.description}</em>
              )}
            </div>
            <div className="load-options tts-engine-options">
              <label className="tts-select">
                <span>Mô hình TTS</span>
                <select
                  value={selectedTtsEngine.id}
                  onChange={(event) => changeTtsModel(event.target.value)}
                  disabled={loading || dubbing || currentTtsEngineOptions.length <= 1}
                >
                  {currentTtsEngineOptions.map((engine) => {
                    const reason = hardwareOptionReason(hwRec, "tts", engine.id);
                    return (
                      <option key={engine.id} value={engine.id} disabled={Boolean(reason)}>
                        {ttsModelLabel(engine)}{reason ? ` (${reason})` : ""}
                      </option>
                    );
                  })}
                </select>
              </label>
              {selectedTtsEngine.id === "omnivoice" && (
                <label className="tts-select">
                  <span>Batch size</span>
                  <select
                    value={Number(ttsConfig.batch_size) || 0}
                    onChange={(event) => updateTtsConfig({ batch_size: Number(event.target.value) || 0 })}
                    disabled={loading || dubbing}
                  >
                    {ttsBatchOptions.map((value) => (
                      <option key={value} value={value}>
                        {value === 0
                          ? "Auto"
                          : `${value}${value === recommendedOmniBatch ? " · đề xuất" : ""}`}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {selectedTtsEngine.description && (
                <em>{selectedTtsEngine.description}</em>
              )}
            </div>
          </div>

          {loading && (
            <div className="load-progress-box">
              {loadStage && <div className="stage">{loadStage}…</div>}
              {showWhisperProgress && (
                <div className="dub-progress compact">
                  <div className="dub-progress-bar">
                    <div className="dub-progress-fill" style={{ width: `${Math.max(0, Math.min(100, loadProgress))}%` }} />
                  </div>
                  <div className="dub-progress-label">
                    <span>Whisper STT</span>
                    <span>{Math.max(0, Math.min(100, loadProgress))}%</span>
                  </div>
                </div>
              )}
            </div>
          )}
          {error && <div className="tag-fail">{error}</div>}

          {meta && (
            <div className="meta-box">
              {meta.thumbnail && <img src={meta.thumbnail} alt="" />}
              <div>
                <div className="meta-title">{meta.title}</div>
                <div className="meta-sub">{meta.channel}</div>
                <div className="meta-sub">ID: {meta.video_id}</div>
              </div>
            </div>
          )}

            {activePrompts.length > 0 && (
              <div className="prompt-workflow">
                <div className="tts-panel">
                  <div className="tts-panel-head">
                    <div>
                      <span>TTS</span>
                      <strong>{selectedTtsEngine.label || selectedTtsEngine.id} / {selectedTtsVoiceLabel}</strong>
                    </div>
                    {selectedTtsEngine.description && <em>{selectedTtsEngine.description}</em>}
                  </div>
	                  <div className="tts-grid">
	                    <label className="tts-select">
	                      <span>Giọng đọc</span>
	                      <select
	                        value={ttsVoiceChoice}
	                        onChange={(e) => changeTtsVoiceChoice(e.target.value)}
	                        disabled={dubbing || currentVoiceOptions.length <= 1}
	                      >
	                        {currentVoiceOptions.map((voice) => (
	                          <option
	                            key={voice.id}
	                            value={selectedTtsEngine.id === "supertonic" ? voice.id : `preset:${voice.id}`}
	                          >
	                            {voice.label || voice.id}
	                          </option>
	                        ))}
	                      </select>
	                    </label>
	                    <label className="tts-select">
                          <span>Chất lượng</span>
                          <select
                            value={ttsConfig.num_step}
                            onChange={(e) => updateTtsConfig({ num_step: Number(e.target.value) })}
                            disabled={dubbing}
                          >
                            {ttsQualityOptions.map((option) => (
                              <option key={option.value} value={option.value}>
                                {option.label}
                              </option>
                            ))}
                          </select>
                        </label>
		                  </div>
                </div>
                <div className="section-head">
                  <div>
                    <h2>Dịch lời thoại</h2>
                    <p>
                      {apiTranslationMode
                        ? "Dịch tự động bằng API rồi tự kiểm tra định dạng trước khi dubbing."
                        : "Copy từng prompt sang ChatGPT rồi dán câu trả lời vào đây."}
                    </p>
                  </div>
                  <div className="translation-tools">
                    <div className="translation-mode-controls">
                      <label className="tts-select">
                        <span>Chế độ dịch</span>
                        <select
                          value={translationMode}
                          onChange={(event) => changeTranslationMode(event.target.value)}
                          disabled={loading || dubbing || translatingAny}
                        >
                          <option value="manual">Thủ công</option>
                          <option value="api">API</option>
                        </select>
                      </label>
                    </div>
                    {apiTranslationMode ? (
                      <div className="translation-api-controls" aria-label="Dịch bằng API">
                        <label className="tts-select">
                          <span>Provider API</span>
                          <select
                            value={translationProvider}
                            onChange={(event) => changeTranslationProvider(event.target.value)}
                            disabled={loading || dubbing || translatingAny}
                          >
                            {translationModelConfig.providers.map((provider) => (
                              <option key={provider.id} value={provider.id}>
                                {provider.label || provider.id}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="tts-select">
                          <span>Model API</span>
                          <select
                            value={translationModel}
                            onChange={(event) => setTranslationModel(event.target.value)}
                            disabled={loading || dubbing || translatingAny || currentTranslationModels.length <= 1}
                          >
                            {currentTranslationModels.map((model) => (
                              <option key={model} value={model}>
                                {model}
                              </option>
                            ))}
                          </select>
                        </label>
                        <button
                          type="button"
                          className="primary"
                          onClick={translateAllByApi}
                          disabled={loading || dubbing || translatingAny || !translationProvider || !translationModel}
                        >
                          {translatingAny ? `Đang dịch ${translatingCount}` : "Dịch tất cả bằng API"}
                        </button>
                      </div>
                    ) : (
                      <div className="provider-actions">
                        {TRANSLATION_PROVIDERS.map((provider) => (
                          <button
                            key={provider.id}
                            className="primary"
                            onClick={() => openTranslationProvider(provider)}
                          >
                            Mở {provider.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

              <div className="prompt-list">
                {activePrompts.map((p, i) => {
                  const v = validated[i];
                  const resp = responses[i] ?? "";
                  const ttsSegments = Array.isArray(v?.segments) ? v.segments : [];
                  const ttsErrorCount = ttsSegments.reduce((total, segment) => (
                    total + (segment.normalization?.errors?.length || 0)
                  ), 0);
                  return (
                    <div className="prompt-card" key={i}>
                      <span className="prompt-no">{i + 1}</span>
                      <div className="prompt-main">
                        {apiTranslationMode ? (
                          <>
                            <button
                              className="chip"
                              disabled={loading || dubbing || translatingPrompts[i] || !translationProvider || !translationModel}
                              onClick={() => translatePromptByApi(i)}
                            >
                              {translatingPrompts[i] ? "Đang dịch" : (resp.trim() ? "Dịch lại API" : "Dịch API")}
                            </button>
                            {resp.trim() && (
                              <button className="chip chip-pasted" onClick={() => setModalIdx(i)}>
                                Sửa nội dung
                              </button>
                            )}
                          </>
                        ) : (
                          <>
                            <button
                              className={"chip chip-copy" + (copiedIdx === i ? " chip-ok" : "")}
                              onClick={() => copyPrompt(p, i)}
                            >
                              {copiedIdx === i ? "Đã copy" : "Copy prompt"}
                            </button>
                            {resp.trim() ? (
                              <button className="chip chip-pasted" onClick={() => setModalIdx(i)}>
                                Sửa nội dung
                              </button>
                            ) : (
                              <button className="chip" onClick={() => setModalIdx(i)}>
                                Dán kết quả dịch
                              </button>
                            )}
                          </>
                        )}
                      </div>
                      <div className="prompt-end">
                        {v == null && <span className="tag-wait">Chờ</span>}
                        {v?.ok && !v.warnings?.length && <span className="tag-ok">{v.segments.length} câu</span>}
                        {v?.ok && v.warnings?.length > 0 && (
                          <span className="tag-fail">{v.warnings.length} cảnh báo</span>
                        )}
                        {v && !v.ok && <span className="tag-fail">Lỗi</span>}
                        <button className="primary" disabled={!resp.trim()} onClick={() => onValidate(i)}>
                          Xác nhận
                        </button>
                      </div>
                      {v && !v.ok && (
                        <div className="prompt-error">
                          <span>Không xác nhận được prompt {i + 1}</span>
                          <p>{v.error || "Kết quả dịch không hợp lệ. Vui lòng kiểm tra lại số dòng và định dạng."}</p>
                          {ttsSegments.length === 0 && <p>Chưa có văn bản hợp lệ để gửi vào TTS.</p>}
                        </div>
                      )}
                      {resp.trim() && (
                        <div className="content-preview">
                          <button
                            className="preview-toggle"
                            onClick={() => setExpandedResponses((cur) => ({
                              ...cur,
                              [`content-${i}`]: !cur[`content-${i}`],
                            }))}
                            aria-expanded={!!expandedResponses[`content-${i}`]}
                          >
                            <span>{expandedResponses[`content-${i}`] ? "▾" : "▸"}</span>
                            Nội dung
                            {ttsErrorCount > 0 && (
                              <b className="fail">{ttsErrorCount} lỗi</b>
                            )}
                            {ttsErrorCount === 0 && v?.warnings?.length > 0 && (
                              <b>{v.warnings.length} cảnh báo</b>
                            )}
                          </button>
                          {expandedResponses[`content-${i}`] && (
                            <div className="content-body">
                              {ttsSegments.length > 0 ? (
                                <div className="tts-text-list">
                                  {ttsSegments.map((segment, segmentIndex) => {
                                    const segmentErrors = segment.normalization?.errors || [];
                                    const segmentWarnings = segment.normalization?.warnings || [];
                                    return (
                                      <div
                                        className={"tts-text-row" + (segmentErrors.length ? " has-error" : "")}
                                        key={segmentIndex}
                                      >
                                        <span>{segmentIndex + 1}</span>
                                        <div>
                                          <label>Văn bản gửi</label>
                                          <p>{segment.vi}</p>
                                          <label>Văn bản vào TTS</label>
                                          <code>{segment.tts || segment.vi}</code>
                                          {segmentErrors.map((error) => (
                                            <em className="error" key={error}>{error}</em>
                                          ))}
                                          {segmentWarnings.map((warning) => (
                                            <em key={warning}>{warning}</em>
                                          ))}
                                        </div>
                                      </div>
                                    );
                                  })}
                                </div>
                              ) : (
                                <div className="raw-content-block">
                                  <label>Văn bản gửi</label>
                                  <pre>{resp}</pre>
                                  <label>Văn bản vào TTS</label>
                                  <p>Chưa có dữ liệu hợp lệ. Bấm xác nhận để kiểm tra hoặc sửa lại nội dung đã dán.</p>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              <div className="publish-bar">
                <label className="background-toggle">
                  <input
                    type="checkbox"
                    checked={ttsConfig.keep_background !== false}
                    disabled={dubbing}
                    onChange={(event) => setTtsConfig((current) => ({
                      ...current,
                      keep_background: event.target.checked,
                    }))}
                  />
                  <span>Giữ nhạc nền (lâu hơn vài phút)</span>
                </label>
                <button className="primary" disabled={!allValid || dubbing} onClick={onDub}>
                  {dubbing ? "Đang xử lý" : "Bắt đầu dubbing"}
                </button>
                {dubbing && (
                  <div className="dub-progress">
                    <div className="dub-progress-bar">
                      <div className="dub-progress-fill" style={{ width: `${progress}%` }} />
                    </div>
                    <div className="dub-progress-label">
                      <span>{stage}</span>
                      <span>{progress}%</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </section>
      </div>

      {modalIdx !== null && (
        <div className="modal-overlay" onClick={() => setModalIdx(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <span>Kết quả dịch — Prompt {modalIdx + 1}</span>
              <button className="modal-x" onClick={() => setModalIdx(null)}>✕</button>
            </div>
            <textarea
              autoFocus
              rows={14}
              placeholder={`Dán câu trả lời AI dịch cho prompt ${modalIdx + 1} vào đây...`}
              value={responses[modalIdx] ?? ""}
              onChange={(e) => updateResponse(modalIdx, e.target.value)}
            />
            <div className="modal-foot">
              <button onClick={() => setModalIdx(null)}>Đóng</button>
              <button
                className="primary"
                onClick={() => setModalIdx(null)}
                disabled={!(responses[modalIdx] ?? "").trim()}
              >
                Lưu
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
