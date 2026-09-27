// SignBridge server: serves the app, translates signs and describes objects with an LLM,
// and turns text into expressive speech. No framework, one dependency.
import http from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { translateWithRules } from "./lib/rules.mjs";
import { wikiSummary } from "./lib/wiki.mjs";
import { englishToGloss, VOCAB, lemma } from "./lib/to-gloss.mjs";
import {
  TONES,
  LANGS,
  TRANSLATE_SYSTEM,
  TRANSLATE_SCHEMA,
  buildTranslatePrompt,
  DESCRIBE_SYSTEM,
  DESCRIBE_SCHEMA,
  buildDescribePrompt,
  TO_SIGN_SYSTEM,
  TO_SIGN_SCHEMA,
  buildToSignPrompt,
  IDENTIFY_SYSTEM,
  IDENTIFY_SCHEMA,
  SCENE_SCHEMA,
  buildIdentifyPrompt,
} from "./lib/prompt.mjs";

try {
  process.loadEnvFile();
} catch {
  // no .env file: everything falls back to defaults
}

const here = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.join(here, "public");
const PORT = Number(process.env.PORT || 5173);
const OLLAMA_URL = (process.env.OLLAMA_URL || "http://127.0.0.1:11434").replace(/\/$/, "");
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "llama3.2";
const OLLAMA_SEES = /llava|vision|gemma3|qwen2\.5vl|qwen3-vl|moondream|minicpm-v/i.test(OLLAMA_MODEL);
const CLAUDE_MODEL = process.env.CLAUDE_MODEL || "claude-opus-5";
const GROQ_MODEL = process.env.GROQ_MODEL || "openai/gpt-oss-120b";
const GROQ_VISION_MODEL = process.env.GROQ_VISION_MODEL || "qwen/qwen3.8-27b";
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-flash-latest";
const ENV_FILE = path.join(here, ".env");
const ELEVEN_VOICE = process.env.ELEVENLABS_VOICE_ID || "EXAVITQu4vr4xnSDxMaL";
const ELEVEN_MODEL = process.env.ELEVENLABS_MODEL || "eleven_v3";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
};

// ---------- LLM backends ----------

async function ollamaStatus() {
  try {
    const res = await fetch(`${OLLAMA_URL}/api/tags`, { signal: AbortSignal.timeout(800) });
    if (!res.ok) return { running: false, hasModel: false };
    const { models = [] } = await res.json();
    const hasModel = models.some((m) => m.name === OLLAMA_MODEL || m.name.startsWith(`${OLLAMA_MODEL}:`));
    return { running: true, hasModel, models: models.map((m) => m.name) };
  } catch {
    return { running: false, hasModel: false };
  }
}

const KEY_FOR = { claude: "ANTHROPIC_API_KEY", groq: "GROQ_API_KEY", gemini: "GEMINI_API_KEY" };
const MODEL_FOR = { ollama: () => OLLAMA_MODEL, claude: () => CLAUDE_MODEL, groq: () => GROQ_MODEL, gemini: () => GEMINI_MODEL };

async function pickBackend() {
  const wanted = (process.env.LLM || "auto").toLowerCase();
  if (wanted === "rules") return "rules";
  if (KEY_FOR[wanted]) return process.env[KEY_FOR[wanted]] ? wanted : "rules";
  const ollama = await ollamaStatus();
  if (wanted === "ollama") return ollama.running && ollama.hasModel ? "ollama" : "rules";
  if (ollama.running && ollama.hasModel) return "ollama";
  // free cloud options first, then Claude
  for (const b of ["groq", "gemini", "claude"]) if (process.env[KEY_FOR[b]]) return b;
  return "rules";
}

async function askOllama({ system, user, schema, image }) {
  const res = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    signal: AbortSignal.timeout(90_000),
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      stream: false,
      format: schema,
      options: { temperature: 0.4 },
      messages: [
        { role: "system", content: system },
        { role: "user", content: user, ...(image && OLLAMA_SEES ? { images: [image] } : {}) },
      ],
    }),
  });
  if (!res.ok) throw new Error(`ollama ${res.status}: ${await res.text()}`);
  return JSON.parse((await res.json()).message.content);
}

let anthropic;
async function askClaude({ system, user, schema, image }) {
  if (!anthropic) {
    const { default: Anthropic } = await import("@anthropic-ai/sdk");
    anthropic = new Anthropic();
  }
  const content = image
    ? [
        { type: "image", source: { type: "base64", media_type: "image/jpeg", data: image } },
        { type: "text", text: user },
      ]
    : user;
  const response = await anthropic.beta.messages.create({
    model: CLAUDE_MODEL,
    max_tokens: 2048,
    // If the primary model declines, the API retries on a fallback model in the same call.
    betas: ["server-side-fallback-2026-07-01"],
    fallbacks: "default",
    output_config: { effort: "low", format: { type: "json_schema", schema } },
    system,
    messages: [{ role: "user", content }],
  });
  if (response.stop_reason === "refusal") throw new Error("claude declined this request");
  const text = response.content.find((b) => b.type === "text")?.text;
  if (!text) throw new Error("claude returned no text");
  return JSON.parse(text);
}

// Groq: OpenAI-compatible, free tier. gpt-oss for text (strict JSON schema), a Qwen model when there's an image.
async function askGroq({ system, user, schema, image }) {
  const model = image ? GROQ_VISION_MODEL : GROQ_MODEL;
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: { authorization: `Bearer ${process.env.GROQ_API_KEY}`, "content-type": "application/json" },
    signal: AbortSignal.timeout(45_000),
    body: JSON.stringify({
      model,
      temperature: 0.4,
      ...(model.startsWith("openai/gpt-oss") ? { reasoning_effort: "low" } : {}),
      response_format: { type: "json_schema", json_schema: { name: "result", strict: true, schema } },
      messages: [
        { role: "system", content: system },
        {
          role: "user",
          content: image ? [{ type: "text", text: user }, { type: "image_url", image_url: { url: `data:image/jpeg;base64,${image}` } }] : user,
        },
      ],
    }),
  });
  if (!res.ok) throw new Error(`groq ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return JSON.parse((await res.json()).choices[0].message.content);
}

// Gemini: free tier with a Google account. JSON mode, with the schema spelled out in the instructions.
async function askGemini({ system, user, schema, image }) {
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(GEMINI_MODEL)}:generateContent`, {
    method: "POST",
    headers: { "x-goog-api-key": process.env.GEMINI_API_KEY, "content-type": "application/json" },
    signal: AbortSignal.timeout(45_000),
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: `${system}\n\nRespond with only a JSON object that matches this JSON Schema:\n${JSON.stringify(schema)}` }] },
      contents: [{ role: "user", parts: [{ text: user }, ...(image ? [{ inline_data: { mime_type: "image/jpeg", data: image } }] : [])] }],
      generationConfig: { responseMimeType: "application/json", temperature: 0.4 },
    }),
  });
  if (!res.ok) throw new Error(`gemini ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const data = await res.json();
  const text = data.candidates?.[0]?.content?.parts?.filter((p) => !p.thought).map((p) => p.text || "").join("");
  if (!text) throw new Error(`gemini returned no text (${data.candidates?.[0]?.finishReason || data.promptFeedback?.blockReason || "unknown"})`);
  return JSON.parse(text.replace(/^```(?:json)?\s*|\s*```$/g, ""));
}

const ASK = { ollama: askOllama, claude: askClaude, groq: askGroq, gemini: askGemini };
async function ask(backend, req) {
  return ASK[backend](req);
}

const cleanTone = (t, fallback = "neutral") => (TONES.includes(t) ? t : fallback);
const cleanLang = (l) => (LANGS.includes(l) ? l : "en");

async function translate(payload) {
  const lang = cleanLang(payload.lang);
  const backend = await pickBackend();
  const started = Date.now();
  if (backend !== "rules") {
    try {
      const raw = await ask(backend, { system: TRANSLATE_SYSTEM, user: buildTranslatePrompt({ ...payload, lang }), schema: TRANSLATE_SCHEMA });
      const text = String(raw.text || "").trim();
      if (!text) throw new Error("model returned no text");
      return { text, speech: String(raw.speech || text), tone: cleanTone(raw.tone), notes: String(raw.notes || ""), lang, backend, ms: Date.now() - started };
    } catch (err) {
      console.warn(`[translate] ${backend} failed, using rules:`, err.message);
      return { ...translateWithRules({ ...payload, lang }), lang: "en", backend: "rules", fallbackFrom: backend, ms: Date.now() - started };
    }
  }
  return { ...translateWithRules({ ...payload, lang }), lang: "en", backend, ms: Date.now() - started };
}

async function describe({ label, score, image, lang: rawLang }) {
  const lang = cleanLang(rawLang);
  const started = Date.now();
  const [backend, wiki] = await Promise.all([pickBackend(), wikiSummary(label)]);
  const img = typeof image === "string" && image.length < 2_000_000 ? image.replace(/^data:image\/\w+;base64,/, "") : null;
  if (backend !== "rules") {
    try {
      const sees = backend !== "ollama" || OLLAMA_SEES;
      const raw = await ask(backend, {
        system: DESCRIBE_SYSTEM,
        user: buildDescribePrompt({ label, score, lang, wiki: wiki?.extract, hasImage: Boolean(img && sees) }),
        schema: DESCRIBE_SCHEMA,
        image: sees ? img : null,
      });
      return {
        name: String(raw.name || label),
        text: String(raw.text || ""),
        speech: String(raw.speech || raw.text || ""),
        facts: Array.isArray(raw.facts) ? raw.facts.slice(0, 3).map(String) : [],
        tone: cleanTone(raw.tone, "calm"),
        lang,
        wiki,
        backend,
        sawImage: Boolean(img && sees),
        ms: Date.now() - started,
      };
    } catch (err) {
      console.warn(`[describe] ${backend} failed, using Wikipedia:`, err.message);
    }
  }
  // offline: first two sentences of the Wikipedia summary
  const sentences = (wiki?.extract || "").match(/[^.!?]+[.!?]+/g) || [];
  const text = sentences.slice(0, 2).join(" ").trim() || `That looks like a ${label}.`;
  return {
    name: wiki?.title || label,
    text,
    speech: text,
    facts: sentences.slice(2, 5).map((s) => s.trim()),
    tone: "calm",
    lang: "en",
    wiki,
    backend: "wikipedia",
    ms: Date.now() - started,
  };
}

// Short, precise names from a vision-capable model: one object crop, or a whole scene.
async function identify({ image, label, scene }) {
  const backend = await pickBackend();
  const sees = backend !== "rules" && (backend !== "ollama" || OLLAMA_SEES);
  if (!sees) throw Object.assign(new Error("Connect Groq, Gemini or Claude in Settings to identify objects with AI."), { status: 501 });
  const img = typeof image === "string" ? image.replace(/^data:image\/\w+;base64,/, "") : "";
  if (!img || img.length > 2_000_000) throw Object.assign(new Error("no image"), { status: 400 });
  const started = Date.now();
  const raw = await ask(backend, { system: IDENTIFY_SYSTEM, user: buildIdentifyPrompt({ label: scene ? null : label }), schema: scene ? SCENE_SCHEMA : IDENTIFY_SCHEMA, image: img });
  const clean = (x) => String(x || "").trim().toLowerCase().replace(/[.!]+$/, "").slice(0, 60);
  if (scene) return { objects: (raw.objects || []).map(clean).filter(Boolean).slice(0, 8), backend, ms: Date.now() - started };
  return { name: clean(raw.name) || label, sure: raw.sure !== false, backend, ms: Date.now() - started };
}

// Spoken text → ASL sign sequence. LLM when available (handles Hindi/Hinglish), rules otherwise.
const VOCAB_LIST = [...VOCAB].sort();
async function toSign({ text, lang }) {
  const started = Date.now();
  const clean = String(text || "").slice(0, 400);
  const backend = await pickBackend();
  if (backend !== "rules") {
    try {
      const raw = await ask(backend, { system: TO_SIGN_SYSTEM, user: buildToSignPrompt(clean, VOCAB_LIST), schema: TO_SIGN_SCHEMA });
      const signs = (raw.signs || []).slice(0, 40).map(({ word, fingerspell }) => {
        const w = String(word || "").toLowerCase().trim();
        const l = !fingerspell && lemma(w);
        return l ? { word: l, video: true } : { word: w.replace(/[^a-z0-9 ]/g, ""), video: false };
      }).filter((x) => x.word);
      return { signs, backend, ms: Date.now() - started };
    } catch (err) {
      console.warn(`[to-sign] ${backend} failed, using rules:`, err.message);
    }
  }
  if (lang === "hi" && /[\u0900-\u097F]/.test(clean)) {
    throw Object.assign(new Error("Hindi → sign needs an AI. Add a free Groq or Gemini key in Settings."), { status: 501 });
  }
  return { signs: englishToGloss(clean), backend: "rules", ms: Date.now() - started };
}

// ---------- voice ----------

// Settings used when the model doesn't take audio tags (eleven_multilingual_v2 etc).
// Lower stability = more expressive; higher style = more exaggerated delivery.
const VOICE_BY_TONE = {
  neutral: { stability: 0.5, style: 0.25 },
  calm: { stability: 0.65, style: 0.15 },
  happy: { stability: 0.35, style: 0.55 },
  excited: { stability: 0.25, style: 0.75 },
  sad: { stability: 0.6, style: 0.45 },
  angry: { stability: 0.3, style: 0.7 },
  surprised: { stability: 0.28, style: 0.65 },
  questioning: { stability: 0.45, style: 0.35 },
};
const TONE_TAG = { happy: "[cheerful]", excited: "[excited]", sad: "[sad]", angry: "[frustrated]", surprised: "[surprised]", calm: "[calm]", questioning: "[curious]" };
const hasTags = (s) => /\[[a-z][a-z ]{1,24}\]/i.test(s);
const stripTags = (s) => s.replace(/\[[a-z][a-z ]{1,24}\]\s*/gi, "").trim();

async function elevenlabs({ text, model, voice, tone }) {
  const v3 = model.startsWith("eleven_v3");
  const input = v3 ? (hasTags(text) || !TONE_TAG[tone] ? text : `${TONE_TAG[tone]} ${text}`) : stripTags(text);
  const settings = v3 ? { stability: 0.5 } : { ...VOICE_BY_TONE[tone], similarity_boost: 0.8, use_speaker_boost: true };
  const res = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(voice)}?output_format=mp3_44100_128`, {
    method: "POST",
    headers: { "xi-api-key": process.env.ELEVENLABS_API_KEY, "content-type": "application/json", accept: "audio/mpeg" },
    signal: AbortSignal.timeout(30_000),
    body: JSON.stringify({ text: input, model_id: model, voice_settings: settings }),
  });
  if (!res.ok) {
    const err = new Error(`elevenlabs ${res.status}: ${(await res.text()).slice(0, 300)}`);
    err.status = res.status;
    throw err;
  }
  return Buffer.from(await res.arrayBuffer());
}

// Sarvam Bulbul: native Indian voices (Hindi, Indian English and 9 more Indian languages).
// Tone maps to pace and "temperature" (how expressive the delivery is).
const SARVAM_TONE = {
  neutral: [1.0, 0.6], calm: [0.94, 0.45], happy: [1.06, 0.85], excited: [1.14, 1.0],
  sad: [0.88, 0.7], angry: [1.06, 0.9], surprised: [1.1, 0.95], questioning: [1.0, 0.7],
};
const SARVAM_SPEAKERS = new Set(["ritu", "priya", "neha", "pooja", "simran", "kavya", "ishita", "shreya", "roopa", "tanya", "shruti", "suhani", "kavitha", "rupali"]);
const SARVAM_DEFAULT = process.env.SARVAM_SPEAKER || "priya";

async function sarvam({ text, tone, speaker, lang }) {
  const [pace, temperature] = SARVAM_TONE[tone] || SARVAM_TONE.neutral;
  const res = await fetch("https://api.sarvam.ai/text-to-speech", {
    method: "POST",
    headers: { "api-subscription-key": process.env.SARVAM_API_KEY, "content-type": "application/json" },
    signal: AbortSignal.timeout(30_000),
    body: JSON.stringify({
      text: stripTags(text),
      language_code: lang === "hi" ? "hi-IN" : "en-IN", // Hinglish (Roman script) reads naturally with en-IN
      speaker: SARVAM_SPEAKERS.has(speaker) ? speaker : SARVAM_DEFAULT,
      model: "bulbul:v3",
      pace,
      temperature,
      speech_sample_rate: 24000,
      enable_preprocessing: true,
    }),
  });
  if (!res.ok) throw Object.assign(new Error(`sarvam ${res.status}: ${(await res.text()).slice(0, 300)}`), { status: res.status });
  const data = await res.json();
  if (!data.audios?.[0]) throw new Error("sarvam returned no audio");
  return { audio: Buffer.from(data.audios[0], "base64"), type: "audio/wav" };
}

// voice ids: "sarvam:<speaker>" or "eleven:<voice id>"; with no match, use whichever service has a key
async function speak({ text, tone, voice = "", lang }) {
  const t = VOICE_BY_TONE[tone] ? tone : "neutral";
  const [kind, id] = voice.includes(":") ? voice.split(":") : ["", voice];
  const hasSarvam = Boolean(process.env.SARVAM_API_KEY);
  const hasEleven = Boolean(process.env.ELEVENLABS_API_KEY);
  const provider = kind === "sarvam" && hasSarvam ? "sarvam" : kind === "eleven" && hasEleven ? "eleven" : hasSarvam ? "sarvam" : "eleven";
  if (provider === "sarvam") return sarvam({ text, tone: t, speaker: kind === "sarvam" ? id : SARVAM_DEFAULT, lang });
  return { audio: await elevenlabsSpeak({ text, tone: t, voice: kind === "eleven" ? id : "" }), type: "audio/mpeg" };
}

async function elevenlabsSpeak({ text, tone: t, voice }) {
  const v = /^[A-Za-z0-9]{16,32}$/.test(voice || "") ? voice : ELEVEN_VOICE;
  try {
    return await elevenlabs({ text, model: ELEVEN_MODEL, voice: v, tone: t });
  } catch (err) {
    // v3 not available on this account/plan: retry on the multilingual model with voice settings instead of tags
    if (ELEVEN_MODEL !== "eleven_multilingual_v2" && err.status >= 400 && err.status < 500 && err.status !== 401) {
      console.warn(`[speak] ${ELEVEN_MODEL} failed (${err.status}), retrying with eleven_multilingual_v2`);
      return elevenlabs({ text, model: "eleven_multilingual_v2", voice: v, tone: t });
    }
    throw err;
  }
}

// ---------- in-app key setup ----------

const CONFIG_KEYS = { groq: "GROQ_API_KEY", gemini: "GEMINI_API_KEY", claude: "ANTHROPIC_API_KEY", elevenlabs: "ELEVENLABS_API_KEY", sarvam: "SARVAM_API_KEY" };

// Cheap authenticated calls that confirm a key works before it's saved.
async function verifyKey(provider, key) {
  const checks = {
    groq: () => fetch("https://api.groq.com/openai/v1/models", { headers: { authorization: `Bearer ${key}` } }),
    gemini: () => fetch("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1", { headers: { "x-goog-api-key": key } }),
    claude: () => fetch("https://api.anthropic.com/v1/models?limit=1", { headers: { "x-api-key": key, "anthropic-version": "2023-06-01" } }),
  };
  if (!checks[provider]) return true; // voice keys (ElevenLabs can be tightly scoped, Sarvam has no free read endpoint): checked on first use
  const res = await checks[provider]().catch(() => null);
  if (!res) throw Object.assign(new Error("Couldn't reach the provider. Check your internet connection."), { status: 502 });
  if (res.status === 401 || res.status === 403 || res.status === 400) throw Object.assign(new Error("That key was rejected. Copy it again and paste the whole key."), { status: 400 });
  return true;
}

async function saveEnv(name, value) {
  let lines = [];
  try {
    lines = (await readFile(ENV_FILE, "utf8")).split("\n");
  } catch {
    try {
      lines = (await readFile(path.join(here, ".env.example"), "utf8")).split("\n");
    } catch {}
  }
  const line = `${name}=${value}`;
  const i = lines.findIndex((l) => l.startsWith(`${name}=`));
  if (i >= 0) lines[i] = line;
  else lines.push(line);
  await writeFile(ENV_FILE, lines.join("\n").replace(/\n*$/, "\n"), { mode: 0o600 });
  if (value) process.env[name] = value;
  else delete process.env[name];
}

// Only this app's own page may change keys (blocks other websites from posting to localhost).
function sameOrigin(req) {
  const origin = req.headers.origin;
  return !origin || origin === `http://localhost:${PORT}` || origin === `http://127.0.0.1:${PORT}`;
}

// ---------- http ----------

function sendJson(res, status, body) {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

async function readJson(req, limit = 256 * 1024) {
  let size = 0;
  const chunks = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw Object.assign(new Error("request too large"), { status: 413 });
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
}

async function serveStatic(req, res) {
  const url = new URL(req.url, "http://localhost");
  const rel = decodeURIComponent(url.pathname === "/" ? "/index.html" : url.pathname);
  const file = path.normalize(path.join(PUBLIC_DIR, rel));
  if (!file.startsWith(PUBLIC_DIR + path.sep)) return sendJson(res, 403, { error: "forbidden" });
  try {
    const body = await readFile(file);
    res.writeHead(200, { "content-type": MIME[path.extname(file)] || "application/octet-stream", "cache-control": "no-cache" });
    res.end(body);
  } catch {
    sendJson(res, 404, { error: "not found" });
  }
}

const server = http.createServer(async (req, res) => {
  try {
    if (req.method === "GET" && req.url === "/api/status") {
      const [backend, ollama] = await Promise.all([pickBackend(), ollamaStatus()]);
      return sendJson(res, 200, {
        llm: backend,
        llmModel: MODEL_FOR[backend]?.() ?? null,
        vision: backend !== "rules" && (backend !== "ollama" || OLLAMA_SEES),
        ollama: { ...ollama, model: OLLAMA_MODEL },
        keys: Object.fromEntries(Object.entries(CONFIG_KEYS).map(([k, v]) => [k, Boolean(process.env[v])])),
        claude: Boolean(process.env.ANTHROPIC_API_KEY),
        voice: process.env.SARVAM_API_KEY ? "sarvam" : process.env.ELEVENLABS_API_KEY ? "elevenlabs" : "browser",
        voices: { sarvam: Boolean(process.env.SARVAM_API_KEY), elevenlabs: Boolean(process.env.ELEVENLABS_API_KEY) },
        voiceModel: process.env.SARVAM_API_KEY ? "bulbul:v3" : process.env.ELEVENLABS_API_KEY ? ELEVEN_MODEL : null,
        defaultVoice: ELEVEN_VOICE,
      });
    }
    if (req.method === "POST" && req.url === "/api/translate") {
      const payload = await readJson(req);
      if (!Array.isArray(payload.signs) || payload.signs.length === 0) return sendJson(res, 400, { error: "no signs" });
      return sendJson(res, 200, await translate(payload));
    }
    if (req.method === "POST" && req.url === "/api/to-sign") {
      const payload = await readJson(req);
      if (!payload.text) return sendJson(res, 400, { error: "no text" });
      return sendJson(res, 200, await toSign(payload));
    }
    if (req.method === "POST" && req.url === "/api/identify") {
      const payload = await readJson(req, 3 * 1024 * 1024);
      return sendJson(res, 200, await identify(payload));
    }
    if (req.method === "POST" && req.url === "/api/describe") {
      const payload = await readJson(req, 3 * 1024 * 1024);
      if (!payload.label) return sendJson(res, 400, { error: "no label" });
      return sendJson(res, 200, await describe(payload));
    }
    if (req.method === "POST" && req.url === "/api/config") {
      if (!sameOrigin(req)) return sendJson(res, 403, { error: "forbidden" });
      const { provider, key } = await readJson(req);
      const name = CONFIG_KEYS[provider];
      if (!name) return sendJson(res, 400, { error: "unknown provider" });
      const value = String(key || "").trim();
      if (value && !/^[\w.\-:]{16,256}$/.test(value)) return sendJson(res, 400, { error: "That doesn't look like an API key." });
      if (value) await verifyKey(provider, value);
      await saveEnv(name, value);
      console.log(`[config] ${value ? "saved" : "removed"} ${name}`);
      return sendJson(res, 200, { ok: true, llm: await pickBackend() });
    }
    if (req.method === "POST" && req.url === "/api/speak") {
      if (!process.env.ELEVENLABS_API_KEY && !process.env.SARVAM_API_KEY) return sendJson(res, 501, { error: "no voice key; use browser voice" });
      const { text, tone, voice, lang } = await readJson(req);
      if (!text) return sendJson(res, 400, { error: "no text" });
      const { audio, type } = await speak({ text: String(text).slice(0, 1200), tone, voice: String(voice || ""), lang });
      res.writeHead(200, { "content-type": type });
      return res.end(audio);
    }
    if (req.method === "GET") return serveStatic(req, res);
    sendJson(res, 405, { error: "method not allowed" });
  } catch (err) {
    console.error("[server]", err);
    sendJson(res, err.status || 500, { error: err.message });
  }
});

server.listen(PORT, "127.0.0.1", async () => {
  const backend = await pickBackend();
  console.log(`\n  SignBridge running at http://localhost:${PORT}`);
  console.log(`  translator: ${backend}   voice: ${process.env.ELEVENLABS_API_KEY ? `ElevenLabs (${ELEVEN_MODEL})` : "browser"}\n`);
});
