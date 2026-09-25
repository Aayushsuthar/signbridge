// SignBridge server: serves the app, translates signs and describes objects with an LLM,
// and turns text into expressive speech. No framework, one dependency.
import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { translateWithRules } from "./lib/rules.mjs";
import { wikiSummary } from "./lib/wiki.mjs";
import {
  TONES,
  LANGS,
  TRANSLATE_SYSTEM,
  TRANSLATE_SCHEMA,
  buildTranslatePrompt,
  DESCRIBE_SYSTEM,
  DESCRIBE_SCHEMA,
  buildDescribePrompt,
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

async function pickBackend() {
  const wanted = (process.env.LLM || "auto").toLowerCase();
  if (wanted === "rules") return "rules";
  if (wanted === "claude") return process.env.ANTHROPIC_API_KEY ? "claude" : "rules";
  const ollama = await ollamaStatus();
  if (wanted === "ollama") return ollama.running && ollama.hasModel ? "ollama" : "rules";
  if (ollama.running && ollama.hasModel) return "ollama";
  if (process.env.ANTHROPIC_API_KEY) return "claude";
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

async function ask(backend, req) {
  return backend === "ollama" ? askOllama(req) : askClaude(req);
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
      const sees = backend === "claude" || OLLAMA_SEES;
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

async function speak({ text, tone, voice }) {
  const t = VOICE_BY_TONE[tone] ? tone : "neutral";
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
        llmModel: backend === "ollama" ? OLLAMA_MODEL : backend === "claude" ? CLAUDE_MODEL : null,
        vision: backend === "claude" || (backend === "ollama" && OLLAMA_SEES),
        ollama: { ...ollama, model: OLLAMA_MODEL },
        claude: Boolean(process.env.ANTHROPIC_API_KEY),
        voice: process.env.ELEVENLABS_API_KEY ? "elevenlabs" : "browser",
        voiceModel: process.env.ELEVENLABS_API_KEY ? ELEVEN_MODEL : null,
        defaultVoice: ELEVEN_VOICE,
      });
    }
    if (req.method === "POST" && req.url === "/api/translate") {
      const payload = await readJson(req);
      if (!Array.isArray(payload.signs) || payload.signs.length === 0) return sendJson(res, 400, { error: "no signs" });
      return sendJson(res, 200, await translate(payload));
    }
    if (req.method === "POST" && req.url === "/api/describe") {
      const payload = await readJson(req, 3 * 1024 * 1024);
      if (!payload.label) return sendJson(res, 400, { error: "no label" });
      return sendJson(res, 200, await describe(payload));
    }
    if (req.method === "POST" && req.url === "/api/speak") {
      if (!process.env.ELEVENLABS_API_KEY) return sendJson(res, 501, { error: "no ElevenLabs key; use browser voice" });
      const { text, tone, voice } = await readJson(req);
      if (!text) return sendJson(res, 400, { error: "no text" });
      const audio = await speak({ text: String(text).slice(0, 1200), tone, voice });
      res.writeHead(200, { "content-type": "audio/mpeg" });
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
