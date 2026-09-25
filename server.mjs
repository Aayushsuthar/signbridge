// SignBridge server: serves the app, translates ASL gloss to English with an LLM,
// and turns the English into emotion-aware speech. No framework, one dependency.
import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { translateWithRules } from "./lib/rules.mjs";
import { SYSTEM_PROMPT, OUTPUT_SCHEMA, buildUserPrompt, TONES } from "./lib/prompt.mjs";

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
const CLAUDE_MODEL = process.env.CLAUDE_MODEL || "claude-opus-5";
const ELEVEN_VOICE = process.env.ELEVENLABS_VOICE_ID || "21m00Tcm4TlvDq8ikWAM";
const ELEVEN_MODEL = process.env.ELEVENLABS_MODEL || "eleven_multilingual_v2";

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

// ---------- backends ----------

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

function normalizeResult(raw, fallbackTone) {
  const english = String(raw?.english || "").trim();
  if (!english) throw new Error("model returned no english");
  const tone = TONES.includes(raw?.tone) ? raw.tone : fallbackTone;
  return { english, tone, notes: String(raw?.notes || "") };
}

async function translateWithOllama(payload) {
  const res = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    signal: AbortSignal.timeout(60_000),
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      stream: false,
      format: OUTPUT_SCHEMA,
      options: { temperature: 0.3 },
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: buildUserPrompt(payload) },
      ],
    }),
  });
  if (!res.ok) throw new Error(`ollama ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return JSON.parse(data.message.content);
}

let anthropic;
async function translateWithClaude(payload) {
  if (!anthropic) {
    const { default: Anthropic } = await import("@anthropic-ai/sdk");
    anthropic = new Anthropic();
  }
  const response = await anthropic.beta.messages.create({
    model: CLAUDE_MODEL,
    max_tokens: 1024,
    // If the primary model declines, the API retries on a fallback model in the same call.
    betas: ["server-side-fallback-2026-07-01"],
    fallbacks: "default",
    output_config: {
      effort: "low",
      format: { type: "json_schema", schema: OUTPUT_SCHEMA },
    },
    system: SYSTEM_PROMPT,
    messages: [{ role: "user", content: buildUserPrompt(payload) }],
  });
  if (response.stop_reason === "refusal") throw new Error("claude declined this request");
  const text = response.content.find((b) => b.type === "text")?.text;
  if (!text) throw new Error("claude returned no text");
  return JSON.parse(text);
}

async function translate(payload) {
  const backend = await pickBackend();
  const fallbackTone = payload?.face?.emotion === "happy" ? "happy" : "neutral";
  const started = Date.now();
  if (backend !== "rules") {
    try {
      const raw = backend === "ollama" ? await translateWithOllama(payload) : await translateWithClaude(payload);
      return { ...normalizeResult(raw, fallbackTone), backend, ms: Date.now() - started };
    } catch (err) {
      console.warn(`[translate] ${backend} failed, using rules:`, err.message);
      return { ...translateWithRules(payload), backend: "rules", fallbackFrom: backend, ms: Date.now() - started };
    }
  }
  return { ...translateWithRules(payload), backend, ms: Date.now() - started };
}

// Voice settings per tone. Lower stability = more expressive; higher style = more exaggerated delivery.
const VOICE_BY_TONE = {
  neutral: { stability: 0.6, style: 0.1 },
  calm: { stability: 0.75, style: 0.05 },
  happy: { stability: 0.4, style: 0.45 },
  excited: { stability: 0.28, style: 0.65 },
  sad: { stability: 0.7, style: 0.35 },
  angry: { stability: 0.3, style: 0.7 },
  surprised: { stability: 0.3, style: 0.55 },
  questioning: { stability: 0.5, style: 0.3 },
};
// eleven_v3 understands inline audio tags, which shape delivery more than settings do.
const V3_TAG = { happy: "[cheerful]", excited: "[excited]", sad: "[sad]", angry: "[angry]", surprised: "[surprised]", calm: "[calm]", questioning: "[curious]" };

async function speak({ text, tone }) {
  const t = VOICE_BY_TONE[tone] ? tone : "neutral";
  const input = ELEVEN_MODEL.startsWith("eleven_v3") && V3_TAG[t] ? `${V3_TAG[t]} ${text}` : text;
  const res = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(ELEVEN_VOICE)}?output_format=mp3_44100_128`,
    {
      method: "POST",
      headers: { "xi-api-key": process.env.ELEVENLABS_API_KEY, "content-type": "application/json", accept: "audio/mpeg" },
      signal: AbortSignal.timeout(30_000),
      body: JSON.stringify({
        text: input,
        model_id: ELEVEN_MODEL,
        voice_settings: { ...VOICE_BY_TONE[t], similarity_boost: 0.75, use_speaker_boost: true },
      }),
    },
  );
  if (!res.ok) throw new Error(`elevenlabs ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return Buffer.from(await res.arrayBuffer());
}

// ---------- http ----------

function sendJson(res, status, body) {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

async function readJson(req) {
  let size = 0;
  const chunks = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 256 * 1024) throw new Error("request too large");
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
        ollama: { ...ollama, model: OLLAMA_MODEL },
        claude: Boolean(process.env.ANTHROPIC_API_KEY),
        voice: process.env.ELEVENLABS_API_KEY ? "elevenlabs" : "browser",
      });
    }
    if (req.method === "POST" && req.url === "/api/translate") {
      const payload = await readJson(req);
      if (!Array.isArray(payload.signs) || payload.signs.length === 0) return sendJson(res, 400, { error: "no signs" });
      return sendJson(res, 200, await translate(payload));
    }
    if (req.method === "POST" && req.url === "/api/speak") {
      if (!process.env.ELEVENLABS_API_KEY) return sendJson(res, 501, { error: "no ElevenLabs key; use browser voice" });
      const { text, tone } = await readJson(req);
      if (!text) return sendJson(res, 400, { error: "no text" });
      const audio = await speak({ text: String(text).slice(0, 1000), tone });
      res.writeHead(200, { "content-type": "audio/mpeg" });
      return res.end(audio);
    }
    if (req.method === "GET") return serveStatic(req, res);
    sendJson(res, 405, { error: "method not allowed" });
  } catch (err) {
    console.error("[server]", err);
    sendJson(res, 500, { error: err.message });
  }
});

server.listen(PORT, "127.0.0.1", async () => {
  const backend = await pickBackend();
  console.log(`\n  SignBridge running at http://localhost:${PORT}`);
  console.log(`  translator: ${backend}   voice: ${process.env.ELEVENLABS_API_KEY ? "ElevenLabs" : "browser"}\n`);
});
