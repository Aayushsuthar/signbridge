import { startBackground } from "./bg-shader.js";
import { createTrackers, Overlay, FeatureExtractor } from "./vision.js";
import { SignClassifier } from "./classifier.js";
import { FaceAnalyzer, MARKER_LABEL } from "./face.js";
import { speak, stopSpeaking, createListener, VOICES } from "./speech.js";
import { markChanges } from "./diff.js";
import { createObjectDetector, ObjectTracker, boxToStage, cropObject } from "./objects.js";

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const n = Object.assign(document.createElement(tag), props);
  n.append(...children);
  return n;
};

// Suggested starter vocabulary: enough for plenty of everyday sentences.
const VOCAB = [
  "HELLO", "ME", "YOU", "MY", "NAME", "WHAT", "WHERE", "HOW", "WHO", "HAPPY", "SAD", "TIRED", "HUNGRY",
  "WANT", "EAT", "DRINK", "GO", "STORE", "HOME", "WORK", "SCHOOL", "FINISH", "NOT", "YES", "NO", "LIKE",
  "LOVE", "THANK-YOU", "PLEASE", "SORRY", "HELP", "FRIEND", "TODAY", "TOMORROW", "YESTERDAY", "GOOD", "BAD", "_rest",
];
const EXAMPLES = ["STORE ME GO YESTERDAY", "YOU HUNGRY", "ME FINISH WORK", "YOUR NAME WHAT", "ME LIKE SCHOOL", "FRIEND HELP ME"];
const VOICE_SAMPLE = {
  en: (n) => `[cheerful] Hi! I'm ${n}. I'll be your voice today.`,
  hi: (n) => `[cheerful] नमस्ते! मैं ${n} हूँ। आज मैं आपकी आवाज़ बनूँगी।`,
  hinglish: (n) => `[cheerful] Hi! Main ${n} hoon, aaj main aapki awaaz banungi.`,
};
const ANNOUNCE = {
  en: (l) => `I see a ${l}.`,
  hi: (l) => `मुझे ${l} दिख रहा है।`,
  hinglish: (l) => `Mujhe ${l} dikh raha hai.`,
};

const PROVIDERS = [
  { id: "groq", name: "Groq", tag: "Free · fastest", how: "Sign up with email, then create a key at", link: "https://console.groq.com/keys", ph: "gsk_…" },
  { id: "gemini", name: "Google Gemini", tag: "Free · Google account", how: "Sign in with Google and click “Create API key” at", link: "https://aistudio.google.com/apikey", ph: "AIza…" },
  { id: "elevenlabs", name: "ElevenLabs voice", tag: "Free tier", how: "Expressive female voice. Create a key at", link: "https://elevenlabs.io/app/settings/api-keys", ph: "sk_…" },
  { id: "claude", name: "Claude", tag: "Paid", how: "Best quality. Create a key at", link: "https://console.anthropic.com/settings/keys", ph: "sk-ant-…" },
];
const LLM_NAME = { ollama: "Ollama", claude: "Claude", groq: "Groq", gemini: "Gemini" };

// recognition tuning
const WINDOW = 12; // frames of predictions considered
const NEED = 8; // agreeing frames needed to commit a sign
const RELEASE = 6; // frames without the locked sign before it can be signed again
const RECORD_FRAMES = 45;

const classifier = new SignClassifier();
const face = new FaceAnalyzer();
const features = new FeatureExtractor();
const objTracker = new ObjectTracker();

const state = {
  mode: "sign",
  tab: "chat",
  trackers: null,
  objectDetector: null,
  loadingObjects: false,
  overlay: null,
  running: false,
  frame: 0,
  lastVideoTime: -1,
  preds: [],
  locked: null,
  handsGoneAt: null,
  signs: [],
  translating: false,
  lastResult: null,
  transcript: [],
  recording: null,
  practice: { active: false, target: null, score: 0, streak: 0 },
  objects: { visible: [], selected: null, announced: new Map(), lastPanel: 0, describing: 0 },
  speaking: false,
  status: null,
  opts: { auto: true, speak: true, mesh: true, announce: false, pause: 1600, lang: "en", voice: VOICES[0].id },
};

const bg = startBackground($("bg"));
let hero3d = null;

// ---------- utilities ----------

let toastTimer;
function toast(msg, ms = 2800) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  t.style.animation = "none";
  void t.offsetWidth;
  t.style.animation = "";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), ms);
}

function download(name, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  el("a", { href: url, download: name }).click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function loadOpts() {
  try {
    Object.assign(state.opts, JSON.parse(localStorage.getItem("signbridge.opts") || "{}"));
  } catch {}
  for (const [id, key] of [["opt-auto", "auto"], ["opt-speak", "speak"], ["opt-mesh", "mesh"], ["opt-announce", "announce"]]) $(id).checked = state.opts[key];
  $("opt-pause").value = state.opts.pause;
  $("opt-pause-val").textContent = `${(state.opts.pause / 1000).toFixed(1)}s`;
  if (state.opts.tolerance) classifier.tolerance = state.opts.tolerance;
  $("tolerance").value = classifier.tolerance;
  $("tolerance-val").textContent = classifier.tolerance.toFixed(1);
}
function saveOpts() {
  try {
    localStorage.setItem("signbridge.opts", JSON.stringify({ ...state.opts, tolerance: classifier.tolerance }));
  } catch {}
}

function say(text, tone, extra = {}) {
  if (!state.opts.speak && !extra.force) return;
  return speak(text, {
    tone,
    lang: extra.lang || state.opts.lang,
    voice: state.opts.voice,
    onStart: () => setSpeaking(true),
    onEnd: () => setSpeaking(false),
    onLevel: pushLevel,
  });
}

// ---------- segmented controls ----------

function positionPill(seg) {
  const btn = seg.querySelector('[aria-selected="true"]');
  const pill = seg.querySelector(".seg-pill");
  if (!btn || !pill) return;
  pill.style.width = `${btn.offsetWidth}px`;
  pill.style.transform = `translateX(${btn.offsetLeft}px)`;
}
function selectIn(container, attr, value) {
  container.querySelectorAll(`[${attr}]`).forEach((b) => b.setAttribute("aria-selected", String(b.getAttribute(attr) === value)));
  if (container.classList.contains("seg")) positionPill(container);
}

// ---------- hero ----------

function initHero() {
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        e.target.classList.add("in");
        io.unobserve(e.target);
        if (e.target.classList.contains("stats")) countUp();
      }
    },
    { threshold: 0.15 },
  );
  // stagger siblings that enter together
  document.querySelectorAll(".hero .reveal").forEach((n, i) => {
    const siblings = [...n.parentElement.children].filter((c) => c.classList.contains("reveal"));
    n.style.setProperty("--d", `${siblings.indexOf(n) * 0.09}s`);
    io.observe(n);
  });

  // 3D tilt on feature cards
  document.querySelectorAll(".tilt").forEach((card) => {
    card.addEventListener("pointermove", (e) => {
      const r = card.getBoundingClientRect();
      card.style.setProperty("--ry", `${((e.clientX - r.left) / r.width - 0.5) * 16}deg`);
      card.style.setProperty("--rx", `${-((e.clientY - r.top) / r.height - 0.5) * 16}deg`);
    });
    card.addEventListener("pointerleave", () => {
      card.style.setProperty("--rx", "0deg");
      card.style.setProperty("--ry", "0deg");
    });
  });

  import("./hero3d.js")
    .then(({ startHero3D }) => (hero3d = startHero3D($("hero-3d"))))
    .catch((err) => console.warn("3D hero unavailable", err));
}

function countUp() {
  document.querySelectorAll("[data-count]").forEach((n) => {
    const target = Number(n.dataset.count);
    const t0 = performance.now();
    const step = (now) => {
      const k = Math.min(1, (now - t0) / 1400);
      n.textContent = Math.round(target * (1 - Math.pow(1 - k, 3)));
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

// ---------- status ----------

async function refreshStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    state.status = s;
    const llm = $("status-llm");
    llm.lastChild.textContent = s.llm === "rules" ? "Offline grammar" : `${LLM_NAME[s.llm]} · ${s.llmModel}`;
    llm.className = `pill ${s.llm === "rules" ? "warn" : "ok"}`;
    llm.title =
      s.llm === "rules"
        ? s.ollama.running
          ? `Ollama is running but "${s.ollama.model}" isn't pulled. Run: ollama pull ${s.ollama.model}`
          : "No AI connected. Add a free Groq or Gemini key in ⚙ Settings."
        : `Translator${s.vision ? " (can see objects)" : ""}`;
    const voice = $("status-voice");
    voice.lastChild.textContent = s.voice === "elevenlabs" ? "ElevenLabs" : "Browser voice";
    voice.className = `pill ${s.voice === "elevenlabs" ? "ok" : ""}`;
    $("voice-hint").textContent =
      s.voice === "elevenlabs"
        ? `ElevenLabs · ${s.voiceModel}${s.voiceModel?.startsWith("eleven_v3") ? " (emotion tags on)" : ""}. Tap a voice to hear it.`
        : "Add a free ElevenLabs key above to use these expressive voices. Until then your browser's best female voice speaks.";
    $("setup-nudge").hidden = s.llm !== "rules";
    renderProviders(s);
  } catch {
    $("status-llm").lastChild.textContent = "Server offline";
    $("status-llm").className = "pill warn";
  }
}

// ---------- camera + loop ----------

async function start() {
  const btn = $("start-btn");
  btn.disabled = true;
  btn.querySelector("span").textContent = "Loading models…";
  $("start-msg").textContent = "Loading hand + face models. The first time takes a few seconds.";
  try {
    const [stream, trackers] = await Promise.all([
      navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false }),
      createTrackers(),
    ]);
    const video = $("video");
    video.srcObject = stream;
    await video.play();
    state.trackers = trackers;
    state.overlay = new Overlay($("overlay"));
    state.running = true;

    // hero → app
    document.body.classList.add("leaving");
    await new Promise((r) => setTimeout(r, 650));
    hero3d?.stop();
    $("hero").hidden = true;
    $("app").hidden = false;
    document.body.classList.remove("leaving", "mode-hero");
    document.body.classList.add("mode-app");
    scrollTo(0, 0);
    bg.setGlow(0.5);
    requestAnimationFrame(() => document.querySelectorAll(".seg").forEach(positionPill));
    document.fonts?.ready.then(() => document.querySelectorAll(".seg").forEach(positionPill));

    const pill = $("status-vision");
    pill.lastChild.textContent = `Vision · ${trackers.delegate}`;
    pill.className = "pill ok";
    requestAnimationFrame(loop);
    if (!classifier.samples.length) toast("No signs trained yet. Open Train to record your first few, or try 🔍 Objects.", 5500);
  } catch (err) {
    console.error(err);
    btn.disabled = false;
    btn.querySelector("span").textContent = "Start camera";
    $("start-msg").textContent =
      err.name === "NotAllowedError" ? "Camera permission was denied. Allow it in your browser and try again." : `Couldn't start: ${err.message}`;
  }
}

let fpsFrames = 0;
let fpsSince = performance.now();

function loop() {
  if (!state.running) return;
  const video = $("video");
  if (video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    const now = performance.now();
    state.frame++;
    state.overlay.resize(video.videoWidth, video.videoHeight);
    if (state.mode === "objects") objectsFrame(video, now);
    else signFrame(video, now);

    fpsFrames++;
    if (now - fpsSince > 1000) {
      $("fps").textContent = `${fpsFrames} fps`;
      fpsFrames = 0;
      fpsSince = now;
    }
  }
  requestAnimationFrame(loop);
}

function signFrame(video, now) {
  const hands = state.trackers.hands.detectForVideo(video, now);
  const faceRes = state.trackers.face.detectForVideo(video, now);
  const handCount = hands.landmarks?.length || 0;
  face.collecting = handCount > 0 || state.signs.length > 0;
  const faceState = face.update(faceRes, now);
  const { vec, count } = features.frame(hands, faceRes, video.videoWidth / video.videoHeight);
  state.overlay.render(hands, faceRes, { showFace: state.opts.mesh });
  if (state.recording) recordFrame(vec, count, now);
  else recognise(vec, count, now);
  renderFace(faceState);
}

// ---------- mode switching ----------

async function setMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode;
  selectIn($("mode-seg"), "data-mode", mode);
  document.body.classList.toggle("objects", mode === "objects");
  state.overlay?.clear();
  $("hud-pause").hidden = true;
  const ph = $("result-text").querySelector(".placeholder");
  if (ph) ph.textContent = mode === "objects" ? "Tap any object to hear what it is." : "Start signing. Your words appear here.";
  if (mode === "objects") {
    $("pred-kicker").textContent = "Objects";
    $("hud-face").hidden = true;
    setTab("objects");
    if (!state.objectDetector && !state.loadingObjects) {
      state.loadingObjects = true;
      setPred("Loading…", true);
      try {
        state.objectDetector = await createObjectDetector();
        toast("Object recognition ready. Tap anything to learn about it.");
      } catch (err) {
        console.error(err);
        toast(`Couldn't load the object model: ${err.message}`, 5000);
      } finally {
        state.loadingObjects = false;
      }
    }
  } else {
    $("pred-kicker").textContent = "Sign";
    $("hud-face").hidden = false;
    objTracker.clear();
    renderObjects([]);
    if (state.tab === "objects") setTab("chat");
  }
}

// ---------- objects ----------

function objectsFrame(video, now) {
  if (!state.objectDetector) return;
  if (state.frame % 2 === 0) {
    const res = state.objectDetector.detectForVideo(video, now);
    state.objects.visible = objTracker.update(res, now);
  }
  renderObjects(state.objects.visible);
  const n = state.objects.visible.length;
  const top = [...state.objects.visible].sort((a, b) => b.score - a.score)[0];
  setPred(top ? top.label : "Looking…", !top, top ? `${n} object${n === 1 ? "" : "s"} · tap one` : "point the camera at something");
  $("pred-bar").style.width = top ? `${Math.round(top.score * 100)}%` : "0";

  if (now - state.objects.lastPanel > 300) {
    state.objects.lastPanel = now;
    renderObjectList();
    if (state.opts.announce) announceNew(now);
  }
}

const objEls = new Map();
function renderObjects(tracks) {
  const layer = $("obj-layer");
  const rect = $("stage").getBoundingClientRect();
  const video = $("video");
  const live = new Set();
  for (const t of tracks) {
    live.add(t.id);
    let node = objEls.get(t.id);
    if (!node) {
      node = el("div", { className: "obj", title: `Tap to learn about this ${t.label}` }, el("span", { className: "obj-tag" }));
      node.onclick = () => describe(t);
      layer.append(node);
      objEls.set(t.id, node);
    }
    const b = boxToStage(t.box, video, rect);
    node.style.transform = `translate(${b.left}px, ${b.top}px)`;
    node.style.width = `${b.width}px`;
    node.style.height = `${b.height}px`;
    node.classList.toggle("selected", state.objects.selected === t.id);
    const tag = node.firstChild;
    const pct = Math.round(t.score * 100);
    if (tag.dataset.v !== `${t.label}${pct}`) {
      tag.dataset.v = `${t.label}${pct}`;
      tag.replaceChildren(t.label, el("b", { textContent: `${pct}%` }));
    }
  }
  for (const [id, node] of objEls) {
    if (!live.has(id)) {
      node.remove();
      objEls.delete(id);
    }
  }
}

function renderObjectList() {
  const labels = [...new Map(state.objects.visible.map((t) => [t.label, t])).values()];
  $("obj-count").textContent = labels.length ? `${labels.length} in view` : "";
  const list = $("obj-list");
  const key = labels.map((t) => t.label).join("|");
  if (list.dataset.key === key) return;
  list.dataset.key = key;
  list.replaceChildren(
    ...labels.map((t) => {
      const b = el("button", { type: "button", textContent: t.label });
      b.onclick = () => describe(state.objects.visible.find((v) => v.label === t.label) || t);
      return b;
    }),
  );
}

function announceNew(now) {
  if (state.speaking) return;
  for (const t of state.objects.visible) {
    const last = state.objects.announced.get(t.label) || 0;
    if (t.hits >= 10 && now - last > 25000) {
      state.objects.announced.set(t.label, now);
      say(ANNOUNCE[state.opts.lang](t.label), "calm", { force: true });
      return;
    }
  }
}

async function describe(track) {
  const video = $("video");
  const req = ++state.objects.describing;
  state.objects.selected = track.id;
  setTab("objects");
  const { base64, preview } = cropObject(video, track.box);
  const card = $("obj-card");
  card.hidden = false;
  card.classList.add("loading");
  card.style.animation = "none";
  void card.offsetWidth;
  card.style.animation = "";
  $("obj-img").src = preview;
  $("obj-label").textContent = `Detected · ${track.label} · ${Math.round(track.score * 100)}%`;
  $("obj-name").textContent = track.label;
  $("obj-text").replaceChildren(el("span", { className: "shimmer" }), el("span", { className: "shimmer", style: "width:80%" }));
  $("obj-facts").replaceChildren();
  $("obj-meta").textContent = "";
  $("obj-wiki").hidden = true;
  try {
    const res = await fetch("/api/describe", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ label: track.label, score: track.score, image: base64, lang: state.opts.lang }),
    });
    const data = await res.json();
    if (req !== state.objects.describing) return; // user tapped something else meanwhile
    if (!res.ok) throw new Error(data.error || res.statusText);
    card.classList.remove("loading");
    const lang = data.lang === "hi" ? "hi" : "en";
    $("obj-name").textContent = data.name;
    $("obj-name").lang = lang;
    $("obj-text").textContent = data.text;
    $("obj-text").lang = lang;
    $("obj-facts").replaceChildren(...data.facts.map((f) => el("li", { textContent: f, lang })));
    $("obj-meta").textContent = `${data.backend}${data.sawImage ? " · saw the photo" : ""} · ${(data.ms / 1000).toFixed(1)}s`;
    if (data.wiki?.url) {
      $("obj-wiki").href = data.wiki.url;
      $("obj-wiki").hidden = false;
    }
    $("obj-speak").onclick = () => say(data.speech, data.tone, { force: true, lang: data.lang });
    say(data.speech, data.tone, { lang: data.lang });
  } catch (err) {
    card.classList.remove("loading");
    $("obj-text").textContent = `Couldn't describe it: ${err.message}`;
  }
}

// ---------- sign recognition ----------

function recognise(vec, handCount, now) {
  const pred = handCount ? classifier.predict(vec) : null;
  state.preds.push(pred?.accepted ? pred.label : null);
  if (state.preds.length > WINDOW) state.preds.shift();
  renderPrediction(pred, handCount);

  // hands down: release the lock and, after a pause, finish the sentence
  if (!handCount) {
    state.locked = null;
    if (!state.signs.length) face.resetSentence(); // idle: start the next sentence's face summary fresh
    state.handsGoneAt ??= now;
    const waited = now - state.handsGoneAt;
    const canAuto = state.opts.auto && state.tab !== "practice" && state.signs.length && !state.translating;
    $("hud-pause").hidden = !canAuto;
    if (canAuto) {
      $("pause-ring").style.strokeDashoffset = String(94.25 * (1 - Math.min(waited / state.opts.pause, 1)));
      if (waited >= state.opts.pause) finishSentence();
    }
    return;
  }
  state.handsGoneAt = null;
  $("hud-pause").hidden = true;

  const counts = new Map();
  for (const l of state.preds) if (l) counts.set(l, (counts.get(l) || 0) + 1);
  let top = null;
  let topN = 0;
  for (const [l, n] of counts) if (n > topN) [top, topN] = [l, n];

  if (state.locked && !state.preds.slice(-RELEASE).includes(state.locked)) state.locked = null;
  if (top && topN >= NEED && top !== state.locked) {
    state.locked = top;
    if (!top.startsWith("_")) commitSign(top);
  }
}

function commitSign(gloss) {
  const label = $("pred-label");
  label.classList.remove("pop");
  void label.offsetWidth;
  label.classList.add("pop");
  if (state.tab === "practice") return practiceCheck(gloss);
  if (state.tab === "train") return;
  state.signs.push({ gloss, markers: face.recentMarkers() });
  renderSigns();
}

function setPred(text, dim, sub = "") {
  const label = $("pred-label");
  if (label.textContent !== text) label.textContent = text;
  label.classList.toggle("dim", dim);
  $("pred-sub").textContent = sub;
}

function renderPrediction(pred, handCount) {
  if (!classifier.samples.length) {
    setPred("Train signs first", true);
    $("pred-bar").style.width = "0";
    return;
  }
  if (!handCount || !pred) {
    setPred(handCount ? "…" : "No hands", true);
    $("pred-bar").style.width = "0";
    return;
  }
  setPred(pred.accepted ? pred.label : `${pred.label}?`, !pred.accepted, `conf ${pred.confidence.toFixed(2)} · dist ${pred.distance.toFixed(2)}/${pred.limit.toFixed(2)}`);
  $("pred-bar").style.width = `${Math.round(pred.confidence * 100)}%`;
}

let lastFaceKey = "";
function renderFace(fs) {
  const key = `${fs.present}|${fs.emotion}|${fs.markers.join()}`;
  if (key === lastFaceKey) return;
  lastFaceKey = key;
  $("orb").dataset.emotion = fs.present ? fs.emotion : "none";
  $("face-emotion").textContent = fs.present ? fs.emotion : "no face";
  $("face-markers").replaceChildren(...fs.markers.map((m) => el("span", { className: "chip", textContent: MARKER_LABEL[m] })));
}

function renderSigns() {
  $("signs").replaceChildren(
    ...state.signs.map((s) => {
      const chip = el("span", { className: "sign", textContent: s.gloss });
      if (s.markers.length) chip.append(el("small", { textContent: s.markers.map((m) => MARKER_LABEL[m].split(" · ")[1]).join(", ") }));
      return chip;
    }),
  );
}

// ---------- translation ----------

async function finishSentence(signs = state.signs, faceSummary = face.summary()) {
  if (!signs.length || state.translating) return;
  state.translating = true;
  $("hud-pause").hidden = true;
  const text = $("result-text");
  text.className = "caption loading";
  text.replaceChildren(el("span", { className: "dots" }, el("span"), el("span"), el("span")));
  $("result-meta").textContent = "";
  $("result-notes").textContent = "";
  $("replay").hidden = true;
  const glosses = signs.map((s) => s.gloss);
  try {
    const res = await fetch("/api/translate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        signs,
        face: faceSummary,
        lang: state.opts.lang,
        history: state.transcript.filter((t) => t.who === "signer").slice(-3).map((t) => t.text),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    state.lastResult = data;
    renderResult(data, glosses, faceSummary);
    addTranscript({ who: "signer", text: data.text, glosses, tone: data.tone, lang: data.lang });
    say(data.speech, data.tone, { lang: data.lang });
  } catch (err) {
    text.className = "caption";
    text.textContent = `Translation failed: ${err.message}`;
  } finally {
    state.signs = [];
    state.translating = false;
    face.resetSentence();
    renderSigns();
  }
}

function renderResult(data, glosses, faceSummary) {
  const text = $("result-text");
  text.className = "caption";
  text.lang = data.lang === "hi" ? "hi" : "en";
  // underline what the AI added or changed (only meaningful when the output is English)
  const parts = data.lang === "en" ? markChanges(data.text, glosses) : data.text.split(/(\s+)/).map((t) => ({ text: t, changed: false }));
  let i = 0;
  text.replaceChildren(
    ...parts.map((p) => {
      if (!p.text.trim()) return p.text;
      const w = el("span", { className: `word${p.changed ? " changed" : ""}`, textContent: p.text });
      w.style.setProperty("--i", i++);
      return w;
    }),
  );
  $("result-meta").replaceChildren(
    el("span", { className: "tone", textContent: data.tone }),
    `${glosses.join(" ")} · ${data.backend}${data.fallbackFrom ? ` (${data.fallbackFrom} failed)` : ""} · ${data.ms} ms`,
  );
  $("result-notes").textContent = data.notes || "";
  $("replay").hidden = false;
}

function addTranscript(entry) {
  state.transcript.push({ ...entry, at: new Date() });
  const list = $("transcript");
  list.querySelector(".empty")?.remove();
  const who = entry.who === "signer" ? `🤟 Signer${entry.tone ? ` · ${entry.tone}` : ""}` : "🗣️ Speaker";
  list.append(el("li", { className: entry.who, lang: entry.lang === "hi" ? "hi" : "en" }, el("span", { className: "who", textContent: who }), entry.text));
  list.lastChild.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// ---------- speaking animation ----------

const levels = new Array(28).fill(0);
let waveRaf = null;
function pushLevel(v) {
  levels.push(v);
  levels.shift();
  if (!waveRaf) waveRaf = requestAnimationFrame(drawWave);
}
function setSpeaking(on) {
  state.speaking = on;
  $("stage").classList.toggle("speaking", on);
  if (!waveRaf) waveRaf = requestAnimationFrame(drawWave);
}
function drawWave() {
  const c = $("wave");
  const dpr = devicePixelRatio || 1;
  if (c.width !== 160 * dpr) {
    c.width = 160 * dpr;
    c.height = 28 * dpr;
  }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, 160, 28);
  const grad = g.createLinearGradient(0, 0, 160, 0);
  grad.addColorStop(0, "#5cf2ff");
  grad.addColorStop(0.5, "#8b5cf6");
  grad.addColorStop(1, "#ff4fd8");
  g.fillStyle = grad;
  let active = false;
  levels.forEach((v, i) => {
    if (!state.speaking) levels[i] *= 0.85;
    const h = Math.max(3, levels[i] * 26 * (0.6 + 0.4 * Math.sin(i * 1.7)));
    if (levels[i] > 0.01) active = true;
    g.beginPath();
    g.roundRect(i * 5.7, 14 - h / 2, 3.4, h, 2);
    g.fill();
  });
  waveRaf = state.speaking || active ? requestAnimationFrame(drawWave) : null;
}

// ---------- training ----------

function recordSign() {
  if (!state.running) return toast("Start the camera first.");
  if (state.mode !== "sign") setMode("sign");
  const label = $("label-input").value.trim().toUpperCase().replace(/\s+/g, "-").replace("_REST", "_rest");
  if (!label) return toast("Type a name for the sign first.");
  if (state.recording) return;
  const cd = $("countdown");
  let n = 3;
  const show = (v) => {
    cd.textContent = v;
    cd.classList.remove("tick");
    void cd.offsetWidth;
    cd.classList.add("tick");
  };
  cd.hidden = false;
  cd.className = "countdown";
  show(n);
  state.recording = { label, phase: "countdown", vecs: [] };
  const timer = setInterval(() => {
    n--;
    if (n > 0) show(n);
    else {
      clearInterval(timer);
      cd.className = "countdown rec";
      cd.textContent = `Recording ${label}… hold it`;
      state.recording.phase = "capture";
      state.recording.started = performance.now();
    }
  }, 700);
}

function recordFrame(vec, handCount, now) {
  const r = state.recording;
  renderPrediction(null, handCount);
  if (r.phase !== "capture") return;
  if (handCount) r.vecs.push(vec);
  $("countdown").textContent = `Recording ${r.label}… ${Math.round((r.vecs.length / RECORD_FRAMES) * 100)}%`;
  if (r.vecs.length >= RECORD_FRAMES || now - r.started > 5000) {
    state.recording = null;
    $("countdown").hidden = true;
    if (r.vecs.length < 10) return toast("Couldn't see your hands. Try again with your hands in frame.");
    classifier.add(r.label, r.vecs);
    renderSignList();
    toast(`✓ Saved ${r.vecs.length} samples for ${r.label}`);
  }
}

function renderSignList() {
  const labels = classifier.labels();
  $("sign-count").textContent = labels.length ? `(${labels.length})` : "";
  const list = $("sign-list");
  if (!labels.length) list.replaceChildren(el("li", { className: "empty", textContent: "No signs yet." }));
  else
    list.replaceChildren(
      ...labels.map(([label, n]) => {
        const x = el("button", { className: "x", title: `Delete ${label}`, textContent: "×" });
        x.onclick = () => {
          classifier.remove(label);
          renderSignList();
        };
        return el("li", {}, el("div", {}, el("b", { textContent: label }), el("span", { textContent: `${n} samples` })), x);
      }),
    );
  const have = new Set(labels.map(([l]) => l));
  $("vocab").replaceChildren(
    ...VOCAB.map((w) => {
      const b = el("button", { type: "button", textContent: w, className: have.has(w) ? "have" : "", title: have.has(w) ? "Trained. Click to add more samples." : "Click to use this name" });
      b.onclick = () => {
        $("label-input").value = w;
        $("label-input").focus();
      };
      return b;
    }),
  );
}

// ---------- practice ----------

function practiceNext() {
  const labels = classifier.labels().map(([l]) => l).filter((l) => !l.startsWith("_"));
  if (!labels.length) {
    $("practice-target").textContent = "—";
    return toast("Train some signs first.");
  }
  let next = labels[Math.floor(Math.random() * labels.length)];
  if (labels.length > 1) while (next === state.practice.target) next = labels[Math.floor(Math.random() * labels.length)];
  state.practice.target = next;
  state.practice.active = true;
  state.locked = null;
  const t = $("practice-target");
  t.textContent = next;
  t.classList.remove("win");
  $("practice-feedback").textContent = "";
  $("practice-feedback").className = "practice-feedback";
}

function practiceCheck(gloss) {
  const p = state.practice;
  if (!p.active) return;
  const fb = $("practice-feedback");
  if (gloss === p.target) {
    p.score++;
    p.streak++;
    p.active = false;
    fb.textContent = p.streak >= 3 ? `🔥 ${p.streak} in a row!` : "✓ Nice!";
    fb.className = "practice-feedback good";
    $("practice-target").classList.add("win");
    setTimeout(practiceNext, 1000);
  } else {
    p.streak = 0;
    fb.textContent = `That looked like ${gloss}`;
    fb.className = "practice-feedback";
  }
  $("practice-score").textContent = p.score;
  $("practice-streak").textContent = p.streak;
}

// ---------- settings ----------

let providersKey = "";
function renderProviders(status) {
  const key = JSON.stringify([status.keys, status.llm]);
  if (key === providersKey) return; // don't wipe a key the user is typing on the periodic refresh
  providersKey = key;
  $("providers").replaceChildren(
    ...PROVIDERS.map((pr) => {
      const on = Boolean(status.keys?.[pr.id]);
      const active = status.llm === pr.id;
      const msg = el("div", { className: "msg" });
      const input = el("input", { type: "password", placeholder: on ? "Connected. Paste a new key to replace it." : `Paste key (${pr.ph})`, autocomplete: "off", spellcheck: false });
      const save = el("button", { className: "btn-glass xs", textContent: "Save" });
      const form = el("form", {}, input, save);
      form.onsubmit = async (e) => {
        e.preventDefault();
        if (!input.value.trim()) return;
        save.disabled = true;
        save.textContent = "Checking…";
        msg.className = "msg";
        msg.textContent = "";
        try {
          const res = await fetch("/api/config", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ provider: pr.id, key: input.value.trim() }) });
          const data = await res.json();
          if (!res.ok) throw new Error(data.error || res.statusText);
          input.value = "";
          toast(pr.id === "elevenlabs" ? "✓ ElevenLabs connected. Tap a voice below to hear it." : `✓ ${pr.name} connected. Translations now use ${LLM_NAME[data.llm] || data.llm}.`, 4000);
          providersKey = "";
          await refreshStatus();
        } catch (err) {
          msg.className = "msg err";
          msg.textContent = err.message;
        } finally {
          save.disabled = false;
          save.textContent = "Save";
        }
      };
      const head = el("div", { className: "provider-head" }, el("b", { textContent: pr.name }), el("span", { className: `tag${pr.tag === "Paid" ? " paid" : ""}`, textContent: on ? (active ? "✓ In use" : "✓ Connected") : pr.tag }));
      const how = el("p", {}, `${pr.how} `, el("a", { href: pr.link, target: "_blank", rel: "noopener", textContent: pr.link.replace("https://", "") }), ".");
      const card = el("div", { className: `provider${on ? " on" : ""}` }, head, how, form, msg);
      if (on) {
        const remove = el("button", { type: "button", className: "btn-glass xs danger", textContent: "Remove" });
        remove.onclick = async () => {
          await fetch("/api/config", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ provider: pr.id, key: "" }) });
          providersKey = "";
          refreshStatus();
        };
        form.append(remove);
      }
      return card;
    }),
  );
}

function renderVoices() {
  $("voices").replaceChildren(
    ...VOICES.map((v) => {
      const b = el("button", { type: "button", className: "voice" }, el("b", { textContent: v.name }), el("span", { textContent: v.note }));
      b.setAttribute("aria-pressed", String(state.opts.voice === v.id));
      b.onclick = () => {
        state.opts.voice = v.id;
        saveOpts();
        renderVoices();
        say(VOICE_SAMPLE[state.opts.lang](v.name), "happy", { force: true });
      };
      return b;
    }),
  );
}

// ---------- wiring ----------

function setTab(tab) {
  state.tab = tab;
  selectIn(document.querySelector(".dock-tabs"), "data-tab", tab);
  document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = p.dataset.panel !== tab));
  if (tab === "practice" && !state.practice.target) practiceNext();
}

function bind() {
  $("start-btn").onclick = start;
  document.querySelectorAll(".dock-tabs [data-tab]").forEach((b) => (b.onclick = () => setTab(b.dataset.tab)));
  document.querySelectorAll("#mode-seg [data-mode]").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  document.querySelectorAll("#lang-seg [data-lang]").forEach(
    (b) =>
      (b.onclick = () => {
        state.opts.lang = b.dataset.lang;
        selectIn($("lang-seg"), "data-lang", state.opts.lang);
        listener.setLang(state.opts.lang);
        saveOpts();
        const name = { en: "English", hi: "हिन्दी", hinglish: "Hinglish" }[state.opts.lang];
        toast(`Speaking ${name}${state.status?.llm === "rules" ? " (offline grammar: simple sentences only)" : ""}`);
      }),
  );
  addEventListener("resize", () => document.querySelectorAll(".seg").forEach(positionPill));

  $("undo").onclick = () => {
    state.signs.pop();
    renderSigns();
  };
  $("clear").onclick = () => {
    state.signs = [];
    face.resetSentence();
    renderSigns();
  };
  $("translate-now").onclick = () => finishSentence();
  $("setup-nudge").onclick = () => setTab("settings");
  $("replay").onclick = () => state.lastResult && say(state.lastResult.speech, state.lastResult.tone, { force: true, lang: state.lastResult.lang });

  $("manual-form").onsubmit = (e) => {
    e.preventDefault();
    const glosses = $("manual-input").value.trim().toUpperCase().split(/\s+/).filter(Boolean);
    if (!glosses.length) return;
    const current = face.state.present ? { emotion: face.state.emotion, intensity: face.state.intensity, markers: face.state.markers } : { emotion: "neutral", intensity: 0, markers: [] };
    finishSentence(glosses.map((gloss) => ({ gloss, markers: current.markers })), current);
    $("manual-input").value = "";
  };
  $("examples").replaceChildren(
    ...EXAMPLES.map((ex) => {
      const b = el("button", { type: "button", textContent: ex });
      b.onclick = () => ($("manual-input").value = ex);
      return b;
    }),
  );

  $("export-transcript").onclick = () => {
    if (!state.transcript.length) return toast("Nothing to export yet.");
    const lines = state.transcript.map(
      (t) => `[${t.at.toLocaleTimeString()}] ${t.who === "signer" ? `Signer (${t.glosses.join(" ")})` : "Speaker"}: ${t.text}`,
    );
    download(`signbridge-${new Date().toISOString().slice(0, 10)}.txt`, lines.join("\n"));
  };

  // two-way: caption the hearing person's reply
  listener = createListener({
    onInterim: (t) => ($("caption").textContent = t),
    onFinal: (t) => {
      if (t) addTranscript({ who: "speaker", text: t, lang: state.opts.lang });
      $("caption").textContent = "";
    },
    onState: (on, err) => {
      $("listen").classList.toggle("live", on);
      $("listen").textContent = on ? "■ Stop" : "🎙️ Listen to reply";
      if (err) toast(err);
    },
  });
  listener.setLang(state.opts.lang);
  $("listen").onclick = () => {
    if (!listener.supported) return toast("Speech recognition needs Chrome or Edge.");
    if (listener.active) listener.stop();
    else {
      stopSpeaking();
      listener.start();
      setTab("chat");
    }
  };

  // settings
  for (const [id, key] of [["opt-auto", "auto"], ["opt-speak", "speak"], ["opt-mesh", "mesh"], ["opt-announce", "announce"]]) {
    $(id).onchange = () => {
      state.opts[key] = $(id).checked;
      saveOpts();
    };
  }
  $("opt-pause").oninput = () => {
    state.opts.pause = Number($("opt-pause").value);
    $("opt-pause-val").textContent = `${(state.opts.pause / 1000).toFixed(1)}s`;
    saveOpts();
  };

  // training
  $("record-btn").onclick = recordSign;
  $("label-input").onkeydown = (e) => e.key === "Enter" && recordSign();
  $("export-signs").onclick = () => {
    if (!classifier.samples.length) return toast("No signs to export.");
    download("signbridge-signs.json", JSON.stringify(classifier.export()), "application/json");
  };
  $("import-signs").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      classifier.import(JSON.parse(await file.text()));
      renderSignList();
      toast(`Imported. You now have ${classifier.labels().length} signs.`);
    } catch (err) {
      toast(err.message, 4000);
    }
    e.target.value = "";
  };
  $("clear-signs").onclick = () => {
    if (classifier.samples.length && confirm("Delete all recorded signs? Export first if you want a backup.")) {
      classifier.clear();
      renderSignList();
    }
  };
  $("calibrate-btn").onclick = async () => {
    if (!state.running) return toast("Start the camera first.");
    if (state.mode !== "sign") await setMode("sign");
    const b = $("calibrate-btn");
    b.disabled = true;
    b.textContent = "Hold a relaxed face…";
    await face.calibrate();
    b.disabled = false;
    b.textContent = "🙂 Calibrate face";
    toast("Face calibrated.");
  };
  $("tolerance").oninput = () => {
    classifier.tolerance = Number($("tolerance").value);
    $("tolerance-val").textContent = classifier.tolerance.toFixed(1);
    saveOpts();
  };

  // practice
  $("practice-start").onclick = practiceNext;
  $("practice-skip").onclick = () => {
    state.practice.streak = 0;
    $("practice-streak").textContent = 0;
    practiceNext();
  };

  // keyboard shortcuts (ignored while typing or before the app opens)
  document.addEventListener("keydown", (e) => {
    if (!state.running || e.target.matches("input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === "enter" && state.mode === "sign") {
      e.preventDefault();
      finishSentence();
    } else if (k === "backspace") {
      e.preventDefault();
      $("undo").click();
    } else if (k === "escape") $("clear").click();
    else if (k === "r" && state.tab === "train") recordSign();
    else if (k === "o") setMode(state.mode === "objects" ? "sign" : "objects");
  });
}

let listener;
bind();
loadOpts();
selectIn($("lang-seg"), "data-lang", state.opts.lang);
renderSignList();
renderSigns();
renderVoices();
initHero();
refreshStatus();
setInterval(refreshStatus, 15000);
