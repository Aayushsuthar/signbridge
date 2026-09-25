import { createTrackers, Overlay, FeatureExtractor } from "./vision.js";
import { SignClassifier } from "./classifier.js";
import { FaceAnalyzer, EMOJI, MARKER_LABEL } from "./face.js";
import { speak, stopSpeaking, createListener } from "./speech.js";
import { markChanges } from "./diff.js";

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

// recognition tuning
const WINDOW = 12; // frames of predictions considered
const NEED = 8; // agreeing frames needed to commit a sign
const RELEASE = 6; // frames without the locked sign before it can be signed again
const RECORD_FRAMES = 45;

const classifier = new SignClassifier();
const face = new FaceAnalyzer();
const features = new FeatureExtractor();

const state = {
  tab: "translate",
  trackers: null,
  overlay: null,
  running: false,
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
  opts: { auto: true, speak: true, mesh: true, pause: 1600 },
};

// ---------- utilities ----------

let toastTimer;
function toast(msg, ms = 2600) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
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
  $("opt-auto").checked = state.opts.auto;
  $("opt-speak").checked = state.opts.speak;
  $("opt-mesh").checked = state.opts.mesh;
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

// ---------- status ----------

async function refreshStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    const llm = $("status-llm");
    llm.textContent = s.llm === "rules" ? "LLM: offline rules" : `LLM: ${s.llm} · ${s.llmModel}`;
    llm.className = `pill ${s.llm === "rules" ? "warn" : "ok"}`;
    llm.title =
      s.llm === "rules"
        ? s.ollama.running
          ? `Ollama is running but "${s.ollama.model}" isn't pulled. Run: ollama pull ${s.ollama.model}`
          : "No LLM connected. Start Ollama or add ANTHROPIC_API_KEY to .env for natural translations."
        : "Translator";
    const voice = $("status-voice");
    voice.textContent = s.voice === "elevenlabs" ? "Voice: ElevenLabs" : "Voice: browser";
    voice.className = `pill ${s.voice === "elevenlabs" ? "ok" : ""}`;
  } catch {
    $("status-llm").textContent = "LLM: server offline";
    $("status-llm").className = "pill warn";
  }
}

// ---------- camera + loop ----------

async function start() {
  const btn = $("start-btn");
  btn.disabled = true;
  $("start-msg").textContent = "Loading hand + face models (first time takes a few seconds)…";
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
    $("start").hidden = true;
    $("hud-pred").hidden = false;
    $("hud-face").hidden = false;
    $("status-vision").textContent = `Vision: live · ${trackers.delegate}`;
    $("status-vision").className = "pill ok";
    requestAnimationFrame(loop);
    if (!classifier.samples.length) {
      toast("No signs trained yet. Open “Train signs” to record your first few.", 5000);
    }
  } catch (err) {
    console.error(err);
    btn.disabled = false;
    $("start-msg").textContent =
      err.name === "NotAllowedError" ? "Camera permission was denied. Allow it in the browser and try again." : `Couldn't start: ${err.message}`;
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
    const hands = state.trackers.hands.detectForVideo(video, now);
    const faceRes = state.trackers.face.detectForVideo(video, now);
    const handCount = hands.landmarks?.length || 0;

    face.collecting = handCount > 0 || state.signs.length > 0;
    const faceState = face.update(faceRes, now);
    const { vec, count } = features.frame(hands, faceRes, video.videoWidth / video.videoHeight);

    state.overlay.resize(video.videoWidth, video.videoHeight);
    state.overlay.render(hands, faceRes, { showFace: state.opts.mesh });

    if (state.recording) recordFrame(vec, count, now);
    else recognise(vec, count, now);

    renderFace(faceState);
    fpsFrames++;
    if (now - fpsSince > 1000) {
      $("fps").textContent = `${fpsFrames} fps`;
      fpsFrames = 0;
      fpsSince = now;
    }
  }
  requestAnimationFrame(loop);
}

// ---------- recognition ----------

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
    const canAuto = state.opts.auto && state.tab === "translate" && state.signs.length && !state.translating;
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
  if (state.tab === "practice") return practiceCheck(gloss);
  if (state.tab !== "translate") return;
  state.signs.push({ gloss, markers: face.recentMarkers() });
  renderSigns();
}

function renderPrediction(pred, handCount) {
  const label = $("pred-label");
  if (!classifier.samples.length) {
    label.textContent = "Train signs first";
    label.className = "hud-label dim";
    $("pred-bar").style.width = "0";
    $("pred-sub").textContent = "";
    return;
  }
  if (!handCount || !pred) {
    label.textContent = handCount ? "…" : "No hands";
    label.className = "hud-label dim";
    $("pred-bar").style.width = "0";
    $("pred-sub").textContent = "";
    return;
  }
  label.textContent = pred.accepted ? pred.label : `${pred.label}?`;
  label.className = `hud-label${pred.accepted ? "" : " dim"}`;
  $("pred-bar").style.width = `${Math.round(pred.confidence * 100)}%`;
  $("pred-sub").textContent = `conf ${pred.confidence.toFixed(2)} · dist ${pred.distance.toFixed(2)}/${pred.limit.toFixed(2)}`;
}

let lastFaceKey = "";
function renderFace(fs) {
  const key = `${fs.present}|${fs.emotion}|${fs.markers.join()}`;
  if (key === lastFaceKey) return;
  lastFaceKey = key;
  $("face-emoji").textContent = fs.present ? EMOJI[fs.emotion] : "🙈";
  $("face-emotion").textContent = fs.present ? fs.emotion : "no face";
  $("face-markers").replaceChildren(...fs.markers.map((m) => el("span", { className: "chip", textContent: MARKER_LABEL[m] })));
}

function renderSigns() {
  const box = $("signs");
  if (!state.signs.length) {
    box.replaceChildren(el("span", { className: "placeholder", textContent: "Recognised signs appear here…" }));
    return;
  }
  box.replaceChildren(
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
  text.className = "result-text loading";
  text.textContent = "Translating…";
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
        history: state.transcript.filter((t) => t.who === "signer").slice(-3).map((t) => t.text),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    state.lastResult = data;
    renderResult(data, glosses, faceSummary);
    addTranscript({ who: "signer", text: data.english, glosses, tone: data.tone });
    if (state.opts.speak) speak(data.english, data.tone);
  } catch (err) {
    text.className = "result-text";
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
  text.className = "result-text";
  text.replaceChildren(...markChanges(data.english, glosses).map((p) => (p.changed ? el("u", { textContent: p.text }) : p.text)));
  const meta = $("result-meta");
  meta.replaceChildren(
    el("span", { className: "tone", textContent: `${EMOJI[faceSummary.emotion] || ""} ${data.tone}` }),
    `${glosses.join(" ")} · ${data.backend}${data.fallbackFrom ? ` (${data.fallbackFrom} failed)` : ""} · ${data.ms} ms`,
  );
  $("result-notes").textContent = data.notes || "";
  $("replay").hidden = false;
}

function addTranscript(entry) {
  state.transcript.push({ ...entry, at: new Date() });
  const list = $("transcript");
  if (list.querySelector(".placeholder")) list.replaceChildren();
  const who = entry.who === "signer" ? `🤟 Signer${entry.tone ? ` · ${entry.tone}` : ""}` : "🗣️ Speaker";
  list.append(el("li", { className: entry.who }, el("span", { className: "who", textContent: who }), entry.text));
  list.scrollTop = list.scrollHeight;
}

// ---------- training ----------

function recordSign() {
  if (!state.running) return toast("Start the camera first.");
  const label = $("label-input").value.trim().toUpperCase().replace(/\s+/g, "-").replace("_REST", "_rest");
  if (!label) return toast("Type a name for the sign first.");
  if (state.recording) return;
  const cd = $("countdown");
  let n = 3;
  cd.hidden = false;
  cd.className = "countdown";
  cd.textContent = n;
  state.recording = { label, phase: "countdown", vecs: [] };
  const tick = setInterval(() => {
    n--;
    if (n > 0) cd.textContent = n;
    else {
      clearInterval(tick);
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
  const timedOut = now - r.started > 5000;
  if (r.vecs.length >= RECORD_FRAMES || timedOut) {
    state.recording = null;
    $("countdown").hidden = true;
    if (r.vecs.length < 10) return toast("Couldn't see your hands. Try again with your hands in frame.");
    classifier.add(r.label, r.vecs);
    renderSignList();
    toast(`Saved ${r.vecs.length} samples for ${r.label}`);
  }
}

function renderSignList() {
  const labels = classifier.labels();
  $("sign-count").textContent = labels.length ? `(${labels.length})` : "";
  const list = $("sign-list");
  if (!labels.length) list.replaceChildren(el("li", { className: "placeholder", textContent: "No signs yet." }));
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
  $("practice-target").textContent = next;
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
    fb.textContent = "✓ Nice!";
    fb.className = "practice-feedback good";
    setTimeout(practiceNext, 900);
  } else {
    p.streak = 0;
    fb.textContent = `That looked like ${gloss}`;
    fb.className = "practice-feedback";
  }
  $("practice-score").textContent = p.score;
  $("practice-streak").textContent = p.streak;
}

// ---------- wiring ----------

function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll("[role=tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
  document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = p.dataset.panel !== tab));
  if (tab === "practice" && !state.practice.target) practiceNext();
}

function bind() {
  $("start-btn").onclick = start;
  document.querySelectorAll("[role=tab]").forEach((b) => (b.onclick = () => setTab(b.dataset.tab)));

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
  $("replay").onclick = () => state.lastResult && speak(state.lastResult.english, state.lastResult.tone);

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
  const listener = createListener({
    onInterim: (t) => ($("caption").textContent = t),
    onFinal: (t) => {
      if (t) addTranscript({ who: "speaker", text: t });
      $("caption").textContent = "";
    },
    onState: (on, err) => {
      $("listen").classList.toggle("live", on);
      $("listen").textContent = on ? "■ Stop listening" : "🎙️ Listen to reply";
      if (err) toast(err);
    },
  });
  $("listen").onclick = () => {
    if (!listener.supported) return toast("Speech recognition needs Chrome or Edge.");
    if (listener.active) listener.stop();
    else {
      stopSpeaking();
      listener.start();
    }
  };

  // settings
  const bindOpt = (id, key) =>
    ($(id).onchange = () => {
      state.opts[key] = $(id).checked;
      saveOpts();
    });
  bindOpt("opt-auto", "auto");
  bindOpt("opt-speak", "speak");
  bindOpt("opt-mesh", "mesh");
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
    $("calibrate-btn").disabled = true;
    $("calibrate-btn").textContent = "Hold a relaxed face…";
    await face.calibrate();
    $("calibrate-btn").disabled = false;
    $("calibrate-btn").textContent = "🙂 Calibrate neutral face";
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

  // keyboard shortcuts (ignored while typing)
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea") || e.metaKey || e.ctrlKey) return;
    if (e.key === "Enter" && state.tab === "translate") {
      e.preventDefault();
      finishSentence();
    } else if (e.key === "Backspace") {
      e.preventDefault();
      $("undo").click();
    } else if (e.key === "Escape") $("clear").click();
    else if (e.key.toLowerCase() === "r" && state.tab === "train") recordSign();
  });
}

bind();
loadOpts();
renderSignList();
refreshStatus();
setInterval(refreshStatus, 15000);
