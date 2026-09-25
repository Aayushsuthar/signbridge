// Voice out (ElevenLabs via the server, or the browser's own voice) and speech in (live captions
// of the hearing person's reply). Female voices by default, in English, Hindi and Hinglish.

// ElevenLabs premade female voices
export const VOICES = [
  { id: "EXAVITQu4vr4xnSDxMaL", name: "Sarah", note: "warm, soft" },
  { id: "9BWtsMINqrJLrRacOk9x", name: "Aria", note: "expressive" },
  { id: "cgSgspJ2msm6clMCkdW9", name: "Jessica", note: "bright, playful" },
  { id: "pFZP5JQG7iQjIQuC4Bku", name: "Lily", note: "gentle, British" },
  { id: "XB0fDUnXU5powFXDhCwa", name: "Charlotte", note: "calm" },
  { id: "21m00Tcm4TlvDq8ikWAM", name: "Rachel", note: "clear, neutral" },
];

export const LANG_TAG = { en: "en-US", hi: "hi-IN", hinglish: "en-IN" };

const BROWSER_TONE = {
  neutral: { rate: 1, pitch: 1.05 },
  calm: { rate: 0.93, pitch: 1 },
  happy: { rate: 1.06, pitch: 1.2 },
  excited: { rate: 1.14, pitch: 1.3 },
  sad: { rate: 0.86, pitch: 0.9 },
  angry: { rate: 1.05, pitch: 0.95 },
  surprised: { rate: 1.1, pitch: 1.35 },
  questioning: { rate: 1, pitch: 1.15 },
};

// Names of female system voices across macOS, Windows, Chrome and Android
const FEMALE = /samantha|victoria|karen|moira|tessa|fiona|veena|lekha|kalpana|zira|heera|swara|neerja|aria|jenny|sonia|libby|natasha|serena|allison|ava|susan|zoe|google us english|google uk english female|google हिन्दी|female|woman/i;
const MALE = /\b(male|daniel|alex|fred|rishi|hemant|ravi|david|mark|guy|george|thomas|oliver)\b/i;

let voicesReady = null;
function systemVoices() {
  if (!("speechSynthesis" in window)) return Promise.resolve([]);
  const now = speechSynthesis.getVoices();
  if (now.length) return Promise.resolve(now);
  voicesReady ??= new Promise((resolve) => {
    speechSynthesis.onvoiceschanged = () => resolve(speechSynthesis.getVoices());
    setTimeout(() => resolve(speechSynthesis.getVoices()), 1500);
  });
  return voicesReady;
}

export async function pickBrowserVoice(lang) {
  const voices = await systemVoices();
  const tag = LANG_TAG[lang] || "en-US";
  const base = tag.split("-")[0];
  const rank = (v) =>
    (v.lang === tag ? 4 : v.lang.startsWith(base) ? 2 : 0) +
    (FEMALE.test(v.name) ? 3 : 0) -
    (MALE.test(v.name) ? 5 : 0) +
    (v.localService ? 0 : 1); // network voices usually sound better
  return voices.filter((v) => v.lang.startsWith(base) || (lang === "hinglish" && v.lang.startsWith("hi"))).sort((a, b) => rank(b) - rank(a))[0] || null;
}

const stripTags = (s) => s.replace(/\[[a-z][a-z ]{1,24}\]\s*/gi, "").trim();

let current = null;
let audioCtx = null;

export function stopSpeaking() {
  if (current) {
    current.audio?.pause();
    current.stop?.();
    current = null;
  }
  if ("speechSynthesis" in window) speechSynthesis.cancel();
}

// onLevel(0..1) is called every frame while speaking, for the waveform animation.
export async function speak(text, { tone = "neutral", lang = "en", voice, onStart, onEnd, onLevel } = {}) {
  stopSpeaking();
  const session = {};
  current = session;
  const finish = () => {
    if (current === session) current = null;
    session.stop?.();
    onLevel?.(0);
    onEnd?.();
  };

  try {
    const res = await fetch("/api/speak", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, tone, voice }),
    });
    if (current !== session) return "cancelled";
    if (res.ok) {
      const url = URL.createObjectURL(await res.blob());
      const audio = new Audio(url);
      session.audio = audio;
      // real loudness from the audio, for the waveform
      try {
        audioCtx ??= new AudioContext();
        const src = audioCtx.createMediaElementSource(audio);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        src.connect(analyser).connect(audioCtx.destination);
        const data = new Uint8Array(analyser.frequencyBinCount);
        let raf;
        const tick = () => {
          analyser.getByteFrequencyData(data);
          let sum = 0;
          for (let i = 2; i < 48; i++) sum += data[i];
          onLevel?.(Math.min(1, sum / (46 * 150)));
          raf = requestAnimationFrame(tick);
        };
        audio.onplay = () => {
          audioCtx.resume();
          onStart?.();
          tick();
        };
        session.stop = () => cancelAnimationFrame(raf);
      } catch {
        audio.onplay = () => onStart?.();
      }
      audio.onended = () => {
        URL.revokeObjectURL(url);
        finish();
      };
      await audio.play();
      return "elevenlabs";
    }
    if (res.status !== 501) console.warn("ElevenLabs failed, using browser voice:", (await res.json()).error);
  } catch (err) {
    console.warn("voice request failed, using browser voice", err);
  }

  if (!("speechSynthesis" in window) || current !== session) return "none";
  const u = new SpeechSynthesisUtterance(stripTags(text));
  const v = await pickBrowserVoice(lang);
  if (v) u.voice = v;
  u.lang = v?.lang || LANG_TAG[lang];
  Object.assign(u, BROWSER_TONE[tone] || BROWSER_TONE.neutral);
  // browsers don't expose TTS audio, so animate the waveform from word boundaries
  let level = 0;
  let raf;
  const tick = () => {
    level *= 0.9;
    onLevel?.(0.25 + level * 0.6 + Math.random() * 0.15);
    raf = requestAnimationFrame(tick);
  };
  u.onboundary = () => (level = 1);
  u.onstart = () => {
    onStart?.();
    tick();
  };
  session.stop = () => cancelAnimationFrame(raf);
  u.onend = finish;
  u.onerror = finish;
  speechSynthesis.speak(u);
  return "browser";
}

// ---------- speech to text ----------

export function createListener({ onInterim, onFinal, onState }) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return { supported: false, start() {}, stop() {}, setLang() {}, get active() { return false; } };
  const rec = new SR();
  rec.lang = "en-IN";
  rec.continuous = true;
  rec.interimResults = true;
  let active = false;
  rec.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      if (r.isFinal) onFinal?.(r[0].transcript.trim());
      else interim += r[0].transcript;
    }
    onInterim?.(interim.trim());
  };
  // Chrome ends continuous recognition after a silence; restart while the user still wants it on.
  rec.onend = () => {
    if (active) {
      try {
        rec.start();
      } catch {
        active = false;
        onState?.(false);
      }
    } else onState?.(false);
  };
  rec.onerror = (e) => {
    if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      active = false;
      onState?.(false, "Microphone permission was denied.");
    }
  };
  return {
    supported: true,
    get active() {
      return active;
    },
    setLang(lang) {
      rec.lang = lang === "hi" ? "hi-IN" : "en-IN";
    },
    start() {
      active = true;
      rec.start();
      onState?.(true);
    },
    stop() {
      active = false;
      rec.stop();
    },
  };
}
