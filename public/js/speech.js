// Voice out (ElevenLabs via the server, or the browser's own voice) and speech in (live captions
// of the hearing person's reply, so the conversation works both ways).

const BROWSER_VOICE = {
  neutral: { rate: 1, pitch: 1 },
  calm: { rate: 0.92, pitch: 0.95 },
  happy: { rate: 1.07, pitch: 1.15 },
  excited: { rate: 1.15, pitch: 1.25 },
  sad: { rate: 0.86, pitch: 0.85 },
  angry: { rate: 1.05, pitch: 0.9 },
  surprised: { rate: 1.1, pitch: 1.3 },
  questioning: { rate: 1, pitch: 1.1 },
};

let current = null;

export function stopSpeaking() {
  if (current) {
    current.pause();
    URL.revokeObjectURL(current.src);
    current = null;
  }
  if ("speechSynthesis" in window) speechSynthesis.cancel();
}

export async function speak(text, tone = "neutral", { onStart, onEnd } = {}) {
  stopSpeaking();
  try {
    const res = await fetch("/api/speak", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, tone }),
    });
    if (res.ok) {
      const url = URL.createObjectURL(await res.blob());
      const audio = new Audio(url);
      current = audio;
      audio.onplay = () => onStart?.();
      audio.onended = () => {
        URL.revokeObjectURL(url);
        if (current === audio) current = null;
        onEnd?.();
      };
      await audio.play();
      return "elevenlabs";
    }
    if (res.status !== 501) console.warn("ElevenLabs failed, using browser voice:", (await res.json()).error);
  } catch (err) {
    console.warn("voice request failed, using browser voice", err);
  }
  if (!("speechSynthesis" in window)) return "none";
  const u = new SpeechSynthesisUtterance(text);
  Object.assign(u, BROWSER_VOICE[tone] || BROWSER_VOICE.neutral);
  u.lang = "en-US";
  u.onstart = () => onStart?.();
  u.onend = () => onEnd?.();
  speechSynthesis.speak(u);
  return "browser";
}

// ---------- speech to text ----------

export function createListener({ onInterim, onFinal, onState }) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return { supported: false, start() {}, stop() {}, get active() { return false; } };
  const rec = new SR();
  rec.lang = "en-US";
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
