// Plays spoken language back as sign language: a real signer's clip for every word that has one
// (WLASL, 1,992 ASL words, streamed from the Hugging Face dataset), and fingerspelling for names and
// words without a sign, drawn from the letter shapes the fingerspelling model learned.

const VIDEO_BASE = "https://huggingface.co/datasets/Voxel51/WLASL/resolve/main/data/";
const WLASL_FPS = 25;
const LETTER_MS = 650;

export class SignPlayer {
  constructor({ video, canvas, caption, onState, drawHand }) {
    Object.assign(this, { video, canvas, caption, onState, drawHand });
    this.index = null;
    this.templates = null;
    this.cache = new Map(); // path → blob URL
    this.token = 0;
    this.probe = Object.assign(document.createElement("canvas"), { width: 16, height: 9 });
  }

  async init() {
    const [index, letters] = await Promise.all([
      fetch("/models/sign-videos.json").then((r) => r.json()),
      fetch("/models/fingerspelling.json").then((r) => r.json()),
    ]);
    this.index = index;
    this.templates = letters.templates;
    return this;
  }

  stop() {
    this.token++;
    this.video.pause();
    this.onState?.(false);
  }

  async #blob(path) {
    if (this.cache.has(path)) return this.cache.get(path);
    const res = await fetch(VIDEO_BASE + path);
    if (!res.ok) throw new Error(`video ${res.status}`);
    const url = URL.createObjectURL(await res.blob());
    this.cache.set(path, url);
    if (this.cache.size > 40) {
      const [k, old] = this.cache.entries().next().value;
      URL.revokeObjectURL(old);
      this.cache.delete(k);
    }
    return url;
  }

  // a few WLASL files use codecs some browsers can't decode (every frame black): detect and skip
  #decodes() {
    const g = this.probe.getContext("2d", { willReadFrequently: true });
    g.drawImage(this.video, 0, 0, 16, 9);
    const px = g.getImageData(0, 0, 16, 9).data;
    let sum = 0;
    for (let i = 0; i < px.length; i += 4) sum += px[i] + px[i + 1] + px[i + 2];
    return sum / (px.length / 4) / 3 > 3;
  }

  async #playClip([path, start, end], token) {
    const v = this.video;
    v.src = await this.#blob(path);
    await new Promise((ok, err) => {
      v.onloadeddata = ok;
      v.onerror = () => err(new Error("unsupported"));
    });
    if (token !== this.token) return true;
    const from = Math.max(0, (start - 1) / WLASL_FPS);
    const to = end > 0 ? end / WLASL_FPS : v.duration;
    v.currentTime = from + 0.05;
    await new Promise((ok) => (v.onseeked = ok));
    if (!this.#decodes()) throw new Error("undecodable");
    v.playbackRate = 1.25;
    await v.play().catch(() => {});
    // never let one stalled clip freeze the reply: give up a little after it should have ended
    const limit = performance.now() + ((to - from) / 1.25) * 1000 + 2500;
    await new Promise((ok) => {
      const tick = () => {
        if (token !== this.token || v.ended || v.currentTime >= to || performance.now() > limit) return ok();
        setTimeout(tick, 40); // setTimeout, not rAF: keeps going if the tab is briefly hidden
      };
      tick();
    });
    v.pause();
    return true;
  }

  async #fingerspell(word, token) {
    this.canvas.hidden = false;
    this.video.hidden = true;
    for (const ch of word.toUpperCase()) {
      if (token !== this.token) return;
      const tpl = this.templates[ch];
      this.drawHand(this.canvas, tpl || null);
      this.caption.textContent = `${word.toUpperCase()} · ${ch}`;
      await new Promise((r) => setTimeout(r, tpl ? LETTER_MS : LETTER_MS / 2));
    }
    this.canvas.hidden = true;
    this.video.hidden = false;
  }

  // signs: [{ word, video }] → plays them in order
  async play(signs) {
    const token = ++this.token;
    this.onState?.(true);
    // warm the cache for the first few clips so playback doesn't stall between words
    for (const s of signs.slice(0, 4)) if (s.video && this.index[s.word]) this.#blob(this.index[s.word][0][0]).catch(() => {});
    for (const [i, s] of signs.entries()) {
      if (token !== this.token) return;
      const next = signs[i + 1];
      if (next?.video && this.index[next.word]) this.#blob(this.index[next.word][0][0]).catch(() => {});
      const clips = s.video ? this.index[s.word] || [] : [];
      this.caption.textContent = s.word.toUpperCase();
      let played = false;
      for (const clip of clips) {
        try {
          played = await this.#playClip(clip, token);
          if (played) break;
        } catch {
          // try the next recording of this word
        }
      }
      if (!played && token === this.token) await this.#fingerspell(s.word.replace(/[^a-z]/gi, ""), token);
    }
    if (token === this.token) this.onState?.(false);
  }
}
