// Pretrained ASL word recognition: 250 signs, no training needed.
//
// Model: 1st-place solution of Google's "Isolated Sign Language Recognition" Kaggle competition
// (Hoyeol Sohn, MIT licence, via huggingface.co/sign/kaggle-asl-signs-1st-place). It takes a clip of
// MediaPipe Holistic landmarks, [frames, 543, 3] in the order face(468) · left hand(21) · pose(33) ·
// right hand(21), with NaN for anything not detected, and returns 250 scores. Internally it only
// uses the lips, eyes, nose and both hands, so our separate Face + Hand landmarkers are enough and
// pose stays NaN.
import { loadLiteRt, loadAndCompile, Tensor } from "https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/+esm";
import { inputShapeOffsets, withInputFrames } from "./tflite-patch.js";

const WASM = "https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/";
export const POINTS = 543;
const FACE = 0;
const LEFT_HAND = 468;
const RIGHT_HAND = 522;
const MAX_FRAMES = 160; // the model was trained on clips up to 384 frames; real signs are 0.5–4 s

let runtime = null;

// The competition's labels squash some words together; show and translate them readably.
const GLOSS = {
  callonphone: "CALL-ON-PHONE", frenchfries: "FRENCH-FRIES", glasswindow: "GLASS/WINDOW", haveto: "HAVE-TO",
  hesheit: "HE/SHE/IT", icecream: "ICE-CREAM", minemy: "MY/MINE", thankyou: "THANK-YOU", weus: "WE/US",
};
export const glossOf = (word) => GLOSS[word] || word.toUpperCase();

// Where to watch a real signer do each word (SignASL.org has videos for common signs)
const VIDEO_SLUG = {
  callonphone: "call", frenchfries: "french-fries", glasswindow: "window", haveto: "have-to", hesheit: "he",
  icecream: "ice-cream", minemy: "my", thankyou: "thank-you", weus: "we", TV: "tv", shhh: "shh", owie: "hurt",
};
export const videoFor = (word) => `https://www.signasl.org/sign/${VIDEO_SLUG[word] || word.toLowerCase()}`;

export class WordRecognizer {
  constructor() {
    this.labels = [];
    this.bytes = null;
    this.offsets = null;
    this.cache = new Map(); // frames → compiled model (small LRU; compiling takes ~25 ms)
  }

  async init(base = "/models") {
    runtime ??= loadLiteRt(WASM);
    const [labels, buf] = await Promise.all([
      fetch(`${base}/asl-signs-labels.json`).then((r) => r.json()),
      fetch(`${base}/asl-signs.tflite`).then((r) => r.arrayBuffer()),
      runtime,
    ]);
    this.labels = Object.entries(labels).sort((a, b) => a[1] - b[1]).map(([w]) => w);
    this.bytes = new Uint8Array(buf);
    this.offsets = inputShapeOffsets(this.bytes);
    return this;
  }

  // One frame of landmarks in the model's layout.
  // HandLandmarker's "Right" label goes in the right-hand slot. Verified on WLASL test clips: this
  // assignment is what the model expects (the swapped one scored 0% on the same clips).
  static frame(handResult, faceResult) {
    const f = new Float32Array(POINTS * 3).fill(NaN);
    const face = faceResult?.faceLandmarks?.[0];
    if (face) for (let i = 0; i < 468; i++) f.set([face[i].x, face[i].y, face[i].z], (FACE + i) * 3);
    (handResult?.landmarks || []).forEach((hand, i) => {
      const label = handResult.handedness?.[i]?.[0]?.categoryName;
      const base = label === "Right" ? RIGHT_HAND : LEFT_HAND;
      for (let j = 0; j < 21; j++) f.set([hand[j].x, hand[j].y, hand[j].z], (base + j) * 3);
    });
    return f;
  }

  static hasHands(frame) {
    return !Number.isNaN(frame[LEFT_HAND * 3]) || !Number.isNaN(frame[RIGHT_HAND * 3]);
  }

  static hasAny(frame) {
    return WordRecognizer.hasHands(frame) || !Number.isNaN(frame[FACE * 3]);
  }

  async model(frames) {
    let m = this.cache.get(frames);
    if (m) {
      this.cache.delete(frames);
      this.cache.set(frames, m);
      return m;
    }
    m = await loadAndCompile(withInputFrames(this.bytes, frames, this.offsets), { accelerator: "wasm" });
    this.cache.set(frames, m);
    if (this.cache.size > 4) {
      const [k, old] = this.cache.entries().next().value;
      this.cache.delete(k);
      old.delete();
    }
    return m;
  }

  // frames: Float32Array[] (each POINTS*3). Returns the top-k words with probabilities.
  async predict(frames, k = 5) {
    // like the training pipeline, drop frames where none of the used landmarks were seen
    let clip = frames.filter(WordRecognizer.hasAny);
    if (!clip.length) return [];
    if (clip.length > MAX_FRAMES) {
      const step = clip.length / MAX_FRAMES;
      clip = Array.from({ length: MAX_FRAMES }, (_, i) => clip[Math.floor(i * step)]);
    }
    const T = clip.length;
    const data = new Float32Array(T * POINTS * 3);
    clip.forEach((f, t) => data.set(f, t * POINTS * 3));
    const model = await this.model(T);
    const input = new Tensor(data, [T, POINTS, 3]);
    let scores;
    try {
      const out = await model.run(input);
      const res = await out[0].moveTo("wasm");
      scores = Array.from(res.toTypedArray());
      res.delete();
    } finally {
      input.delete();
    }
    // the exported model returns logits; softmax them into probabilities
    const max = Math.max(...scores);
    const exp = scores.map((s) => Math.exp(s - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp
      .map((e, i) => ({ word: this.labels[i], p: e / sum }))
      .sort((a, b) => b.p - a.p)
      .slice(0, k);
  }
}

// Live segmentation on top of the clip model. The model was trained on one sign per clip, but people
// sign continuously, so we score a sliding ~1.5 s window a few times a second and commit a word when
// it's confidently on top twice in a row; when the hands drop, the whole gesture gets a final look.
export class WordStream {
  constructor(recognizer, { window = 42, every = 350, commitP = 0.5, finalP = 0.3, onWord, onGuess } = {}) {
    Object.assign(this, { recognizer, window, every, commitP, finalP, onWord, onGuess });
    this.reset();
  }

  reset() {
    this.frames = []; // current gesture (hands up), capped
    this.lastRun = 0;
    this.lastTop = null;
    this.committedInGesture = [];
    this.lastWord = null;
    this.lastWordAt = 0;
    this.busy = false;
    this.handsGoneAt = null;
  }

  // call every tracked frame
  push(frame, now) {
    const hands = WordRecognizer.hasHands(frame);
    if (hands) {
      this.handsGoneAt = null;
      this.frames.push(frame);
      if (this.frames.length > 240) this.frames.shift();
      if (now - this.lastRun > this.every && this.frames.length >= 12 && !this.busy) this.#score(this.frames.slice(-this.window), now, false);
    } else if (this.frames.length) {
      this.handsGoneAt ??= now;
      // 350 ms tolerates brief tracking dropouts without splitting one sign into two gestures
      if (now - this.handsGoneAt > 350 && !this.busy) {
        const gesture = this.frames;
        const committed = this.committedInGesture; // snapshot: scoring is async
        this.frames = [];
        this.lastTop = null;
        this.committedInGesture = [];
        if (gesture.length >= 8) this.#score(gesture, now, true, committed);
      }
    }
  }

  async #score(clip, now, final, committed = this.committedInGesture) {
    this.busy = true;
    this.lastRun = now;
    try {
      const top = await this.recognizer.predict(clip, 5);
      if (!top.length) return;
      this.onGuess?.(top, final);
      const best = top[0];
      // the same word again needs a real pause in between (signing it twice on purpose)
      const repeatOk = best.word !== this.lastWord || now - this.lastWordAt > 2500;
      if (final) {
        if (!committed.length && best.p >= this.finalP && repeatOk) this.#commit(best, top, now);
      } else if (best.p >= this.commitP && this.lastTop === best.word && repeatOk && !committed.includes(best.word)) {
        this.#commit(best, top, now);
      }
      this.lastTop = best.word;
    } finally {
      this.busy = false;
    }
  }

  #commit(best, top, now) {
    this.committedInGesture.push(best.word);
    this.lastWord = best.word;
    this.lastWordAt = now;
    this.onWord?.(best, top);
  }
}
