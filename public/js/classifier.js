// k-nearest-neighbour sign classifier trained in the browser from the user's own recordings.
// Small, instant to retrain, and good at static and short-motion signs.
import { FEATURE_DIM } from "./vision.js";

const STORAGE_KEY = "signbridge.samples.v1";

function dist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

export class SignClassifier {
  constructor() {
    this.samples = []; // { label, vec: Float32Array }
    this.spread = new Map(); // label -> typical distance of its samples from their centroid
    this.tolerance = 2.2;
    this.load();
  }

  // ----- persistence -----

  load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) this.import(JSON.parse(raw), { replace: true, persist: false });
    } catch (err) {
      console.warn("could not load saved signs", err);
    }
  }

  save() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.export()));
    } catch (err) {
      console.warn("could not save signs (storage full or blocked)", err);
    }
  }

  export() {
    return {
      app: "signbridge",
      version: 1,
      dim: FEATURE_DIM,
      samples: this.samples.map((s) => ({ label: s.label, vec: Array.from(s.vec, (v) => Math.round(v * 1e4) / 1e4) })),
    };
  }

  import(data, { replace = false, persist = true } = {}) {
    if (!data || data.dim !== FEATURE_DIM || !Array.isArray(data.samples)) {
      throw new Error("This file isn't a SignBridge sign set (or it's from an incompatible version).");
    }
    if (replace) this.samples = [];
    for (const s of data.samples) {
      if (typeof s.label === "string" && Array.isArray(s.vec) && s.vec.length === FEATURE_DIM) {
        this.samples.push({ label: s.label, vec: Float32Array.from(s.vec) });
      }
    }
    this.recompute();
    if (persist) this.save();
  }

  // ----- training -----

  add(label, vecs) {
    for (const vec of vecs) this.samples.push({ label, vec });
    this.recompute();
    this.save();
  }

  remove(label) {
    this.samples = this.samples.filter((s) => s.label !== label);
    this.recompute();
    this.save();
  }

  clear() {
    this.samples = [];
    this.recompute();
    this.save();
  }

  labels() {
    const counts = new Map();
    for (const s of this.samples) counts.set(s.label, (counts.get(s.label) || 0) + 1);
    return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }

  recompute() {
    this.spread.clear();
    const byLabel = new Map();
    for (const s of this.samples) {
      if (!byLabel.has(s.label)) byLabel.set(s.label, []);
      byLabel.get(s.label).push(s.vec);
    }
    for (const [label, vecs] of byLabel) {
      const c = new Float32Array(FEATURE_DIM);
      for (const v of vecs) for (let i = 0; i < FEATURE_DIM; i++) c[i] += v[i] / vecs.length;
      const ds = vecs.map((v) => dist(v, c)).sort((a, b) => a - b);
      const p80 = ds[Math.floor(ds.length * 0.8)] ?? ds[ds.length - 1];
      // floor keeps a very consistent recording from producing an impossibly tight threshold
      this.spread.set(label, Math.max(p80, 0.6));
    }
  }

  // ----- inference -----

  predict(vec, k = 7) {
    if (!this.samples.length) return null;
    const scored = this.samples.map((s) => ({ label: s.label, d: dist(vec, s.vec) }));
    scored.sort((a, b) => a.d - b.d);
    const top = scored.slice(0, k);
    const votes = new Map();
    let total = 0;
    for (const { label, d } of top) {
      const w = 1 / (d + 1e-3);
      votes.set(label, (votes.get(label) || 0) + w);
      total += w;
    }
    let label = null;
    let best = 0;
    for (const [l, w] of votes) if (w > best) [label, best] = [l, w];
    const nearest = top.find((t) => t.label === label).d;
    const limit = this.spread.get(label) * this.tolerance * 2;
    return {
      label,
      confidence: best / total,
      distance: nearest,
      limit,
      accepted: nearest <= limit && best / total >= 0.55,
    };
  }
}
