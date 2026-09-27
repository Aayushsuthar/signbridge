// Trains the ASL fingerspelling classifier (A–Z) from MediaPipe hand landmarks, with no dependencies.
//
//   node scripts/train-fingerspelling.mjs <a.json[,b.jsonl…]> [--holdout c.jsonl] [--folds 5] [--out public/models/fingerspelling.json]
//
// Datasets: JSON arrays or JSON-lines of { label: "A", pts: [[x,y,z] × 21] } (see public/lab/extract-landmarks.html).
// Reports k-fold cross-validated accuracy; with --holdout, also trains on the given sets and tests on a
// dataset from a different source (the honest "new people, new camera" number). Then trains on
// everything and writes the model.
import { readFile, writeFile } from "node:fs/promises";
import { normalize, features, FEATURES, mlpForward, standardize } from "../public/js/fingerspell-features.js";

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const DATA = args[0].split(",");
const HOLDOUT = opt("holdout", null);
const FOLDS = Number(opt("folds", 5));
const OUT = opt("out", "public/models/fingerspelling.json");
const EPOCHS = Number(opt("epochs", 120));
const HIDDEN = [128, 64];
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

// ---------- seeded randomness ----------
let seed = 1234567;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
const shuffle = (a) => {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};

// ---------- data ----------
async function load(path) {
  const text = await readFile(path, "utf8");
  const raw = text.trimStart().startsWith("[") ? JSON.parse(text) : text.split("\n").filter(Boolean).map((l) => JSON.parse(l));
  const seen = new Set();
  const out = [];
  for (const s of raw) {
    if (!LETTERS.includes(s.label) || s.pts?.length !== 21) continue;
    const key = s.pts.map((p) => p.map((v) => v.toFixed(3)).join(",")).join(";");
    if (seen.has(key)) continue; // exact duplicates would leak between train and test
    seen.add(key);
    out.push({ y: LETTERS.indexOf(s.label), norm: normalize(s.pts), src: path });
  }
  console.log(`${path}: ${out.length} unique samples (${raw.length - out.length} duplicates/invalid dropped)`);
  return out;
}
const samples = (await Promise.all(DATA.map(load))).flat();
const holdout = HOLDOUT ? await load(HOLDOUT) : null;

// augmentation: mirror (left-handed signers), small in-plane rotation, aspect stretch, jitter
function augment(norm) {
  const mirror = rand() < 0.5 ? -1 : 1;
  const a = ((rand() * 2 - 1) * 15 * Math.PI) / 180;
  const sx = 1 + (rand() * 2 - 1) * 0.08;
  const sy = 1 + (rand() * 2 - 1) * 0.08;
  const c = Math.cos(a), s = Math.sin(a);
  return norm.map(([x, y, z]) => {
    const mx = x * mirror;
    return [(c * mx - s * y) * sx + gauss() * 0.02, (s * mx + c * y) * sy + gauss() * 0.02, z + gauss() * 0.02];
  });
}

// ---------- MLP ----------
function init(sizes) {
  return sizes.slice(1).map((out, i) => {
    const inp = sizes[i];
    const lim = Math.sqrt(6 / (inp + out));
    return {
      W: Float32Array.from({ length: out * inp }, () => (rand() * 2 - 1) * lim),
      b: new Float32Array(out),
      mW: new Float32Array(out * inp), vW: new Float32Array(out * inp),
      mb: new Float32Array(out), vb: new Float32Array(out),
    };
  });
}

function train(trainSet, { epochs = EPOCHS, lr = 2e-3, batch = 64, dropout = 0.15, wd = 1e-4 } = {}) {
  // standardization stats from augmented training features
  const probe = [];
  for (let r = 0; r < 3; r++) for (const s of trainSet) probe.push(features(augment(s.norm)));
  const mean = new Float32Array(FEATURES), std = new Float32Array(FEATURES);
  for (const f of probe) for (let i = 0; i < FEATURES; i++) mean[i] += f[i] / probe.length;
  for (const f of probe) for (let i = 0; i < FEATURES; i++) std[i] += (f[i] - mean[i]) ** 2 / probe.length;
  for (let i = 0; i < FEATURES; i++) std[i] = Math.sqrt(std[i]) || 1;

  const sizes = [FEATURES, ...HIDDEN, LETTERS.length];
  const layers = init(sizes);
  let t = 0;
  for (let ep = 0; ep < epochs; ep++) {
    const lrNow = lr * 0.5 * (1 + Math.cos((Math.PI * ep) / epochs)); // cosine decay
    const order = shuffle(trainSet.slice());
    for (let bs = 0; bs < order.length; bs += batch) {
      const chunk = order.slice(bs, bs + batch);
      const grads = layers.map((L) => ({ W: new Float32Array(L.W.length), b: new Float32Array(L.b.length) }));
      for (const s of chunk) {
        const f = features(augment(s.norm));
        const x = new Float32Array(FEATURES);
        for (let i = 0; i < FEATURES; i++) x[i] = (f[i] - mean[i]) / std[i];
        // forward, keeping activations and dropout masks
        const acts = [x];
        const masks = [];
        layers.forEach((L, li) => {
          const h = acts[li];
          const out = new Float32Array(L.b.length);
          for (let j = 0; j < out.length; j++) {
            let v = L.b[j];
            const row = j * h.length;
            for (let i = 0; i < h.length; i++) v += L.W[row + i] * h[i];
            out[j] = v;
          }
          if (li < layers.length - 1) {
            const m = new Float32Array(out.length);
            for (let j = 0; j < out.length; j++) {
              const keep = rand() >= dropout ? 1 / (1 - dropout) : 0;
              m[j] = out[j] > 0 ? keep : 0;
              out[j] = out[j] > 0 ? out[j] * keep : 0;
            }
            masks.push(m);
          }
          acts.push(out);
        });
        // softmax cross-entropy gradient
        const logits = acts[acts.length - 1];
        const mx = Math.max(...logits);
        const e = Array.from(logits, (v) => Math.exp(v - mx));
        const sum = e.reduce((a, b) => a + b, 0);
        let delta = Float32Array.from(e, (v, j) => v / sum - (j === s.y ? 1 : 0));
        for (let li = layers.length - 1; li >= 0; li--) {
          const L = layers[li];
          const h = acts[li];
          const g = grads[li];
          const prev = li > 0 ? new Float32Array(h.length) : null;
          for (let j = 0; j < delta.length; j++) {
            const d = delta[j];
            if (d === 0) continue;
            g.b[j] += d;
            const row = j * h.length;
            for (let i = 0; i < h.length; i++) {
              g.W[row + i] += d * h[i];
              if (prev) prev[i] += d * L.W[row + i];
            }
          }
          if (prev) {
            const m = masks[li - 1];
            for (let i = 0; i < prev.length; i++) prev[i] *= m[i];
            delta = prev;
          }
        }
      }
      // Adam with decoupled weight decay
      t++;
      const b1 = 0.9, b2 = 0.999, eps = 1e-8;
      const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
      layers.forEach((L, li) => {
        const g = grads[li];
        for (let k = 0; k < L.W.length; k++) {
          const gw = g.W[k] / chunk.length;
          L.mW[k] = b1 * L.mW[k] + (1 - b1) * gw;
          L.vW[k] = b2 * L.vW[k] + (1 - b2) * gw * gw;
          L.W[k] -= lrNow * ((L.mW[k] / c1) / (Math.sqrt(L.vW[k] / c2) + eps) + wd * L.W[k]);
        }
        for (let k = 0; k < L.b.length; k++) {
          const gb = g.b[k] / chunk.length;
          L.mb[k] = b1 * L.mb[k] + (1 - b1) * gb;
          L.vb[k] = b2 * L.vb[k] + (1 - b2) * gb * gb;
          L.b[k] -= lrNow * ((L.mb[k] / c1) / (Math.sqrt(L.vb[k] / c2) + eps));
        }
      });
    }
  }
  return { labels: LETTERS, mean: Array.from(mean), std: Array.from(std), layers: layers.map((L) => ({ W: Array.from(L.W), b: Array.from(L.b) })) };
}

function evaluate(model, set) {
  let ok = 0, ok3 = 0;
  const confusion = new Map();
  const per = LETTERS.map(() => ({ n: 0, ok: 0 }));
  for (const s of set) {
    const p = mlpForward(model, standardize(model, features(s.norm)));
    const ranked = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    per[s.y].n++;
    if (ranked[0][1] === s.y) {
      ok++;
      per[s.y].ok++;
    } else {
      const k = `${LETTERS[s.y]}→${LETTERS[ranked[0][1]]}`;
      confusion.set(k, (confusion.get(k) || 0) + 1);
    }
    if (ranked.slice(0, 3).some((r) => r[1] === s.y)) ok3++;
  }
  return { acc: ok / set.length, top3: ok3 / set.length, per, confusion };
}

// ---------- k-fold cross-validation (stratified) ----------
const byClass = LETTERS.map((_, c) => shuffle(samples.filter((s) => s.y === c)));
const folds = Array.from({ length: Math.max(FOLDS, 1) }, () => []);
byClass.forEach((list) => list.forEach((s, i) => folds[i % folds.length].push(s)));

const results = [];
const perAll = LETTERS.map(() => ({ n: 0, ok: 0 }));
const confAll = new Map();
for (let k = 0; FOLDS > 1 && k < FOLDS; k++) {
  const test = folds[k];
  const trainSet = folds.filter((_, i) => i !== k).flat();
  const t0 = Date.now();
  const model = train(trainSet);
  const r = evaluate(model, test);
  results.push(r);
  r.per.forEach((p, i) => { perAll[i].n += p.n; perAll[i].ok += p.ok; });
  for (const [c, n] of r.confusion) confAll.set(c, (confAll.get(c) || 0) + n);
  console.log(`fold ${k + 1}/${FOLDS}: top-1 ${(r.acc * 100).toFixed(2)}%  top-3 ${(r.top3 * 100).toFixed(2)}%  (${test.length} test, ${((Date.now() - t0) / 1000).toFixed(0)}s)`);
}
const accs = results.map((r) => r.acc);
const mean = accs.length ? accs.reduce((a, b) => a + b, 0) / accs.length : NaN;
const sd = accs.length ? Math.sqrt(accs.reduce((a, b) => a + (b - mean) ** 2, 0) / accs.length) : NaN;
const top3 = results.length ? results.reduce((a, r) => a + r.top3, 0) / results.length : NaN;
if (results.length) {
  console.log(`\nCV top-1: ${(mean * 100).toFixed(2)}% ± ${(sd * 100).toFixed(2)}   top-3: ${(top3 * 100).toFixed(2)}%`);
  console.log("per letter:", LETTERS.map((l, i) => `${l} ${((perAll[i].ok / perAll[i].n) * 100).toFixed(0)}%`).join("  "));
  console.log("most confused:", [...confAll].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([c, n]) => `${c} ×${n}`).join(", "));
}

// ---------- cross-source test ----------
let cross = null;
if (holdout) {
  const m = train(samples);
  const r = evaluate(m, holdout);
  cross = { top1: +r.acc.toFixed(4), top3: +r.top3.toFixed(4), n: holdout.length };
  console.log(`\ncross-source (train ${DATA.join("+")} → test ${HOLDOUT}): top-1 ${(r.acc * 100).toFixed(2)}%  top-3 ${(r.top3 * 100).toFixed(2)}%`);
  console.log("most confused:", [...r.confusion].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([c, n]) => `${c} ×${n}`).join(", "));
  if (opt("holdout-only", null)) process.exit(0);
}

// ---------- final model on all data ----------
const final = train(holdout ? samples.concat(holdout) : samples);
final.meta = {
  task: "ASL fingerspelling A–Z from MediaPipe hand landmarks",
  data: opt("data-note", DATA.join(" + ")),
  samples: samples.length + (holdout?.length || 0),
  cv: { folds: FOLDS, top1: +mean.toFixed(4), top1_sd: +sd.toFixed(4), top3: +top3.toFixed(4) },
  crossSource: cross,
  perLetter: Object.fromEntries(LETTERS.map((l, i) => [l, +(perAll[i].ok / perAll[i].n).toFixed(3)])),
  architecture: `MLP ${[FEATURES, ...HIDDEN, 26].join("-")}, ReLU, dropout 0.15, Adam, ${EPOCHS} epochs`,
  trained: new Date().toISOString().slice(0, 10),
};
// per-letter mean shape, used by Learn mode to draw the target handshape and give finger-level hints.
// Samples mix left and right hands, so flip each to one handedness first (index knuckle on the +x side
// of the pinky knuckle), otherwise mirror images average out into a flat line.
final.templates = templatesFor(samples.concat(holdout || []));
export function templatesFor(list) {
  const canon = (norm) => (norm[5][0] >= norm[17][0] ? norm : norm.map(([x, y, z]) => [-x, y, z]));
  return Object.fromEntries(
    LETTERS.map((l, c) => {
      const mine = list.filter((s) => s.y === c).map((s) => canon(s.norm));
      return [l, Array.from({ length: 21 }, (_, j) => [0, 1, 2].map((d) => +(mine.reduce((a, n) => a + n[j][d], 0) / mine.length).toFixed(3)))];
    }),
  );
}
const round = (a) => a.map((v) => +v.toFixed(5));
final.mean = round(final.mean);
final.std = round(final.std);
final.layers = final.layers.map((L) => ({ W: round(L.W), b: round(L.b) }));
await writeFile(OUT, JSON.stringify(final));
console.log(`\nwrote ${OUT}`);
