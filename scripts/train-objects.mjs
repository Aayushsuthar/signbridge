// Trains the everyday-object recognizer on top of CLIP, with no dependencies.
//
//   node scripts/train-objects.mjs <text.jsonl> <train.jsonl[,web.jsonl]> <test.jsonl> [--extra other.jsonl] [--out public/models/objects-clip.json]
//
// Inputs come from public/lab/clip-embed.html (the app's own CLIP module):
//   text.jsonl   { id, name, emb }   one prompt-ensembled text embedding per object in objects-kb.json
//   train.jsonl  { obj, emb }        Caltech-256 training photos mapped to our objects
//   test.jsonl   { obj, emb }        Caltech-256 test photos (never used for fitting)
//
// Model: zero-shot CLIP (image·text) plus a Tip-Adapter-style cache: for objects that have training
// photos, an affinity to their visual prototype is added to the logit. α (strength) and β (sharpness)
// are chosen by 5-fold cross-validation on the training split only; the test split is scored once.
// Every score is top-k among ALL objects in the vocabulary, not just those with photos.
import { readFileSync, writeFileSync } from "node:fs";

const args = process.argv.slice(2);
const [TEXT, TRAIN, TEST] = args;
const outIdx = args.indexOf("--out");
const OUT = outIdx >= 0 ? args[outIdx + 1] : "public/models/objects-clip.json";
// --extra: photos from a different source (e.g. ImageNet) incl. objects WITHOUT training photos.
// Half A joins model selection (so the adapter can't win by over-favouring trained objects),
// half B is only scored at the end.
const extraIdx = args.indexOf("--extra");
const EXTRA = extraIdx >= 0 ? args[extraIdx + 1] : null;
const SCALE = 100; // CLIP's learned logit scale

const lines = (p) => readFileSync(p, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
const unit = (v) => {
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
};
const dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

const text = lines(TEXT).sort((a, b) => a.id - b.id);
const ids = text.map((t) => t.id);
const T = text.map((t) => unit(t.emb));
const col = new Map(ids.map((id, i) => [id, i]));
// labels come from the current vocabulary's Caltech mapping (not what was stored at embedding time),
// so fixing a mapping in objects-vocab.mjs doesn't require re-embedding any photos
const kb = JSON.parse(readFileSync(new URL("../public/models/objects-kb.json", import.meta.url), "utf8"));
const caltechTo = new Map(kb.flatMap((o) => o.caltech.map((c) => [c, o.id])));
// Caltech rows are labelled via the vocabulary's mapping; other sources (e.g. CLIP-filtered web
// photos from public/lab) carry the object id directly. TRAIN may list several files, comma-separated.
const label = (r) => col.get(r.caltech ? caltechTo.get(r.caltech) : r.obj);
const train = TRAIN.split(",").flatMap((f) => lines(f)).map((r) => ({ y: label(r), e: unit(r.emb) })).filter((r) => r.y !== undefined);
const test = lines(TEST).map((r) => ({ y: label(r), e: unit(r.emb) })).filter((r) => r.y !== undefined);
const extra = EXTRA ? lines(EXTRA).map((r) => ({ y: col.get(r.obj), e: unit(r.emb) })).filter((r) => r.y !== undefined) : [];
const extraA = extra.filter((_, i) => i % 2 === 0);
const extraB = extra.filter((_, i) => i % 2 === 1);
console.log(`${ids.length} objects · ${train.length} training photos · ${test.length} test photos${extra.length ? ` · ${extra.length} extra-source photos` : ""}`);

function prototypes(rows) {
  const sums = new Map();
  for (const r of rows) {
    const s = sums.get(r.y) || { v: new Float64Array(r.e.length), n: 0 };
    for (let i = 0; i < r.e.length; i++) s.v[i] += r.e[i];
    s.n++;
    sums.set(r.y, s);
  }
  const P = new Array(ids.length).fill(null);
  for (const [y, s] of sums) P[y] = unit(Array.from(s.v));
  return P;
}

// Objects without training photos get an estimated visual prototype, learned from the objects that
// have both: "gap" shifts the text embedding by the average text→image offset (CLIP's modality gap);
// "krr" fits kernel ridge regression from text embeddings to image prototypes. Without this, the
// adapter pulls predictions toward objects that happened to have photos.
function fillPrototypes(P, method, lambda = 0.5) {
  if (method === "none") return P;
  const have = P.map((p, i) => (p ? i : -1)).filter((i) => i >= 0);
  const out = P.slice();
  if (method === "gap") {
    const g = new Float64Array(T[0].length);
    for (const i of have) for (let d = 0; d < g.length; d++) g[d] += (P[i][d] - T[i][d]) / have.length;
    P.forEach((p, i) => { if (!p) out[i] = unit(T[i].map((v, d) => v + g[d])); });
    return out;
  }
  // kernel ridge: A = (K + λI)^-1 · P_have, prototype_j = unit(k_j · A), with k_j = t_j · T_haveᵀ
  const n = have.length;
  const K = have.map((i) => have.map((j) => dot(T[i], T[j]) + (i === j ? lambda : 0)));
  const dim = T[0].length;
  const B = have.map((i) => Array.from(P[i]));
  // Gauss-Jordan solve K · A = B
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(K[r][c]) > Math.abs(K[piv][c])) piv = r;
    [K[c], K[piv]] = [K[piv], K[c]];
    [B[c], B[piv]] = [B[piv], B[c]];
    const d = K[c][c];
    for (let k = 0; k < n; k++) K[c][k] /= d;
    for (let k = 0; k < dim; k++) B[c][k] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c || !K[r][c]) continue;
      const f = K[r][c];
      for (let k = 0; k < n; k++) K[r][k] -= f * K[c][k];
      for (let k = 0; k < dim; k++) B[r][k] -= f * B[c][k];
    }
  }
  P.forEach((p, j) => {
    if (p) return;
    const v = new Float64Array(dim);
    have.forEach((i, r) => {
      const w = dot(T[j], T[i]);
      for (let k = 0; k < dim; k++) v[k] += w * B[r][k];
    });
    out[j] = unit(Array.from(v));
  });
  return out;
}

function logits(e, P, alpha, beta) {
  const out = T.map((t) => SCALE * dot(e, t));
  if (!alpha) return out;
  // affinity to each visual prototype, centred so objects without photos aren't penalised
  const aff = [];
  P.forEach((p, i) => p && aff.push([i, Math.exp(-beta * (1 - dot(e, p)))]));
  const mean = aff.reduce((a, [, v]) => a + v, 0) / (aff.length || 1);
  for (const [i, v] of aff) out[i] += alpha * (v - mean);
  return out;
}

function score(rows, P, alpha, beta) {
  let top1 = 0, top3 = 0, top5 = 0;
  const confusion = new Map();
  for (const r of rows) {
    const l = logits(r.e, P, alpha, beta);
    const rank = l.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    const pos = rank.findIndex(([, i]) => i === r.y);
    if (pos === 0) top1++;
    else {
      const k = `${text[r.y].name} → ${text[rank[0][1]].name}`;
      confusion.set(k, (confusion.get(k) || 0) + 1);
    }
    if (pos < 3) top3++;
    if (pos < 5) top5++;
  }
  const n = rows.length || 1;
  return { top1: top1 / n, top3: top3 / n, top5: top5 / n, confusion };
}

// ---------- choose α, β by 5-fold CV on the training split ----------
let seed = 7;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const shuffled = train.slice().sort(() => rand() - 0.5);
const folds = Array.from({ length: 5 }, (_, k) => shuffled.filter((_, i) => i % 5 === k));
const grid = [];
for (const method of ["none", "gap", "krr"])
  for (const alpha of [0, 2, 5, 8, 12, 16, 22, 30, 40])
    for (const beta of alpha ? [1, 2, 3, 5.5, 8] : [0]) if (alpha || method === "none") grid.push({ method, alpha, beta });
let best = null;
for (const g of grid) {
  let acc = 0;
  for (let k = 0; k < 5; k++) {
    const P = fillPrototypes(prototypes(folds.filter((_, i) => i !== k).flat()), g.method);
    acc += score(folds[k], P, g.alpha, g.beta).top1 / 5;
  }
  if (extraA.length) acc = (acc + score(extraA, fillPrototypes(prototypes(train), g.method), g.alpha, g.beta).top1) / 2;
  if (!best || acc > best.acc) best = { ...g, acc };
}
console.log(`model selection: best ${best.method} α=${best.alpha} β=${best.beta} (selection top-1 ${(best.acc * 100).toFixed(1)}%)`);

// ---------- final: prototypes from all training photos, scored once on test ----------
const trained = new Set(prototypes(train).map((p, i) => (p ? i : -1)).filter((i) => i >= 0));
const P = fillPrototypes(prototypes(train), best.method);
const zs = score(test, P, 0, 0);
const ad = score(test, P, best.alpha, best.beta);
const pct = (x) => `${(x * 100).toFixed(1)}%`;
console.log(`\ntest, zero-shot CLIP:        top-1 ${pct(zs.top1)}  top-3 ${pct(zs.top3)}  top-5 ${pct(zs.top5)}`);
console.log(`test, CLIP + trained adapter: top-1 ${pct(ad.top1)}  top-3 ${pct(ad.top3)}  top-5 ${pct(ad.top5)}`);
console.log("most confused:", [...ad.confusion].sort((a, b) => b[1] - a[1]).slice(0, 10).map(([c, n]) => `${c} ×${n}`).join(", "));
let extraReport = null;
if (extraB.length) {
  const withPhotos = extraB.filter((r) => trained.has(r.y));
  const textOnly = extraB.filter((r) => !trained.has(r.y));
  const row = (rows) => ({ zs: score(rows, P, 0, 0), ad: score(rows, P, best.alpha, best.beta), n: rows.length });
  const [t, o, all] = [row(withPhotos), row(textOnly), row(extraB)];
  console.log(`\nindependent source (${extraB.length} photos, never used for fitting):`);
  console.log(`  objects WITH training photos (${t.n}):    zero-shot ${pct(t.zs.top1)} → adapter ${pct(t.ad.top1)} top-1 (top-3 ${pct(t.ad.top3)})`);
  console.log(`  objects WITHOUT training photos (${o.n}): zero-shot ${pct(o.zs.top1)} → adapter ${pct(o.ad.top1)} top-1 (top-3 ${pct(o.ad.top3)})`);
  console.log(`  all (${all.n}):                            zero-shot ${pct(all.zs.top1)} → adapter ${pct(all.ad.top1)} top-1 (top-3 ${pct(all.ad.top3)})`);
  extraReport = { trained: { n: t.n, zeroShot: +t.zs.top1.toFixed(4), adapter: +t.ad.top1.toFixed(4) }, textOnly: { n: o.n, zeroShot: +o.zs.top1.toFixed(4), adapter: +o.ad.top1.toFixed(4) }, all: { n: all.n, zeroShot: +all.zs.top1.toFixed(4), adapter: +all.ad.top1.toFixed(4), adapterTop3: +all.ad.top3.toFixed(4) } };
}

const r5 = (v) => v.map((x) => +x.toFixed(5));
writeFileSync(
  OUT,
  JSON.stringify({
    dim: T[0].length,
    scale: SCALE,
    alpha: best.alpha,
    method: best.method,
    beta: best.beta,
    ids,
    names: text.map((t) => t.name),
    weights: T.map(r5),
    prototypes: P.map((p) => (p ? r5(p) : null)),
    meta: {
      base: "CLIP ViT-B/32 (Xenova/clip-vit-base-patch32), prompt-ensembled zero-shot + Tip-Adapter-style prototypes (estimated for objects without photos)",
      data: "Caltech-256 (ilee0022/Caltech-256) photos mapped to the SignBridge vocabulary + CLIP-filtered Wikimedia Commons photos for every object",
      trainPhotos: train.length,
      testPhotos: test.length,
      test: { zeroShot: { top1: +zs.top1.toFixed(4), top5: +zs.top5.toFixed(4) }, adapter: { top1: +ad.top1.toFixed(4), top3: +ad.top3.toFixed(4), top5: +ad.top5.toFixed(4) } },
      independent: extraReport,
      trained: new Date().toISOString().slice(0, 10),
    },
  }),
);
console.log(`\nwrote ${OUT}`);
