// Shared by training (scripts/train-fingerspelling.mjs) and the browser: turn 21 MediaPipe hand landmarks into a feature vector
// that ignores where the hand is and how big it looks, but keeps its shape and orientation.
// Pure functions, no dependencies, so the exact same code runs in both places.

export const TIPS = [4, 8, 12, 16, 20];
const CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]];
// coords 63 · tip distances 15 · joint bend angles 15 · spread between neighbouring fingers 4 · palm normal 3 · hand direction 2
export const FEATURES = 63 + 15 + 15 + 4 + 3 + 2;

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a) => Math.hypot(a[0], a[1], a[2]) || 1e-6;
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const angle = (a, b) => Math.acos(Math.max(-1, Math.min(1, dot(a, b) / (len(a) * len(b)))));

// pts: [[x,y,z] × 21] in image-normalized coordinates (x scaled by aspect ratio if known)
export function normalize(pts) {
  const [wx, wy, wz] = pts[0];
  const palm = Math.hypot(pts[9][0] - wx, pts[9][1] - wy, pts[9][2] - wz) || 1e-6;
  return pts.map(([x, y, z]) => [(x - wx) / palm, (y - wy) / palm, (z - wz) / palm]);
}

export function features(norm) {
  const f = new Float32Array(FEATURES);
  norm.forEach(([x, y, z], i) => {
    f[i * 3] = x;
    f[i * 3 + 1] = y;
    f[i * 3 + 2] = z * 0.5; // depth is noisier than x/y
  });
  // fingertip-to-fingertip and fingertip-to-wrist distances: robust cues for touching fingers (E/S/T/M/N…)
  let k = 63;
  for (let a = 0; a < TIPS.length; a++) {
    for (let b = a + 1; b < TIPS.length; b++) {
      const p = norm[TIPS[a]], q = norm[TIPS[b]];
      f[k++] = Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    }
  }
  for (const t of TIPS) f[k++] = Math.hypot(...norm[t]);
  // how much each joint bends: independent of camera distance, position and aspect ratio
  for (const c of CHAINS) {
    for (let j = 1; j <= 3; j++) f[k++] = angle(sub(norm[c[j]], norm[c[j - 1]]), sub(norm[c[j + 1]], norm[c[j]]));
  }
  // spread between neighbouring fingers (V vs U, W, etc.)
  for (let c = 0; c < 4; c++) f[k++] = angle(sub(norm[CHAINS[c][4]], norm[CHAINS[c][1]]), sub(norm[CHAINS[c + 1][4]], norm[CHAINS[c + 1][1]]));
  // palm facing direction and where the hand points (H vs U, G, P/K, Q all differ mainly by orientation)
  const n = cross(norm[5], norm[17]);
  const nl = len(n);
  f[k++] = n[0] / nl;
  f[k++] = n[1] / nl;
  f[k++] = n[2] / nl;
  const d = norm[9];
  const dl = Math.hypot(d[0], d[1]) || 1e-6;
  f[k++] = d[0] / dl;
  f[k++] = d[1] / dl;
  return f;
}

// Tiny MLP forward pass: weights as produced by scripts/train-fingerspelling.mjs
export function mlpForward(model, x) {
  let h = x;
  model.layers.forEach((L, li) => {
    const out = new Float32Array(L.b.length);
    for (let j = 0; j < L.b.length; j++) {
      let s = L.b[j];
      const row = j * h.length;
      for (let i = 0; i < h.length; i++) s += L.W[row + i] * h[i];
      out[j] = li < model.layers.length - 1 ? Math.max(0, s) : s;
    }
    h = out;
  });
  const max = Math.max(...h);
  const e = Array.from(h, (v) => Math.exp(v - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

export function standardize(model, f) {
  const out = new Float32Array(f.length);
  for (let i = 0; i < f.length; i++) out[i] = (f[i] - model.mean[i]) / model.std[i];
  return out;
}
