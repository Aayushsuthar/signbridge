// Pretrained ASL fingerspelling (A–Z) from one hand's landmarks, plus finger-by-finger coaching for Learn mode.
// The model is trained by scripts/train-fingerspelling.mjs; features come from fingerspell-features.js so
// training and the browser compute exactly the same thing.
import { normalize, features, standardize, mlpForward } from "./fingerspell-features.js";

export const LETTER_TIPS = {
  A: "Make a fist. Rest your thumb against the side of your index finger.",
  B: "Hold four fingers straight up, together. Fold your thumb across your palm.",
  C: "Curve your fingers and thumb into the shape of a C.",
  D: "Point your index finger up. Touch your other fingertips to your thumb.",
  E: "Bend all fingertips down so they rest on your thumb, tucked under them.",
  F: "Touch your index fingertip to your thumb. Keep the other three fingers up and spread.",
  G: "Point your index finger and thumb sideways, parallel, with a small gap.",
  H: "Point your index and middle fingers sideways, together.",
  I: "Raise your pinky. Keep the other fingers in a fist, thumb across them.",
  J: "Raise your pinky, then trace a J in the air: down and hook toward you.",
  K: "Raise your index and middle fingers in a V. Touch your thumb to the middle finger.",
  L: "Make an L with your thumb and index finger.",
  M: "Tuck your thumb under your first three fingers.",
  N: "Tuck your thumb under your first two fingers.",
  O: "Touch all your fingertips to your thumb to make an O.",
  P: "Make a K, then turn it so your fingers point down.",
  Q: "Make a G, then turn it so your fingers point down.",
  R: "Cross your middle finger over your index finger.",
  S: "Make a fist with your thumb across the front of your fingers.",
  T: "Tuck your thumb between your index and middle fingers.",
  U: "Raise your index and middle fingers, held together.",
  V: "Raise your index and middle fingers, spread apart.",
  W: "Raise your index, middle and ring fingers, spread apart.",
  X: "Bend your index finger into a hook. Keep the others in a fist.",
  Y: "Stick out your thumb and pinky. Fold the other fingers down.",
  Z: "Point your index finger and trace a Z in the air.",
};
export const MOTION_LETTERS = new Set(["J", "Z"]);

const FINGERS = [
  { name: "thumb", chain: [1, 2, 3, 4] },
  { name: "index finger", chain: [5, 6, 7, 8] },
  { name: "middle finger", chain: [9, 10, 11, 12] },
  { name: "ring finger", chain: [13, 14, 15, 16] },
  { name: "pinky", chain: [17, 18, 19, 20] },
];

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const angle = (a, b) => {
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return Math.acos(Math.max(-1, Math.min(1, d / ((Math.hypot(...a) || 1e-6) * (Math.hypot(...b) || 1e-6)))));
};
// total bend of a finger: 0 = straight, ~3 = fully curled
function curl(norm, chain) {
  const pts = [norm[0], ...chain.map((i) => norm[i])];
  let c = 0;
  for (let j = 1; j < pts.length - 1; j++) c += angle(sub(pts[j], pts[j - 1]), sub(pts[j + 1], pts[j]));
  return c;
}

export class FingerspellRecognizer {
  async init(url = "/models/fingerspelling.json") {
    this.model = await (await fetch(url)).json();
    this.labels = this.model.labels;
    return this;
  }

  // landmarks: one hand from HandLandmarker; aspect: video width / height (square-pixel coordinates)
  static points(landmarks, aspect) {
    return landmarks.map((p) => [p.x * aspect, p.y, p.z * aspect]);
  }

  predict(landmarks, aspect, k = 3) {
    const probs = mlpForward(this.model, standardize(this.model, features(normalize(FingerspellRecognizer.points(landmarks, aspect)))));
    return probs
      .map((p, i) => ({ letter: this.labels[i], p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, k);
  }

  template(letter) {
    return this.model.templates?.[letter] || null;
  }

  // The single most useful correction to get from the user's hand to the target letter's shape.
  hint(letter, landmarks, aspect) {
    const tpl = this.template(letter);
    if (!tpl || !landmarks) return null;
    const mine = normalize(FingerspellRecognizer.points(landmarks, aspect));
    let worst = null;
    for (const f of FINGERS) {
      const diff = curl(mine, f.chain) - curl(tpl, f.chain);
      if (Math.abs(diff) > 0.7 && (!worst || Math.abs(diff) > Math.abs(worst.diff))) worst = { finger: f.name, diff };
    }
    if (worst) return worst.diff > 0 ? `Straighten your ${worst.finger}` : `Curl your ${worst.finger} more`;
    // orientation: where the hand points (wrist → middle knuckle), mirror-independent
    const dir = (n) => Math.atan2(n[9][1], Math.abs(n[9][0]));
    const turn = dir(mine) - dir(tpl);
    if (Math.abs(turn) > 0.9) return turn > 0 ? "Point your fingers more upward" : "Point your fingers more downward";
    return null;
  }
}
