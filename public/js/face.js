// Reads the face: emotion for the voice, and ASL non-manual markers (facial grammar) for the translator.
// Everything comes from MediaPipe's 52 blendshapes plus the head-pose matrix.

export const EMOTIONS = ["neutral", "happy", "sad", "surprised", "angry"];
export const EMOJI = { neutral: "😐", happy: "😊", sad: "😢", surprised: "😮", angry: "😠" };
export const MARKER_LABEL = {
  "brows-raised": "Brows up · yes/no Q",
  "brows-furrowed": "Brows down · wh-Q",
  "head-shake": "Head shake · NOT",
  "head-nod": "Nod · affirm",
  "cheek-puff": "Cheek puff · very",
  "mouth-pursed": "'mm' · easily",
  "tongue-out": "'th' · carelessly",
};

const clamp = (v) => Math.max(0, Math.min(1, v));
const avg = (a, b) => (a + b) / 2;

// Detect back-and-forth head motion: count direction reversals bigger than `step` degrees.
function oscillation(samples, step) {
  if (samples.length < 6) return { reversals: 0, range: 0 };
  let reversals = 0;
  let dir = 0; // 0 = undecided, 1 = rising, -1 = falling
  let extreme = samples[0].v;
  let min = Infinity;
  let max = -Infinity;
  for (const { v } of samples) {
    min = Math.min(min, v);
    max = Math.max(max, v);
    if (dir === 0) {
      if (v - extreme > step) [dir, extreme] = [1, v];
      else if (extreme - v > step) [dir, extreme] = [-1, v];
    } else if (dir === 1) {
      if (v > extreme) extreme = v;
      else if (extreme - v > step) [reversals, dir, extreme] = [reversals + 1, -1, v];
    } else {
      if (v < extreme) extreme = v;
      else if (v - extreme > step) [reversals, dir, extreme] = [reversals + 1, 1, v];
    }
  }
  return { reversals, range: max - min };
}

export class FaceAnalyzer {
  constructor() {
    this.baseline = null;
    this.calibration = null;
    this.smooth = {};
    this.yaw = [];
    this.pitch = [];
    this.recent = []; // { t, markers: string[] } for per-sign markers
    this.state = { present: false, emotion: "neutral", intensity: 0, scores: {}, markers: [], values: {} };
    this.resetSentence();
  }

  // Record ~1.5 s of the user's resting face so a naturally raised brow or smile doesn't read as a cue.
  calibrate(ms = 1500) {
    return new Promise((resolve) => {
      this.calibration = { until: performance.now() + ms, sums: {}, n: 0, resolve };
    });
  }

  update(result, now) {
    const cats = result?.faceBlendshapes?.[0]?.categories;
    if (!cats) {
      this.state = { ...this.state, present: false, markers: [] };
      return this.state;
    }
    const raw = {};
    for (const c of cats) raw[c.categoryName] = c.score;

    if (this.calibration) {
      const cal = this.calibration;
      for (const k in raw) cal.sums[k] = (cal.sums[k] || 0) + raw[k];
      cal.n++;
      if (now >= cal.until && cal.n > 5) {
        this.baseline = {};
        for (const k in cal.sums) this.baseline[k] = cal.sums[k] / cal.n;
        this.calibration = null;
        cal.resolve(true);
      }
    }

    const b = (name) => {
      const base = this.baseline?.[name] ?? 0;
      const v = clamp(((raw[name] ?? 0) - base) / (1 - Math.min(base, 0.9)));
      const prev = this.smooth[name] ?? v;
      return (this.smooth[name] = prev + (v - prev) * 0.45);
    };

    const smile = avg(b("mouthSmileLeft"), b("mouthSmileRight"));
    const frown = avg(b("mouthFrownLeft"), b("mouthFrownRight"));
    const browDown = avg(b("browDownLeft"), b("browDownRight"));
    const browInner = b("browInnerUp");
    const browOuter = avg(b("browOuterUpLeft"), b("browOuterUpRight"));
    const eyeWide = avg(b("eyeWideLeft"), b("eyeWideRight"));
    const jaw = b("jawOpen");
    const sneer = avg(b("noseSneerLeft"), b("noseSneerRight"));
    const press = avg(b("mouthPressLeft"), b("mouthPressRight"));
    const squint = avg(b("cheekSquintLeft"), b("cheekSquintRight"));
    const cheekPuff = b("cheekPuff");
    const roll = avg(b("mouthRollLower"), b("mouthRollUpper"));
    const pucker = b("mouthPucker");
    const tongue = b("tongueOut");

    const scores = {
      neutral: 0.22,
      happy: clamp(smile * 1.1 + squint * 0.3),
      sad: clamp(frown * 1.3 + browInner * 0.45 - smile * 0.6),
      surprised: clamp(eyeWide * 0.5 + browOuter * 0.45 + jaw * 0.45 - smile * 0.3),
      angry: clamp(browDown * 0.85 + sneer * 0.6 + press * 0.3 - smile * 0.5),
    };
    let emotion = "neutral";
    for (const e of EMOTIONS) if (scores[e] > scores[emotion]) emotion = e;

    // head pose from the 4x4 column-major transform: row 2 of the rotation is [m2, m6, m10]
    const m = result.facialTransformationMatrixes?.[0]?.data;
    if (m) {
      const yaw = (Math.atan2(-m[2], Math.hypot(m[6], m[10])) * 180) / Math.PI;
      const pitch = (Math.atan2(m[6], m[10]) * 180) / Math.PI;
      this.yaw.push({ t: now, v: yaw });
      this.pitch.push({ t: now, v: pitch });
    }
    while (this.yaw.length && now - this.yaw[0].t > 1300) this.yaw.shift();
    while (this.pitch.length && now - this.pitch[0].t > 1300) this.pitch.shift();
    const shake = oscillation(this.yaw, 3.5);
    const nod = oscillation(this.pitch, 3);

    const markers = [];
    const browRaise = avg(browInner, browOuter);
    if (browRaise > 0.32 && browDown < 0.25) markers.push("brows-raised");
    if (browDown > 0.33 && browRaise < 0.3) markers.push("brows-furrowed");
    if (shake.reversals >= 2 && shake.range > 8) markers.push("head-shake");
    else if (nod.reversals >= 2 && nod.range > 6) markers.push("head-nod");
    if (cheekPuff > 0.35) markers.push("cheek-puff");
    if (roll > 0.35 || (pucker > 0.55 && jaw < 0.15)) markers.push("mouth-pursed");
    if (tongue > 0.25) markers.push("tongue-out");

    this.recent.push({ t: now, markers });
    while (this.recent.length && now - this.recent[0].t > 900) this.recent.shift();

    this.state = {
      present: true,
      emotion,
      intensity: scores[emotion] === 0.22 && emotion === "neutral" ? 0 : scores[emotion],
      scores,
      markers,
      values: { browRaise, browDown, smile, frown, cheekPuff },
    };
    this.accumulate();
    return this.state;
  }

  // Markers active for a good part of the last ~0.9 s, attached to a sign when it's committed.
  recentMarkers() {
    const n = this.recent.length || 1;
    const counts = {};
    for (const r of this.recent) for (const mk of r.markers) counts[mk] = (counts[mk] || 0) + 1;
    return Object.keys(counts).filter((mk) => counts[mk] / n >= (mk === "head-shake" || mk === "head-nod" ? 0.12 : 0.3));
  }

  // ----- whole-sentence summary -----

  resetSentence() {
    this.sentence = { frames: 0, scores: Object.fromEntries(EMOTIONS.map((e) => [e, 0])), markers: {} };
  }

  accumulate() {
    const s = this.sentence;
    if (!this.collecting) return;
    s.frames++;
    for (const e of EMOTIONS) s.scores[e] += this.state.scores[e];
    for (const mk of this.state.markers) s.markers[mk] = (s.markers[mk] || 0) + 1;
  }

  summary() {
    const s = this.sentence;
    const n = Math.max(s.frames, 1);
    let emotion = "neutral";
    for (const e of EMOTIONS) if (s.scores[e] > s.scores[emotion] * (emotion === "neutral" ? 1.05 : 1)) emotion = e;
    const markers = Object.keys(s.markers).filter(
      (mk) => s.markers[mk] / n >= (mk === "head-shake" || mk === "head-nod" ? 0.06 : 0.2),
    );
    return { emotion, intensity: Number((s.scores[emotion] / n).toFixed(2)), markers, frames: s.frames };
  }
}
