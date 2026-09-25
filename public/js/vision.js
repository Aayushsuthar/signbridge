// MediaPipe setup, drawing, and the per-frame feature vector the sign classifier learns from.
import {
  FilesetResolver,
  HandLandmarker,
  FaceLandmarker,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

export async function createTrackers() {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const make = (delegate) =>
    Promise.all([
      HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: HAND_MODEL, delegate },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      }),
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      }),
    ]);
  try {
    const [hands, face] = await make("GPU");
    return { hands, face, delegate: "GPU" };
  } catch (err) {
    console.warn("GPU delegate failed, falling back to CPU", err);
    const [hands, face] = await make("CPU");
    return { hands, face, delegate: "CPU" };
  }
}

// ---------- drawing ----------

export class Overlay {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.draw = new DrawingUtils(this.ctx);
  }

  resize(w, h) {
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  render(handResult, faceResult, { showFace = true, showHands = true } = {}) {
    const { ctx } = this;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    const face = faceResult?.faceLandmarks?.[0];
    if (showFace && face) {
      this.draw.drawConnectors(face, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "rgba(120, 200, 255, 0.10)", lineWidth: 0.6 });
      this.draw.drawConnectors(face, FaceLandmarker.FACE_LANDMARKS_CONTOURS, { color: "rgba(120, 200, 255, 0.55)", lineWidth: 1.2 });
      this.draw.drawLandmarks(face, { color: "rgba(150, 220, 255, 0.55)", radius: 0.7, lineWidth: 0 });
    }
    if (showHands) {
      (handResult?.landmarks || []).forEach((hand, i) => {
        const side = handResult.handedness?.[i]?.[0]?.categoryName;
        const color = side === "Left" ? "#ff9f43" : "#2ee6a6";
        this.draw.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color, lineWidth: 3 });
        this.draw.drawLandmarks(hand, { color: "#ffffff", fillColor: color, radius: 3.5, lineWidth: 1 });
      });
    }
  }
}

// ---------- features ----------
//
// Per hand slot (Left, Right):
//   shape    63  every landmark relative to the wrist, scaled by palm size (handshape + palm orientation)
//   location  3  wrist position relative to the nose, in face widths, plus apparent hand size (where the sign is made)
//   motion    2  wrist travel over the last few frames, in face widths (movement)
//   present   1
// Handshape, orientation, location and movement are four of the five ASL parameters; the fifth
// (non-manual markers) comes from the face analyzer and goes to the LLM rather than the classifier.

const SHAPE_W = 1.0;
const LOC_W = 2.5;
const MOTION_W = 2.0;
const PRESENT_W = 3.0;
const MOTION_FRAMES = 8;
const SLOT_DIM = 63 + 3 + 2 + 1;
export const FEATURE_DIM = SLOT_DIM * 2;

export class FeatureExtractor {
  constructor() {
    this.trail = { Left: [], Right: [] };
  }

  frame(handResult, faceResult, aspect) {
    const vec = new Float32Array(FEATURE_DIM);
    const face = faceResult?.faceLandmarks?.[0];
    const anchor = face ? { x: face[1].x * aspect, y: face[1].y } : { x: 0.5 * aspect, y: 0.45 };
    const faceWidth = face ? Math.hypot((face[454].x - face[234].x) * aspect, face[454].y - face[234].y) || 0.2 : 0.2;

    // assign hands to slots; if both claim the same side, the second takes the free slot
    const slots = {};
    (handResult?.landmarks || []).forEach((hand, i) => {
      let side = handResult.handedness?.[i]?.[0]?.categoryName === "Left" ? "Left" : "Right";
      if (slots[side]) side = side === "Left" ? "Right" : "Left";
      if (!slots[side]) slots[side] = hand;
    });

    let count = 0;
    ["Left", "Right"].forEach((side, s) => {
      const hand = slots[side];
      const trail = this.trail[side];
      if (!hand) {
        trail.length = 0;
        return;
      }
      count++;
      const o = s * SLOT_DIM;
      const w = hand[0];
      const palm = Math.hypot((hand[9].x - w.x) * aspect, hand[9].y - w.y) || 1e-3;
      for (let j = 0; j < 21; j++) {
        vec[o + j * 3] = (((hand[j].x - w.x) * aspect) / palm) * SHAPE_W;
        vec[o + j * 3 + 1] = ((hand[j].y - w.y) / palm) * SHAPE_W;
        vec[o + j * 3 + 2] = ((hand[j].z - w.z) / palm) * 0.5 * SHAPE_W;
      }
      const wx = w.x * aspect;
      vec[o + 63] = ((wx - anchor.x) / faceWidth) * LOC_W;
      vec[o + 64] = ((w.y - anchor.y) / faceWidth) * LOC_W;
      vec[o + 65] = (palm / faceWidth) * LOC_W;
      trail.push({ x: wx, y: w.y });
      if (trail.length > MOTION_FRAMES) trail.shift();
      const first = trail[0];
      vec[o + 66] = ((wx - first.x) / faceWidth) * MOTION_W;
      vec[o + 67] = ((w.y - first.y) / faceWidth) * MOTION_W;
      vec[o + 68] = PRESENT_W;
    });

    return { vec, count };
  }
}
