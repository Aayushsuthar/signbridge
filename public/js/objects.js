// Object detection (80 COCO classes, MediaPipe EfficientDet-Lite2) with smoothed tracks, so boxes glide
// instead of flicker, plus a crop helper for sending the tapped object to a vision model.
import { FilesetResolver, ObjectDetector } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite";

export async function createObjectDetector() {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: "VIDEO",
    scoreThreshold: 0.3, // high recall here; the tracker decides what's solid enough to show
    maxResults: 10,
  });
  try {
    return await ObjectDetector.createFromOptions(fileset, opts("GPU"));
  } catch {
    return ObjectDetector.createFromOptions(fileset, opts("CPU"));
  }
}

const iou = (a, b) => {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter || 1);
};

// Tracks objects across frames. Each track keeps a score-weighted vote over every label the detector
// has given it, so a mug doesn't flicker between "cup" and "vase"; a track is shown only once it has
// been seen a few times with a solid average score.
export class ObjectTracker {
  constructor() {
    this.tracks = [];
    this.nextId = 1;
  }

  update(result, now) {
    const dets = (result?.detections || [])
      .filter((d) => d.boundingBox && d.categories?.[0])
      .map((d) => ({
        label: d.categories[0].categoryName,
        score: d.categories[0].score,
        box: { x: d.boundingBox.originX, y: d.boundingBox.originY, w: d.boundingBox.width, h: d.boundingBox.height },
      }));
    const used = new Set();
    for (const tr of this.tracks) {
      let best = null;
      let bestScore = 0;
      dets.forEach((d, i) => {
        if (used.has(i)) return;
        const v = iou(tr.box, d.box);
        // same place is enough when the overlap is strong; weaker overlaps must agree on the label
        const score = d.label === tr.label ? v : v > 0.55 ? v * 0.8 : 0;
        if (score > 0.25 && score > bestScore) [best, bestScore] = [i, score];
      });
      if (best !== null) {
        used.add(best);
        const d = dets[best];
        for (const k of ["x", "y", "w", "h"]) tr.box[k] += (d.box[k] - tr.box[k]) * 0.45;
        tr.votes[d.label] = (tr.votes[d.label] || 0) + d.score;
        tr.scoreSum += d.score;
        tr.lastSeen = now;
        tr.hits++;
        const [label, weight] = Object.entries(tr.votes).sort((a, b) => b[1] - a[1])[0];
        tr.label = label;
        tr.score = weight / tr.hits; // average confidence for the winning label
      }
    }
    dets.forEach((d, i) => {
      if (!used.has(i)) this.tracks.push({ id: this.nextId++, ...d, box: { ...d.box }, votes: { [d.label]: d.score }, scoreSum: d.score, born: now, lastSeen: now, hits: 1 });
    });
    this.tracks = this.tracks.filter((t) => now - t.lastSeen < 700);
    return this.tracks.filter((t) => t.hits >= 4 && t.scoreSum / t.hits >= 0.45);
  }

  clear() {
    this.tracks = [];
  }
}

// Map a box in video pixels to CSS pixels inside a mirrored, object-fit: cover stage.
export function boxToStage(box, video, stageRect) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scale = Math.max(stageRect.width / vw, stageRect.height / vh);
  const ox = (stageRect.width - vw * scale) / 2;
  const oy = (stageRect.height - vh * scale) / 2;
  const left = stageRect.width - (ox + (box.x + box.w) * scale); // mirrored
  return { left, top: oy + box.y * scale, width: box.w * scale, height: box.h * scale };
}

// JPEG of the whole (unmirrored) frame, max 768 px, for "what's in view?"
export function snapshot(video) {
  const s = Math.min(1, 768 / Math.max(video.videoWidth, video.videoHeight));
  const c = document.createElement("canvas");
  c.width = Math.round(video.videoWidth * s);
  c.height = Math.round(video.videoHeight * s);
  c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
  return { base64: c.toDataURL("image/jpeg", 0.85).split(",")[1], preview: c.toDataURL("image/jpeg", 0.6) };
}

// JPEG crop of the object (with some context), max 512 px, base64 without the data: prefix.
export function cropObject(video, box, pad = 0.12) {
  const x = Math.max(0, box.x - box.w * pad);
  const y = Math.max(0, box.y - box.h * pad);
  const w = Math.min(video.videoWidth - x, box.w * (1 + 2 * pad));
  const h = Math.min(video.videoHeight - y, box.h * (1 + 2 * pad));
  const s = Math.min(1, 512 / Math.max(w, h));
  const c = document.createElement("canvas");
  c.width = Math.round(w * s);
  c.height = Math.round(h * s);
  c.getContext("2d").drawImage(video, x, y, w, h, 0, 0, c.width, c.height);
  return { base64: c.toDataURL("image/jpeg", 0.85).split(",")[1], preview: c.toDataURL("image/jpeg", 0.7) };
}
