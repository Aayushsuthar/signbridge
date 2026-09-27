// CLIP (OpenAI ViT-B/32, 8-bit quantized ONNX via transformers.js) for open-vocabulary object recognition.
// Image and text land in the same embedding space, so an image can be matched against any list of
// object names. The app only needs the vision half at runtime: the object classifier (text prompts
// mixed with prototypes learned from labelled photos) is precomputed into public/models/objects-clip.json.
import {
  AutoTokenizer,
  CLIPTextModelWithProjection,
  AutoProcessor,
  CLIPVisionModelWithProjection,
  RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";

export const CLIP_MODEL = "Xenova/clip-vit-base-patch32";

// WebGPU fp16 is ~14× faster than CPU (24 ms vs 340 ms per image, measured); CPU 8-bit is the fallback.
export const CLIP_RUNTIME = globalThis.navigator?.gpu ? { device: "webgpu", dtype: "fp16" } : { device: "wasm", dtype: "q8" };

// Prompt ensemble: averaging several phrasings is a standard, reliable boost for zero-shot CLIP.
export const TEMPLATES = [
  "a photo of a {}.",
  "a close-up photo of a {}.",
  "a photo of a {} on a table.",
  "a photo of a {} held in a hand.",
  "a blurry photo of a {}.",
  "a photo of the {} at home.",
];

export function l2(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  s = Math.sqrt(s) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / s;
  return out;
}

export class Clip {
  async initVision() {
    [this.processor, this.vision] = await Promise.all([
      AutoProcessor.from_pretrained(CLIP_MODEL),
      CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL, CLIP_RUNTIME).catch(() =>
        CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL, { device: "wasm", dtype: "q8" }),
      ),
    ]);
    return this;
  }

  async initText() {
    [this.tokenizer, this.text] = await Promise.all([
      AutoTokenizer.from_pretrained(CLIP_MODEL),
      CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL, CLIP_RUNTIME).catch(() =>
        CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL, { device: "wasm", dtype: "q8" }),
      ),
    ]);
    return this;
  }

  // source: canvas, ImageBitmap-drawn canvas, or Blob
  async embedImage(source) {
    const image = source instanceof Blob ? await RawImage.fromBlob(source) : RawImage.fromCanvas(source);
    const inputs = await this.processor(image);
    const { image_embeds } = await this.vision(inputs);
    return l2(image_embeds.data);
  }

  // one averaged, normalized embedding per name, over the prompt templates
  async embedNames(names) {
    const out = [];
    for (const name of names) {
      const texts = TEMPLATES.map((t) => t.replace("{}", name));
      const inputs = this.tokenizer(texts, { padding: true, truncation: true });
      const { text_embeds } = await this.text(inputs);
      const dim = text_embeds.dims[1];
      const mean = new Float32Array(dim);
      for (let r = 0; r < texts.length; r++) {
        const row = l2(text_embeds.data.subarray(r * dim, (r + 1) * dim));
        for (let i = 0; i < dim; i++) mean[i] += row[i] / texts.length;
      }
      out.push(l2(mean));
    }
    return out;
  }
}

// Classifier built by scripts/train-objects.mjs: zero-shot text weights for every object, plus
// visual prototypes (Tip-Adapter style) for objects that had training photos. Mirrors the trainer's
// logits exactly, so the accuracy measured there is what the app gets.
export class ObjectClassifier {
  constructor(model) {
    this.model = model;
    this.W = model.weights.map((w) => Float32Array.from(w));
    this.P = model.prototypes.map((p) => (p ? Float32Array.from(p) : null));
  }

  static dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  // embedding: normalized CLIP image embedding → top-k [{ id, name, p }]
  classify(embedding, k = 3) {
    const { scale, alpha, beta } = this.model;
    const logits = this.W.map((w) => scale * ObjectClassifier.dot(embedding, w));
    if (alpha) {
      const aff = [];
      this.P.forEach((p, i) => p && aff.push([i, Math.exp(-beta * (1 - ObjectClassifier.dot(embedding, p)))]));
      const mean = aff.reduce((a, [, v]) => a + v, 0) / (aff.length || 1);
      for (const [i, v] of aff) logits[i] += alpha * (v - mean);
    }
    const max = Math.max(...logits);
    const e = logits.map((l) => Math.exp(l - max));
    const sum = e.reduce((a, b) => a + b, 0);
    return e
      .map((v, i) => ({ id: this.model.ids[i], name: this.model.names[i], p: v / sum }))
      .sort((a, b) => b.p - a.p)
      .slice(0, k);
  }
}
