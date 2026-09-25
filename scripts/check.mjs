// Sanity checks for the parts that don't need a camera: `npm run check`
import assert from "node:assert/strict";
import { translateWithRules } from "../lib/rules.mjs";
import { buildTranslatePrompt, buildDescribePrompt, TRANSLATE_SCHEMA, TONES } from "../lib/prompt.mjs";
import { markChanges } from "../public/js/diff.js";
import { FaceAnalyzer } from "../public/js/face.js";

const sign = (gloss, markers = []) => ({ gloss, markers });
let passed = 0;
const test = (name, fn) => {
  fn();
  passed++;
  console.log(`  ✓ ${name}`);
};

test("rules: topic-comment + time reorder", () => {
  const r = translateWithRules({ signs: ["STORE", "ME", "GO", "YESTERDAY"].map((g) => sign(g)) });
  assert.match(r.text, /yesterday\.$/i);
  assert.ok(TONES.includes(r.tone));
});

test("rules: copula + raised brows makes a question", () => {
  const r = translateWithRules({ signs: [sign("YOU"), sign("HUNGRY")], face: { emotion: "neutral", markers: ["brows-raised"] } });
  assert.equal(r.text, "You are hungry?");
  assert.equal(r.tone, "questioning");
});

test("rules: head shake negates", () => {
  const r = translateWithRules({ signs: [sign("ME"), sign("TIRED")], face: { emotion: "neutral", markers: ["head-shake"] } });
  assert.equal(r.text, "I am not tired.");
});

test("prompt includes cues and history", () => {
  const p = buildTranslatePrompt({ signs: [sign("YOU", ["brows-raised"]), sign("HUNGRY")], face: { emotion: "happy", markers: ["cheek-puff"] }, history: ["Hi!"] });
  assert.match(p, /YOU {2}\[brows-raised\]/);
  assert.match(p, /puffed cheeks/);
  assert.match(p, /Hi!/);
  assert.deepEqual(TRANSLATE_SCHEMA.required, ["text", "speech", "tone", "notes"]);
});

test("prompts ask for the chosen language", () => {
  assert.match(buildTranslatePrompt({ signs: [sign("ME")], lang: "hi" }), /Devanagari/);
  assert.match(buildTranslatePrompt({ signs: [sign("ME")], lang: "hinglish" }), /Roman script/);
  assert.match(buildDescribePrompt({ label: "cup", score: 0.9, lang: "en", hasImage: true }), /photo of the object is attached/);
});

test("rules: non-English request explains the limitation", () => {
  const r = translateWithRules({ signs: [sign("ME"), sign("HAPPY")], lang: "hi" });
  assert.equal(r.text, "I am happy.");
  assert.match(r.notes, /Hindi/);
});

test("diff underlines only AI-added words", () => {
  const parts = markChanges("I went to the store yesterday.", ["STORE", "ME", "GO", "YESTERDAY"]);
  const changed = parts.filter((p) => p.changed).map((p) => p.text);
  assert.deepEqual(changed, ["to", "the"]);
});

test("face: head shake detected from yaw oscillation", () => {
  const f = new FaceAnalyzer();
  const cats = [{ categoryName: "mouthSmileLeft", score: 0 }];
  let marked = false;
  for (let i = 0; i < 40; i++) {
    const yaw = 14 * Math.sin(i / 3); // degrees, back and forth
    const r = (yaw * Math.PI) / 180;
    const m = [Math.cos(r), 0, -Math.sin(r), 0, 0, 1, 0, 0, Math.sin(r), 0, Math.cos(r), 0, 0, 0, 0, 1];
    const s = f.update({ faceBlendshapes: [{ categories: cats }], facialTransformationMatrixes: [{ data: m }] }, i * 33);
    marked ||= s.markers.includes("head-shake");
  }
  assert.ok(marked);
});

test("face: still head is not a shake, smile reads as happy", () => {
  const f = new FaceAnalyzer();
  const cats = [
    { categoryName: "mouthSmileLeft", score: 0.8 },
    { categoryName: "mouthSmileRight", score: 0.8 },
  ];
  const m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  let s;
  for (let i = 0; i < 30; i++) s = f.update({ faceBlendshapes: [{ categories: cats }], facialTransformationMatrixes: [{ data: m }] }, i * 33);
  assert.equal(s.emotion, "happy");
  assert.ok(!s.markers.includes("head-shake"));
});

console.log(`\n${passed} checks passed`);
