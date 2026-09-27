// Sanity checks for the parts that don't need a camera: `npm run check`
import assert from "node:assert/strict";
import { translateWithRules } from "../lib/rules.mjs";
import { buildTranslatePrompt, buildDescribePrompt, TRANSLATE_SCHEMA, TONES } from "../lib/prompt.mjs";
import { markChanges } from "../public/js/diff.js";
import { FaceAnalyzer } from "../public/js/face.js";
import { readFileSync } from "node:fs";
import { englishToGloss } from "../lib/to-gloss.mjs";
import { features, standardize, mlpForward, FEATURES } from "../public/js/fingerspell-features.js";

const sign = (gloss, markers = []) => ({ gloss, markers });
let passed = 0;
const test = (name, fn) => {
  fn();
  passed++;
  console.log(`  ✓ ${name}`);
};

const say = (glosses, lang = "en", markers = []) =>
  translateWithRules({ signs: glosses.split(" ").map((g) => sign(g)), face: { emotion: "neutral", markers }, lang }).text;

test("rules: topic-comment + time reorder, in 3 languages", () => {
  assert.equal(say("STORE ME GO YESTERDAY"), "I went to the store yesterday.");
  assert.equal(say("STORE ME GO YESTERDAY", "hi"), "मैं कल दुकान गया।");
  assert.equal(say("STORE ME GO YESTERDAY", "hinglish"), "Main kal dukaan gaya.");
  assert.ok(TONES.includes(translateWithRules({ signs: [sign("ME")] }).tone));
});

test("rules: wh-questions, want, ergative, future", () => {
  assert.equal(say("YOUR NAME WHAT"), "What is your name?");
  assert.equal(say("YOUR NAME WHAT", "hi"), "तुम्हारा नाम क्या है?");
  assert.equal(say("ME WANT WATER", "hinglish"), "Mujhe paani chahiye.");
  assert.equal(say("ME FINISH WORK", "hi"), "मैंने काम खत्म किया।");
  assert.equal(say("TOMORROW ME GO WORK"), "I will go to work tomorrow.");
  assert.equal(say("FRIEND HELP ME", "hinglish"), "Dost meri madad karta hai.");
  assert.equal(say("HELLO ME HAPPY TODAY"), "Hello! I am happy today.");
});

test("rules: copula + raised brows makes a question", () => {
  const r = translateWithRules({ signs: [sign("YOU"), sign("HUNGRY")], face: { emotion: "neutral", markers: ["brows-raised"] } });
  assert.equal(r.text, "Are you hungry?");
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

test("rules: yes/no question and negation in Hindi", () => {
  assert.equal(say("YOU GO STORE YESTERDAY", "hi", ["brows-raised"]), "क्या तुम कल दुकान गए?");
  assert.equal(say("ME GO SCHOOL", "hinglish", ["head-shake"]), "Main school nahi jata.");
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

test("fingerspelling model: loads, right shape, recognises its own letter templates", () => {
  const model = JSON.parse(readFileSync(new URL("../public/models/fingerspelling.json", import.meta.url)));
  assert.equal(model.mean.length, FEATURES);
  assert.equal(model.labels.length, 26);
  let ok = 0;
  for (const letter of model.labels) {
    const probs = mlpForward(model, standardize(model, features(model.templates[letter])));
    const top3 = probs.map((p, i) => [p, model.labels[i]]).sort((a, b) => b[0] - a[0]).slice(0, 3).map((x) => x[1]);
    if (top3.includes(letter)) ok++;
  }
  assert.ok(ok >= 24, `only ${ok}/26 templates recognised`);
});

test("word model files are present with 250 labels", () => {
  const labels = JSON.parse(readFileSync(new URL("../public/models/asl-signs-labels.json", import.meta.url)));
  assert.equal(Object.keys(labels).length, 250);
  assert.ok(readFileSync(new URL("../public/models/asl-signs.tflite", import.meta.url)).length > 1e6);
});

test("English → ASL gloss: time first, wh last, names fingerspelled", () => {
  const g = (t) => englishToGloss(t).map((x) => (x.video ? x.word : `fs:${x.word}`)).join(" ");
  assert.equal(g("I went to the store yesterday."), "yesterday me go store");
  assert.equal(g("What is your name?"), "your name what");
  assert.equal(g("My name is Aayush"), "my name fs:aayush");
  assert.equal(g("Thank you!"), "thank you");
});

test("object knowledge base: every entry complete, names unique", () => {
  const kb = JSON.parse(readFileSync(new URL("../public/models/objects-kb.json", import.meta.url)));
  assert.ok(kb.length >= 150);
  assert.equal(new Set(kb.map((o) => o.name)).size, kb.length);
  for (const o of kb) assert.ok(o.name && o.hi && o.cat && o.use, `incomplete: ${o.name}`);
});

test("object model matches the knowledge base and has a prototype for every object", () => {
  const kb = JSON.parse(readFileSync(new URL("../public/models/objects-kb.json", import.meta.url)));
  const model = JSON.parse(readFileSync(new URL("../public/models/objects-clip.json", import.meta.url)));
  assert.deepEqual(model.names, kb.map((o) => o.name));
  assert.equal(model.weights[0].length, model.dim);
  assert.ok(model.prototypes.filter(Boolean).length >= kb.length - 2);
  assert.ok(model.meta.test.adapter.top1 > 0.85, "held-out accuracy regressed");
});

console.log(`\n${passed} checks passed`);
