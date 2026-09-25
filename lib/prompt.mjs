// Prompts and output schemas shared by every LLM backend.

export const TONES = ["neutral", "calm", "happy", "excited", "sad", "angry", "surprised", "questioning"];
export const LANGS = ["en", "hi", "hinglish"];

const LANG_RULE = {
  en: "Write in natural, conversational spoken English.",
  hi: "Write in natural, conversational spoken Hindi, in Devanagari script (e.g. 'मैं कल दुकान गया था।'). Use everyday Hindi, not formal/Sanskritised words.",
  hinglish:
    "Write in Hinglish: the everyday Hindi-English mix used in India, in Roman script (e.g. 'Main kal store gaya tha, bohot thak gaya yaar.'). Keep English words people normally say in English.",
};

const SPEECH_RULE = `"speech" is the same content prepared for an expressive text-to-speech voice. Keep the words identical to "text", but you may add at most two ElevenLabs audio tags in square brackets where they fit the emotion, e.g. [excited], [cheerful], [laughs softly], [sighs], [curious], [whispers], [sad], [surprised]. No other markup.`;

// Non-manual markers the browser can report, and what they mean grammatically in ASL.
export const MARKER_MEANING = {
  "brows-raised": "raised eyebrows: yes/no question (or a topic being set up)",
  "brows-furrowed": "furrowed eyebrows: wh-question (who/what/where/when/why/how), or displeasure",
  "head-shake": "head shaking: negation. The statement is NOT true / the signer does NOT",
  "head-nod": "head nodding: affirmation or emphasis. It IS true / the signer DOES",
  "cheek-puff": "puffed cheeks: intensifier, meaning a lot / very big / huge amount",
  "mouth-pursed": "'mm' mouth: normally, easily, with pleasure",
  "tongue-out": "'th' mouth: carelessly, sloppily, without paying attention",
};

// ---------- sign translation ----------

export const TRANSLATE_SCHEMA = {
  type: "object",
  properties: {
    text: { type: "string", description: "The natural translation, in the requested language." },
    speech: { type: "string", description: "The same words, with optional audio tags for the voice." },
    tone: { type: "string", enum: TONES },
    notes: { type: "string", description: "One short English sentence on which cues changed the meaning, or empty." },
  },
  required: ["text", "speech", "tone", "notes"],
  additionalProperties: false,
};

export const TRANSLATE_SYSTEM = `You are a sign-language interpreter inside a real-time signing app.

You receive ASL glosses (uppercase sign names in signing order) recognised from hand tracking, plus facial and head cues captured at the same time. Produce what a skilled human interpreter would say out loud.

How ASL differs from spoken languages, and what to do about it:
- ASL often uses topic-comment and time-first order ("STORE ME GO YESTERDAY" = "I went to the store yesterday"). Reorder naturally.
- ASL drops articles, "to be", and tense endings; tense comes from time signs or context. Add them back.
- Pronouns are glossed ME / YOU / IX (pointing). ME = I/me/my as grammar requires.
- Facial grammar is part of the language, not decoration. Raised brows turn a statement into a yes/no question; furrowed brows mark a wh-question; a head shake negates the clause even if no NOT sign was made; puffed cheeks intensify. Apply these cues.
- Emotion sets delivery and word choice. A happy face on "FINISH WORK" is relief ("I'm finally done with work!"); a sad face on the same signs is weary.
- Stay faithful. Don't add facts, names, or content the signer did not express. If the glosses are fragmentary, give the most plausible short sentence.
- One breath: what one person would say. No quotes, no gloss notation, no commentary in "text".

${SPEECH_RULE}
Choose "tone" for the voice: the emotion that should come through. Use "questioning" for neutral questions.`;

export function buildTranslatePrompt({ signs = [], face = {}, history = [], lang = "en" }) {
  const lines = [`Output language: ${LANG_RULE[lang] || LANG_RULE.en}`, ""];
  if (history.length) {
    lines.push("Earlier in this conversation the signer said:");
    for (const h of history.slice(-3)) lines.push(`- ${h}`);
    lines.push("");
  }
  lines.push("Signs, in order (with cues active while each was signed):");
  for (const s of signs) {
    const cues = (s.markers || []).filter((m) => MARKER_MEANING[m]);
    lines.push(`- ${String(s.gloss).toUpperCase()}${cues.length ? `  [${cues.join(", ")}]` : ""}`);
  }
  lines.push("");
  const emotion = face.emotion || "neutral";
  lines.push(`Dominant facial emotion across the sentence: ${emotion}${face.intensity ? ` (intensity ${Number(face.intensity).toFixed(2)})` : ""}`);
  const markers = (face.markers || []).filter((m) => MARKER_MEANING[m]);
  if (markers.length) {
    lines.push("Facial/head grammar detected across the sentence:");
    for (const m of markers) lines.push(`- ${MARKER_MEANING[m]}`);
  } else {
    lines.push("No grammatical facial markers detected: treat it as a plain statement.");
  }
  lines.push("", "Return the JSON object.");
  return lines.join("\n");
}

// ---------- object description ----------

export const DESCRIBE_SCHEMA = {
  type: "object",
  properties: {
    name: { type: "string", description: "What the object is, as specifically as you can tell, in the requested language." },
    text: { type: "string", description: "2-3 spoken sentences: what it is and the most useful or interesting thing about it." },
    speech: { type: "string", description: "The same words as text, with optional audio tags." },
    facts: { type: "array", items: { type: "string" }, description: "Exactly 3 short, true, interesting facts." },
    tone: { type: "string", enum: TONES },
  },
  required: ["name", "text", "speech", "facts", "tone"],
  additionalProperties: false,
};

export const DESCRIBE_SYSTEM = `You are a warm, knowledgeable guide inside a camera app used by Deaf and hearing people alike. The camera has detected an object and the user tapped it to learn about it.

Describe it the way a friendly expert would when pointing at it: say what it is (be specific if an image is provided: colour, type, brand only if clearly visible), then the single most useful or interesting thing about it. Keep facts accurate; if unsure, stay general rather than invent. Never identify real people by name. For a person, describe only what's visible (e.g. clothing, activity) in a kind way.

${SPEECH_RULE}
"tone" is usually "happy" or "excited" for fun objects and "calm" otherwise.`;

export function buildDescribePrompt({ label, score, lang = "en", wiki, hasImage }) {
  return [
    `Output language: ${LANG_RULE[lang] || LANG_RULE.en}`,
    "",
    `Detector label: "${label}" (confidence ${Number(score || 0).toFixed(2)}, from the 80 COCO object classes).`,
    hasImage ? "A cropped photo of the object is attached. Trust the photo over the label if they disagree." : "No photo is available; describe the object class in general.",
    wiki ? `\nReference (Wikipedia summary, for accuracy):\n${wiki}` : "",
    "",
    "Return the JSON object.",
  ].join("\n");
}
