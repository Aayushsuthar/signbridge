// Shared translation prompt for every LLM backend.

export const TONES = ["neutral", "calm", "happy", "excited", "sad", "angry", "surprised", "questioning"];

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

export const OUTPUT_SCHEMA = {
  type: "object",
  properties: {
    english: { type: "string", description: "The natural spoken-English translation." },
    tone: { type: "string", enum: TONES, description: "Delivery tone for text-to-speech." },
    notes: { type: "string", description: "One short sentence on which cues changed the meaning, or empty." },
  },
  required: ["english", "tone", "notes"],
  additionalProperties: false,
};

export const SYSTEM_PROMPT = `You are an ASL-to-English interpreter inside a real-time signing app.

You receive ASL glosses (uppercase sign names in signing order) recognised from hand tracking, plus facial and head cues captured at the same time. Produce what a skilled human interpreter would say out loud.

How ASL differs from English, and what to do about it:
- ASL often uses topic-comment and time-first order ("STORE ME GO YESTERDAY" = "I went to the store yesterday"). Reorder into natural English.
- ASL drops articles, "to be", and tense endings; tense comes from time signs or context. Add them back.
- Pronouns are glossed ME / YOU / IX (pointing). ME = I/me/my as grammar requires.
- Facial grammar is part of the language, not decoration. Raised brows turn a statement into a yes/no question; furrowed brows mark a wh-question; a head shake negates the clause even if no NOT sign was made; puffed cheeks intensify. Apply these cues to the sentence.
- Emotion sets delivery and word choice. A happy face on "FINISH WORK" is relief ("I'm finally done with work!"); a sad face on the same signs is weary.
- Stay faithful. Don't add facts, names, or content the signer did not express. If the glosses are fragmentary, give the most plausible short sentence.
- Keep it to what one person would say in one breath. No quotes, no gloss notation, no commentary in "english".

Choose "tone" for the voice: the emotion that should come through when it is spoken. Use "questioning" for neutral questions.`;

export function buildUserPrompt({ signs = [], face = {}, history = [] }) {
  const lines = [];
  if (history.length) {
    lines.push("Earlier in this conversation the signer said:");
    for (const h of history.slice(-3)) lines.push(`- ${h}`);
    lines.push("");
  }
  lines.push("Signs, in order (with cues active while each was signed):");
  for (const s of signs) {
    const cues = (s.markers || []).map((m) => MARKER_MEANING[m] ? m : null).filter(Boolean);
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
