// Offline fallback translator: no LLM, just enough grammar to produce a readable sentence.
// It is deliberately simple. The LLM backends are where the real translation happens.

const PRONOUNS = { ME: "I", I: "I", MY: "my", YOU: "you", YOUR: "your", HE: "he", SHE: "she", IX: "that", WE: "we", THEY: "they" };
const WH = new Set(["WHO", "WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHICH"]);
const TIME = new Set(["YESTERDAY", "TODAY", "TOMORROW", "NOW", "LATER", "MORNING", "TONIGHT", "BEFORE", "FUTURE"]);
const PAST = new Set(["YESTERDAY", "BEFORE", "FINISH", "PAST"]);
const ADJ = new Set(["HAPPY", "SAD", "TIRED", "HUNGRY", "SICK", "GOOD", "BAD", "FINE", "READY", "BUSY", "HOT", "COLD", "EXCITED", "SORRY", "DEAF", "HEARING"]);
const PHRASES = { "THANK-YOU": "thank you", THANKYOU: "thank you", "HOW-YOU": "how are you", "NICE-MEET-YOU": "nice to meet you", "SEE-YOU-LATER": "see you later", "I-LOVE-YOU": "I love you", ILY: "I love you" };

const TONE_BY_EMOTION = { happy: "happy", sad: "sad", angry: "angry", surprised: "surprised", neutral: "neutral" };

function word(gloss) {
  const g = gloss.toUpperCase();
  if (PHRASES[g]) return PHRASES[g];
  if (PRONOUNS[g]) return PRONOUNS[g];
  return g.toLowerCase().replace(/[-_]+/g, " ");
}

export function translateWithRules({ signs = [], face = {}, lang = "en" }) {
  const glosses = signs.map((s) => String(s.gloss).toUpperCase()).filter((g) => !g.startsWith("_"));
  const markers = new Set([...(face.markers || []), ...signs.flatMap((s) => s.markers || [])]);
  const past = glosses.some((g) => PAST.has(g));

  // time signs lead in ASL but trail in English
  const time = glosses.filter((g) => TIME.has(g));
  const wh = glosses.filter((g) => WH.has(g));
  let core = glosses.filter((g) => !TIME.has(g) && !WH.has(g) && g !== "FINISH" && g !== "PAST");

  let words = core.map(word);
  const subject = words[0];
  const isPronounSubject = ["I", "you", "he", "she", "we", "they"].includes(subject);

  // add a copula before a bare adjective: "ME HAPPY" -> "I am happy"
  if (isPronounSubject && core[1] && ADJ.has(core[1])) {
    const be = past ? (subject === "I" || subject === "he" || subject === "she" ? "was" : "were") : subject === "I" ? "am" : ["he", "she"].includes(subject) ? "is" : "are";
    words.splice(1, 0, be);
  }

  if (markers.has("head-shake") && !core.includes("NOT") && !core.includes("NO")) {
    const i = words.findIndex((w) => ["am", "is", "are", "was", "were"].includes(w));
    if (i >= 0) words.splice(i + 1, 0, "not");
    else if (isPronounSubject) words.splice(1, 0, past ? "didn't" : subject === "he" || subject === "she" ? "doesn't" : "don't");
    else words.unshift("not");
  }
  if (markers.has("cheek-puff")) {
    const i = words.findIndex((w) => ADJ.has(w.toUpperCase()));
    words.splice(i >= 0 ? i : words.length, 0, "really");
  }

  let sentence = [...wh.map(word), ...words, ...time.map(word)].join(" ").replace(/\s+/g, " ").trim();
  const question = wh.length > 0 || markers.has("brows-raised");
  if (!sentence) sentence = "…";
  sentence = sentence.replace(/\bi\b/g, "I");
  sentence = sentence[0].toUpperCase() + sentence.slice(1);
  const emotion = face.emotion || "neutral";
  const exclaim = (emotion === "happy" || emotion === "surprised" || emotion === "angry") && (face.intensity ?? 0) > 0.45;
  sentence += question ? "?" : exclaim ? "!" : ".";

  const tone = question && emotion === "neutral" ? "questioning" : TONE_BY_EMOTION[emotion] || "neutral";
  const notes =
    lang === "en"
      ? "Offline rule-based translation. Connect Ollama or Claude for natural English."
      : "Offline mode can only produce basic English. Connect Ollama or Claude for Hindi and Hinglish.";
  return { text: sentence, speech: sentence, tone, notes };
}
