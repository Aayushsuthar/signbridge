// English → ASL gloss, offline. ASL isn't English word-for-word: time comes first, articles and
// "to be" drop out, question words go last. Words with a sign video play it; anything else
// (names, rare words) is fingerspelled. The LLM path in server.mjs does this better and handles Hindi.
import { readFileSync } from "node:fs";

export const SIGN_VIDEOS = JSON.parse(readFileSync(new URL("../public/models/sign-videos.json", import.meta.url), "utf8"));
export const VOCAB = new Set(Object.keys(SIGN_VIDEOS));

const CONTRACTIONS = {
  "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are", "he's": "he is", "she's": "she is", "it's": "it is",
  "don't": "do not", "doesn't": "does not", "didn't": "did not", "can't": "can not", "cannot": "can not", "won't": "will not",
  "isn't": "is not", "aren't": "are not", "wasn't": "was not", "i'll": "i will", "you'll": "you will", "i've": "i have", "let's": "let us",
  "what's": "what is", "where's": "where is", "how's": "how is", "that's": "that is",
};
const DROP = new Set(["a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being", "to", "of", "do", "does", "did", "so", "just", "um", "uh"]);
const PRONOUN = { i: "me", myself: "me", mine: "my", us: "we", him: "he", her: "she", them: "they", your: "your", yours: "your" };
const IRREGULAR = {
  went: "go", gone: "go", going: "go", ate: "eat", eaten: "eat", saw: "see", seen: "see", had: "have", has: "have", made: "make",
  got: "get", bought: "buy", came: "come", said: "say", told: "tell", thought: "think", took: "take", gave: "give", knew: "know",
  felt: "feel", found: "find", left: "leave", brought: "bring", taught: "teach", drank: "drink", slept: "sleep", wrote: "write",
  read: "read", ran: "run", sat: "sit", stood: "stand", met: "meet", paid: "pay", sent: "send", spoke: "speak", children: "child",
  men: "man", women: "woman", feet: "foot", teeth: "tooth", better: "better", best: "best", hi: "hello", hey: "hello", thanks: "thank you",
  mom: "mother", mum: "mother", mummy: "mother", dad: "father", papa: "father",
};
const TIME = new Set(["yesterday", "today", "tomorrow", "now", "later", "tonight", "morning", "night", "afternoon", "evening", "week", "month", "year", "last", "next", "before", "soon"]);
const WH = new Set(["what", "where", "when", "who", "why", "how", "which"]);

export function lemma(w) {
  if (VOCAB.has(w)) return w;
  if (IRREGULAR[w] && VOCAB.has(IRREGULAR[w])) return IRREGULAR[w];
  for (const [suf, rep] of [["ies", "y"], ["ing", ""], ["ing", "e"], ["ed", ""], ["ed", "e"], ["es", ""], ["s", ""], ["er", ""], ["ly", ""]]) {
    if (w.endsWith(suf) && w.length > suf.length + 2) {
      const base = w.slice(0, -suf.length) + rep;
      if (VOCAB.has(base)) return base;
      // doubled consonant: running → run, stopped → stop
      if (base.length > 3 && base.at(-1) === base.at(-2) && VOCAB.has(base.slice(0, -1))) return base.slice(0, -1);
    }
  }
  return null;
}

// → [{ word, video: bool }]; words without a video are fingerspelled by the player.
// Each sentence is reordered on its own (a question's WH-word ends that sentence, not the reply).
export function englishToGloss(text) {
  const sentences = String(text).match(/[^.?!]+[.?!]*/g) || [];
  return sentences.flatMap(sentenceToGloss);
}

function sentenceToGloss(text) {
  let s = ` ${String(text).toLowerCase().replace(/[’`]/g, "'")} `;
  for (const [k, v] of Object.entries(CONTRACTIONS)) s = s.replaceAll(` ${k} `, ` ${v} `);
  const words = s.replace(/[^a-z0-9' ]+/g, " ").split(/\s+/).filter(Boolean);
  const isQuestion = /\?\s*$/.test(String(text)) || WH.has(words[0]);
  const time = [], wh = [], body = [];
  // "thank you" is one sign
  for (let i = 0; i < words.length; i++) {
    let w = words[i];
    if (w === "thank" && words[i + 1] === "you") {
      body.push("thank you");
      i++;
      continue;
    }
    if (DROP.has(w)) continue;
    w = PRONOUN[w] || w;
    if (TIME.has(w)) time.push(w);
    else if (WH.has(w) && isQuestion) wh.push(w);
    else body.push(w);
  }
  return [...time, ...body, ...wh].map((w) => {
    const l = lemma(w);
    return l ? { word: l, video: true } : { word: w, video: false };
  });
}
