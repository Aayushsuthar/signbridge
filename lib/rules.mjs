// Offline translator: a small ASL-gloss grammar that renders English, Hindi and Hinglish without any AI.
// It covers the starter vocabulary well (subject/verb/object, tense from time signs, questions, negation,
// possessives, "want to", Hindi ergative and dative forms). The LLM backends handle everything else.

// t: pron | poss | person | noun | place | time | wh | neg | adj | verb | phrase
// en/hi/hg: English / Hindi / Hinglish surface forms
const LEX = {
  ME: { t: "pron", p: 1, en: "I", eo: "me", hi: "मैं", hg: "main" },
  I: { t: "pron", p: 1, en: "I", eo: "me", hi: "मैं", hg: "main" },
  YOU: { t: "pron", p: 2, en: "you", eo: "you", hi: "तुम", hg: "tum" },
  HE: { t: "pron", p: 3, en: "he", eo: "him", hi: "वह", hg: "woh" },
  SHE: { t: "pron", p: 3, en: "she", eo: "her", hi: "वह", hg: "woh" },
  IX: { t: "pron", p: 3, en: "that", eo: "that", hi: "वह", hg: "woh" },
  WE: { t: "pron", p: 4, en: "we", eo: "us", hi: "हम", hg: "hum" },
  THEY: { t: "pron", p: 4, en: "they", eo: "them", hi: "वे", hg: "woh log" },

  MY: { t: "poss", en: "my", hi: "मेरा", hg: "mera" },
  YOUR: { t: "poss", en: "your", hi: "तुम्हारा", hg: "tumhara" },
  OUR: { t: "poss", en: "our", hi: "हमारा", hg: "hamara" },
  HIS: { t: "poss", en: "his", hi: "उसका", hg: "uska" },
  HER: { t: "poss", en: "her", hi: "उसका", hg: "uska" },

  FRIEND: { t: "person", en: "friend", hi: "दोस्त", hg: "dost" },
  MOTHER: { t: "person", en: "mother", hi: "माँ", hg: "maa" },
  FATHER: { t: "person", en: "father", hi: "पापा", hg: "papa" },
  TEACHER: { t: "person", en: "teacher", hi: "टीचर", hg: "teacher" },

  STORE: { t: "place", en: "the store", hi: "दुकान", hg: "dukaan" },
  HOME: { t: "place", en: "home", to: "home", hi: "घर", hg: "ghar" },
  SCHOOL: { t: "place", en: "school", hi: "स्कूल", hg: "school" },
  WORK: { t: "place", en: "work", hi: "काम", hig: "काम पर", hg: "kaam", hgg: "kaam pe" },
  HOSPITAL: { t: "place", en: "the hospital", hi: "अस्पताल", hg: "hospital" },
  FOOD: { t: "noun", en: "food", hi: "खाना", hg: "khana" },
  WATER: { t: "noun", en: "water", hi: "पानी", hg: "paani" },
  NAME: { t: "noun", en: "name", hi: "नाम", hg: "naam" },
  BOOK: { t: "noun", en: "the book", hi: "किताब", hg: "kitaab" },
  PHONE: { t: "noun", en: "the phone", hi: "फ़ोन", hg: "phone" },

  YESTERDAY: { t: "time", tense: "past", en: "yesterday", hi: "कल", hg: "kal" },
  TODAY: { t: "time", en: "today", hi: "आज", hg: "aaj" },
  TOMORROW: { t: "time", tense: "future", en: "tomorrow", hi: "कल", hg: "kal" },
  NOW: { t: "time", en: "now", hi: "अभी", hg: "abhi" },
  LATER: { t: "time", tense: "future", en: "later", hi: "बाद में", hg: "baad mein" },
  MORNING: { t: "time", en: "in the morning", hi: "सुबह", hg: "subah" },
  TONIGHT: { t: "time", tense: "future", en: "tonight", hi: "आज रात", hg: "aaj raat" },
  BEFORE: { t: "time", tense: "past", en: "before", hi: "पहले", hg: "pehle" },

  WHAT: { t: "wh", en: "what", hi: "क्या", hg: "kya" },
  WHERE: { t: "wh", en: "where", hi: "कहाँ", hg: "kahan" },
  WHEN: { t: "wh", en: "when", hi: "कब", hg: "kab" },
  WHY: { t: "wh", en: "why", hi: "क्यों", hg: "kyun" },
  HOW: { t: "wh", en: "how", hi: "कैसे", hg: "kaise" },
  WHO: { t: "wh", en: "who", hi: "कौन", hg: "kaun" },

  NOT: { t: "neg" },

  HAPPY: { t: "adj", en: "happy", hi: "खुश", hg: "khush" },
  SAD: { t: "adj", en: "sad", hi: "उदास", hg: "udaas" },
  TIRED: { t: "adj", en: "tired", hi: "थका हुआ", hg: "thaka hua" },
  HUNGRY: { t: "adj", en: "hungry", hi: "भूखा", hg: "bhookha" },
  SICK: { t: "adj", en: "sick", hi: "बीमार", hg: "beemar" },
  GOOD: { t: "adj", en: "good", hi: "अच्छा", hg: "accha" },
  BAD: { t: "adj", en: "bad", hi: "बुरा", hg: "bura" },
  FINE: { t: "adj", en: "fine", hi: "ठीक", hg: "theek" },
  READY: { t: "adj", en: "ready", hi: "तैयार", hg: "taiyaar" },
  BUSY: { t: "adj", en: "busy", hi: "व्यस्त", hg: "busy" },
  EXCITED: { t: "adj", en: "excited", hi: "उत्साहित", hg: "excited" },

  // en: [base, past, 3rd-person]; hi/hg: stem, past (masc sg / pl); tr: transitive (Hindi ergative in past)
  // obj: how Hindi marks a pronoun object: ko (मुझे), se (मुझसे), of (मेरी)
  GO: { t: "verb", en: ["go", "went", "goes"], hi: "जा", hiP: ["गया", "गए"], hg: "ja", hgP: ["gaya", "gaye"] },
  EAT: { t: "verb", tr: 1, en: ["eat", "ate", "eats"], hi: "खा", hiP: ["खाया"], hg: "kha", hgP: ["khaya"] },
  DRINK: { t: "verb", tr: 1, en: ["drink", "drank", "drinks"], hi: "पी", hiP: ["पिया"], hg: "pi", hgP: ["piya"] },
  WANT: { t: "verb", en: ["want", "wanted", "wants"], hi: "चाह", hiP: ["चाहता", "चाहते"], hg: "chaah", hgP: ["chahta", "chahte"] },
  LIKE: { t: "verb", tr: 1, obj: "ko", en: ["like", "liked", "likes"], hi: "पसंद कर", hiP: ["पसंद किया"], hg: "pasand kar", hgP: ["pasand kiya"] },
  LOVE: { t: "verb", tr: 1, obj: "se", en: ["love", "loved", "loves"], hi: "प्यार कर", hiP: ["प्यार किया"], hg: "pyaar kar", hgP: ["pyaar kiya"] },
  HELP: { t: "verb", tr: 1, obj: "of", en: ["help", "helped", "helps"], hi: "मदद कर", hiP: ["मदद की"], hg: "madad kar", hgP: ["madad ki"] },
  FINISH: { t: "verb", tr: 1, en: ["finish", "finished", "finishes"], hi: "खत्म कर", hiP: ["खत्म किया"], hg: "khatam kar", hgP: ["khatam kiya"] },
  SEE: { t: "verb", tr: 1, obj: "ko", en: ["see", "saw", "sees"], hi: "देख", hiP: ["देखा"], hg: "dekh", hgP: ["dekha"] },
  COME: { t: "verb", en: ["come", "came", "comes"], hi: "आ", hiP: ["आया", "आए"], hg: "aa", hgP: ["aaya", "aaye"] },
  SLEEP: { t: "verb", en: ["sleep", "slept", "sleeps"], hi: "सो", hiP: ["सोया", "सोए"], hg: "so", hgP: ["soya", "soye"] },

  HELLO: { t: "phrase", en: "Hello!", hi: "नमस्ते!", hg: "Namaste!" },
  HI: { t: "phrase", en: "Hi!", hi: "नमस्ते!", hg: "Hi!" },
  "THANK-YOU": { t: "phrase", en: "Thank you!", hi: "धन्यवाद!", hg: "Thank you!" },
  THANKYOU: { t: "phrase", en: "Thank you!", hi: "धन्यवाद!", hg: "Thank you!" },
  PLEASE: { t: "phrase", en: "Please.", hi: "कृपया।", hg: "Please." },
  SORRY: { t: "phrase", en: "Sorry.", hi: "माफ़ करना।", hg: "Sorry yaar." },
  YES: { t: "phrase", en: "Yes.", hi: "हाँ।", hg: "Haan." },
  NO: { t: "phrase", en: "No.", hi: "नहीं।", hg: "Nahi." },
  "I-LOVE-YOU": { t: "phrase", en: "I love you!", hi: "मैं तुमसे प्यार करता हूँ!", hg: "Main tumse pyaar karta hoon!" },
  ILY: { t: "phrase", en: "I love you!", hi: "मैं तुमसे प्यार करता हूँ!", hg: "Main tumse pyaar karta hoon!" },
  "NICE-MEET-YOU": { t: "phrase", en: "Nice to meet you!", hi: "आपसे मिलकर खुशी हुई!", hg: "Aapse milkar khushi hui!" },
  "SEE-YOU-LATER": { t: "phrase", en: "See you later!", hi: "फिर मिलते हैं!", hg: "Phir milte hain!" },
  "GOOD-MORNING": { t: "phrase", en: "Good morning!", hi: "सुप्रभात!", hg: "Good morning!" },
  "HOW-YOU": { t: "phrase", en: "How are you?", hi: "तुम कैसे हो?", hg: "Tum kaise ho?" },
};

// Hindi/Hinglish agreement: person 1 = मैं, 2 = तुम, 3 = वह / a noun, 4 = हम / वे
const COP = {
  hi: { present: ["", "हूँ", "हो", "है", "हैं"], past: ["", "था", "थे", "था", "थे"], future: ["", "रहूँगा", "रहोगे", "रहेगा", "रहेंगे"] },
  hg: { present: ["", "hoon", "ho", "hai", "hain"], past: ["", "tha", "the", "tha", "the"], future: ["", "rahunga", "rahoge", "rahega", "rahenge"] },
};
const HABIT = { hi: ["", "ता हूँ", "ते हो", "ता है", "ते हैं"], hg: ["", "ta hoon", "te ho", "ta hai", "te hain"] };
const PROG = { hi: ["", " रहा हूँ", " रहे हो", " रहा है", " रहे हैं"], hg: ["", " raha hoon", " rahe ho", " raha hai", " rahe hain"] };
const HABIT_NEG = { hi: ["", "ता", "ते", "ता", "ते"], hg: ["", "ta", "te", "ta", "te"] };
const FUT_V = { hi: ["", "ऊँगा", "ओगे", "एगा", "एँगे"], hg: ["", "unga", "oge", "ega", "enge"] }; // after a vowel
const FUT_C = { hi: ["", "ूँगा", "ोगे", "ेगा", "ेंगे"], hg: ["", "unga", "oge", "ega", "enge"] }; // after a consonant
const ERG = {
  hi: { 1: "मैंने", 2: "तुमने", 3: "उसने", 4: "हमने" },
  hg: { 1: "maine", 2: "tumne", 3: "usne", 4: "humne" },
};
const DAT = {
  hi: { 1: "मुझे", 2: "तुम्हें", 3: "उसे", 4: "हमें" },
  hg: { 1: "mujhe", 2: "tumhe", 3: "use", 4: "humein" },
};
const OBJ = {
  ko: DAT,
  se: { hi: { 1: "मुझसे", 2: "तुमसे", 3: "उससे", 4: "हमसे" }, hg: { 1: "mujhse", 2: "tumse", 3: "usse", 4: "humse" } },
  of: { hi: { 1: "मेरी", 2: "तुम्हारी", 3: "उसकी", 4: "हमारी" }, hg: { 1: "meri", 2: "tumhari", 3: "uski", 4: "hamari" } },
};
const THEY_FIX = { hi: { "हमने": "उन्होंने", "हमें": "उन्हें" }, hg: { humne: "unhone", humein: "unhe" } };
const MATRA = /[ािीुूेैोौ]$/;

const TONE_BY_EMOTION = { happy: "happy", sad: "sad", angry: "angry", surprised: "surprised", neutral: "neutral" };

const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

function parse(glosses, markers) {
  const words = glosses.map((g) => ({ g, ...(LEX[g] || { t: "noun", en: g.toLowerCase().replace(/[-_]+/g, " "), hi: g.toLowerCase(), hg: g.toLowerCase() }) }));
  const verbs = words.filter((w) => w.t === "verb");
  const firstVerb = words.findIndex((w) => w.t === "verb");
  const s = {
    verbs,
    adj: words.find((w) => w.t === "adj"),
    wh: words.find((w) => w.t === "wh"),
    times: words.filter((w) => w.t === "time"),
    neg: words.some((w) => w.t === "neg") || markers.has("head-shake"),
    intense: markers.has("cheek-puff"),
    subj: null,
    obj: null,
    place: null,
    noun: null,
    poss: null,
  };
  // possessive + noun ("YOUR NAME")
  words.forEach((w, i) => {
    if (w.t === "poss" && words[i + 1] && ["noun", "person", "place"].includes(words[i + 1].t)) {
      words[i + 1].possBy = w;
      w.used = true;
    }
  });
  // subject: a pronoun or person before the verb (topic-comment), else the first pronoun anywhere
  const before = firstVerb < 0 ? words : words.slice(0, firstVerb);
  s.subj = before.find((w) => w.t === "pron" || w.t === "person" || (w.t === "noun" && w.possBy && !s.verbs.length)) || words.find((w) => w.t === "pron");
  // a pronoun after the verb that isn't the subject is the object ("FRIEND HELP ME")
  if (firstVerb >= 0) s.obj = words.slice(firstVerb + 1).find((w) => (w.t === "pron" || w.t === "person") && w !== s.subj) || null;
  s.place = words.find((w) => w.t === "place" && w !== s.subj);
  s.noun = words.find((w) => w.t === "noun" && w !== s.subj);
  s.poss = s.subj?.possBy ? s.subj : null;

  const hasPast = s.times.some((t) => t.tense === "past") || (verbs.some((v) => v.g === "FINISH") && verbs.length === 1);
  const hasFuture = s.times.some((t) => t.tense === "future");
  // FINISH next to another verb is a completed-action marker, not the main verb
  if (verbs.length > 1 && verbs[0].g === "FINISH") {
    s.verbs = verbs.slice(1);
    s.tense = "past";
  } else s.tense = hasPast ? "past" : hasFuture ? "future" : "present";
  if (s.tense === "future" && s.verbs.length > 1 && s.verbs[0].g === "WANT") s.tense = "present";
  s.question = Boolean(s.wh) || markers.has("brows-raised");
  // ASL often drops the subject: statements default to I, yes/no questions to you
  if (!s.subj && (s.verbs.length || s.adj)) s.subj = s.question && !s.wh ? LEX.YOU : LEX.ME;
  return s;
}

const personOf = (w) => (w?.t === "pron" ? w.p : 3);

// ---------------- English ----------------

function nounEn(w) {
  if (!w) return "";
  if (w.possBy) return `${w.possBy.en} ${w.en.replace(/^the /, "")}`;
  if (w.t === "person") return `a ${w.en}`;
  return w.en;
}
function subjEn(w) {
  return w.t === "pron" ? w.en : nounEn(w);
}

function renderEn(s) {
  const p = personOf(s.subj);
  const be = { present: p === 1 ? "am" : p === 3 ? "is" : "are", past: p === 1 || p === 3 ? "was" : "were", future: "will be" }[s.tense];
  const subj = s.subj ? subjEn(s.subj) : "";
  const time = s.times.map((t) => t.en).join(" ");
  let out;

  if (s.verbs.length) {
    let [v, v2] = s.verbs;
    let comp = "";
    if (v2 && ["WANT", "LIKE", "LOVE"].includes(v.g)) comp = ` to ${v2.en[0]}`;
    else v2 = null;
    const target = v2 || v;
    let rest = "";
    if (target.g === "GO" || target.g === "COME") rest = s.place ? ` ${s.place.to || (s.place.en.startsWith("the") ? `to ${s.place.en}` : `to ${s.place.en}`)}` : "";
    else if (s.obj) rest = ` ${s.obj.t === "pron" ? s.obj.eo : nounEn(s.obj)}`;
    else if (s.noun || s.place) rest = ` ${nounEn(s.noun || s.place)}`;
    if (s.intense) rest += " a lot";
    const third = p === 3;
    const aux = { present: third ? "does" : "do", past: "did", future: "will" }[s.tense];
    if (s.question) {
      const neg = s.neg ? (s.tense === "future" ? "won't" : `${aux}n't`) : aux;
      if (s.wh && s.wh.g === "WHO" && !s.subj?.t) out = `who ${v.en[s.tense === "past" ? 1 : 2]}${comp}${rest}`;
      else if (s.wh && v.g === "GO" && s.tense === "present" && !comp) out = `${s.wh.en} ${p === 1 ? "am" : p === 3 ? "is" : "are"} ${subj} going${rest}`;
      else out = `${s.wh ? `${s.wh.en} ` : ""}${neg} ${subj} ${v.en[0]}${comp}${rest}`;
    } else {
      let verb;
      if (s.neg) verb = `${{ present: third ? "doesn't" : "don't", past: "didn't", future: "won't" }[s.tense]} ${v.en[0]}`;
      else verb = s.tense === "past" ? v.en[1] : s.tense === "future" ? `will ${v.en[0]}` : third ? v.en[2] : v.en[0];
      out = `${subj} ${verb}${comp}${rest}`;
    }
  } else if (s.adj) {
    const adj = `${s.intense ? "really " : ""}${s.adj.en}`;
    if (s.question) {
      const [b0, ...bRest] = be.split(" ");
      out = `${s.wh ? `${s.wh.en} ` : ""}${b0}${s.neg && s.tense !== "future" ? "n't" : ""} ${subj}${bRest.length ? ` ${bRest.join(" ")}` : ""}${s.neg && s.tense === "future" ? " not" : ""} ${adj}`;
    } else out = `${subj} ${be}${s.neg ? " not" : ""} ${adj}`;
  } else if (s.wh) {
    // "YOUR NAME WHAT" → What is your name?
    const thing = s.subj ? subjEn(s.subj) : s.noun ? nounEn(s.noun) : "that";
    out = `${s.wh.en} ${s.tense === "past" ? "was" : "is"} ${thing}`;
  } else if (s.subj?.possBy && s.subj.g === "NAME" && s.noun) {
    out = `${nounEn(s.subj)} is ${cap(s.noun.en)}`;
  } else {
    out = [s.subj && subjEn(s.subj), s.noun && nounEn(s.noun), s.place && s.place.en].filter(Boolean).join(" ");
  }
  if (time) out = `${out} ${time}`;
  out = out.replace(/\s+/g, " ").trim().replace(/\bi\b/g, "I");
  return cap(out);
}

// ---------------- Hindi / Hinglish ----------------

function nounHi(w, L) {
  if (!w) return "";
  const base = w[L];
  return w.possBy ? `${w.possBy[L]} ${base}` : base;
}

function renderHi(s, L) {
  const p = personOf(s.subj);
  const they = s.subj?.g === "THEY";
  const fix = (x) => (they && THEY_FIX[L][x]) || x;
  const subj = s.subj ? (s.subj.t === "pron" ? s.subj[L] : nounHi(s.subj, L)) : "";
  const time = s.times.map((t) => t[L]).join(" ");
  const neg = s.neg ? (L === "hi" ? "नहीं" : "nahi") : "";
  const very = s.intense ? (L === "hi" ? "बहुत" : "bahut") : "";
  const wh = s.wh ? s.wh[L] : "";
  const parts = [];

  if (s.verbs.length) {
    let [v, v2] = s.verbs;
    if (!(v2 && v.g === "WANT")) v2 = null;
    const target = v2 || v;
    let obj = "";
    if (target.g === "GO" || target.g === "COME") obj = s.place ? (L === "hi" ? s.place.hig || s.place.hi : s.place.hgg || s.place.hg) : "";
    else if (s.obj) obj = s.obj.t === "pron" ? fix(OBJ[target.obj || "ko"][L][s.obj.p]) : nounHi(s.obj, L);
    else if (s.noun || s.place) obj = nounHi(s.noun || s.place, L);

    if (v.g === "WANT" && !v2) {
      // WANT + thing: मुझे पानी चाहिए
      const dat = s.subj?.t === "pron" ? fix(DAT[L][p]) : `${subj} ${L === "hi" ? "को" : "ko"}`;
      parts.push(dat, time, obj, very, neg, L === "hi" ? "चाहिए" : "chahiye", s.tense === "past" ? (L === "hi" ? "था" : "tha") : "");
    } else {
      const stem = target[L];
      const inf = v2 ? `${stem}${L === "hi" ? "ना" : "na"}` : "";
      const verbStem = v2 ? v[L] : stem; // with WANT + verb, WANT is the conjugated verb
      const main = v2 ? v : target;
      let verb;
      let s0 = subj;
      if (s.tense === "past") {
        if (main.tr && !v2) {
          s0 = s.subj?.t === "pron" ? fix(ERG[L][p]) : `${subj}${L === "hi" ? " ने" : " ne"}`;
          verb = main[`${L}P`][0];
        } else if (v2) verb = `${main[`${L}P`][p === 2 || p === 4 ? 1 : 0]} ${COP[L].past[p]}`;
        else verb = main[`${L}P`][p === 2 || p === 4 ? 1 : 0] || main[`${L}P`][0];
      } else if (s.tense === "future") {
        const ends = MATRA.test(verbStem) || (L === "hg" && /[aeiou]$/.test(verbStem)) ? FUT_V : FUT_C;
        verb = `${verbStem}${ends[L][p]}`;
      } else if (s.wh && (main.g === "GO" || main.g === "COME") && !s.neg) {
        verb = `${verbStem}${PROG[L][p]}`; // "कहाँ जा रहे हो?"
      } else {
        verb = s.neg ? `${verbStem}${HABIT_NEG[L][p]}` : `${verbStem}${HABIT[L][p]}`;
      }
      parts.push(s0, time, inf ? "" : obj, inf ? `${obj} ${inf}` : "", very, s.wh ? wh : "", neg, verb);
    }
  } else if (s.adj) {
    parts.push(subj, time, very, s.adj[L], neg, COP[L][s.tense][p]);
  } else if (s.wh) {
    const thing = s.subj ? subj : s.noun ? nounHi(s.noun, L) : L === "hi" ? "यह" : "yeh";
    parts.push(thing, wh, s.tense === "past" ? COP[L].past[3] : COP[L].present[3]);
  } else if (s.subj?.possBy && s.subj.g === "NAME" && s.noun) {
    parts.push(subj, cap(s.noun.en), COP[L].present[3]);
  } else {
    parts.push(subj, time, s.noun && nounHi(s.noun, L), s.place && s.place[L]);
  }

  let out = parts.filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
  if (s.question && !s.wh) out = `${L === "hi" ? "क्या" : "kya"} ${out}`;
  return L === "hg" ? cap(out) : out;
}

// ---------------- entry point ----------------

export function translateWithRules({ signs = [], face = {}, lang = "en" }) {
  const glosses = signs.map((s) => String(s.gloss).toUpperCase()).filter((g) => g && !g.startsWith("_"));
  const markers = new Set([...(face.markers || []), ...signs.flatMap((s) => s.markers || [])]);
  const L = lang === "hi" ? "hi" : lang === "hinglish" ? "hg" : "en";

  // set phrases (HELLO, THANK-YOU…) become their own short sentences, in signing order
  const chunks = [];
  let clause = [];
  for (const g of glosses) {
    if (LEX[g]?.t === "phrase" && !(g === "NO" && glosses.length > 1 && clause.length)) {
      if (clause.length) chunks.push({ clause });
      chunks.push({ phrase: LEX[g] });
      clause = [];
    } else clause.push(g === "NO" ? "NOT" : g);
  }
  if (clause.length) chunks.push({ clause });

  const emotion = face.emotion || "neutral";
  const exclaim = ["happy", "surprised", "angry"].includes(emotion) && (face.intensity ?? 0) > 0.45;
  let anyQuestion = false;
  const sentences = chunks.map((c) => {
    if (c.phrase) return c.phrase[L];
    const s = parse(c.clause, markers);
    anyQuestion ||= s.question;
    const body = L === "en" ? renderEn(s) : renderHi(s, L);
    if (!body) return "";
    return body + (s.question ? "?" : exclaim ? "!" : L === "hi" ? "।" : ".");
  });
  const text = sentences.filter(Boolean).join(" ") || "…";
  const tone = anyQuestion && emotion === "neutral" ? "questioning" : TONE_BY_EMOTION[emotion] || "neutral";
  return {
    text,
    speech: text,
    tone,
    notes: "Offline grammar (no AI). Add a free Groq or Gemini key in ⚙ Settings for fully natural sentences.",
  };
}
