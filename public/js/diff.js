// Marks which words in the English translation the AI added or changed, i.e. words that don't
// come straight from a sign. Those get underlined, like in the original demo.

const EXPAND = {
  me: ["i", "me", "my", "mine", "myself", "im", "ive", "id", "ill"],
  i: ["i", "me", "my", "im", "ive", "id", "ill"],
  my: ["my", "mine", "i"],
  you: ["you", "your", "yours", "yourself", "youre", "youve", "youll", "youd"],
  your: ["your", "yours", "you"],
  he: ["he", "him", "his", "hes"],
  she: ["she", "her", "hers", "shes"],
  we: ["we", "us", "our", "were", "weve"],
  they: ["they", "them", "their", "theyre"],
  ix: ["it", "that", "this", "there", "he", "she", "they", "its", "thats"],
  not: ["not", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "cant", "wont", "no", "never"],
  finish: ["finish", "finished", "done", "already"],
  "thank-you": ["thank", "thanks", "you"],
  want: ["want", "wanna", "would", "like"],
  go: ["go", "going", "went", "gone"],
  eat: ["eat", "ate", "eating", "eaten"],
  see: ["see", "saw", "seen"],
  have: ["have", "has", "had", "got"],
  good: ["good", "well", "great"],
};

const stem = (w) => (w.length > 4 ? w.replace(/(ing|ed|es|s|ly)$/, "") : w);
const clean = (w) => w.toLowerCase().replace(/[^a-z]/g, "");

export function markChanges(english, glosses) {
  const vocab = new Set();
  for (const g of glosses) {
    const whole = g.toLowerCase();
    for (const w of EXPAND[whole] || []) vocab.add(w);
    for (const part of whole.split(/[-_\s]+/)) {
      if (!part) continue;
      vocab.add(part);
      for (const w of EXPAND[part] || []) vocab.add(w);
    }
  }
  const stems = new Set([...vocab].map(stem));
  return english.split(/(\s+)/).map((token) => {
    const w = clean(token);
    if (!w) return { text: token, changed: false };
    return { text: token, changed: !(vocab.has(w) || stems.has(stem(w))) };
  });
}
