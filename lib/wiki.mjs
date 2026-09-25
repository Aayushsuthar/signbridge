// Free, keyless grounding for object descriptions: the Wikipedia page summary for a detector label.

// COCO labels whose Wikipedia article has a different title
const TITLES = {
  person: "Human",
  "cell phone": "Mobile_phone",
  tv: "Television",
  "potted plant": "Houseplant",
  "dining table": "Table_(furniture)",
  "sports ball": "Ball",
  remote: "Remote_control",
  mouse: "Computer_mouse",
  keyboard: "Computer_keyboard",
  "hair drier": "Hair_dryer",
  "tennis racket": "Racket_(sports_equipment)",
  "wine glass": "Wine_glass",
  "hot dog": "Hot_dog",
  "teddy bear": "Teddy_bear",
};

const cache = new Map();

export async function wikiSummary(label) {
  const key = String(label).toLowerCase().trim();
  if (cache.has(key)) return cache.get(key);
  const title = TITLES[key] || key.replace(/\s+/g, "_").replace(/^./, (c) => c.toUpperCase());
  let result = null;
  try {
    const res = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`, {
      headers: { "user-agent": "SignBridge/0.2 (https://github.com/Aayushsuthar/signbridge)" },
      signal: AbortSignal.timeout(4000),
    });
    if (res.ok) {
      const data = await res.json();
      if (data.type !== "disambiguation" && data.extract) {
        result = { title: data.title, extract: data.extract, url: data.content_urls?.desktop?.page, image: data.thumbnail?.source };
      }
    }
  } catch {
    // offline or slow: describe without grounding
  }
  cache.set(key, result);
  return result;
}
