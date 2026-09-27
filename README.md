# 🤟 SignBridge

**Real-time sign language → natural speech, with the emotion kept in. English · हिन्दी · Hinglish.**

SignBridge watches you sign through your webcam. It tracks **21 points on each hand** and **468 points on your face**. Hand tracking recognises the signs and builds them into a sentence. Face tracking picks up the part most translators miss: emotion, and ASL's *facial grammar*. Both go to an LLM, which writes a natural English sentence (words the AI added or changed are underlined). A voice then speaks it in a tone that matches your expression.

```
 webcam ──► MediaPipe Hands (2 × 21 pts) ──► k-NN sign classifier ──► gloss sentence ─┐
        └─► MediaPipe Face (468 pts + 52 blendshapes + head pose)                     ├─► LLM ──► English ──► voice
                 ├─ emotion: happy / sad / surprised / angry / neutral ───────────────┤   (Ollama / Claude)   (ElevenLabs /
                 └─ grammar: brow raise, furrow, head shake/nod, cheek puff, mouth ───┘                         browser)
```

## Why the face matters

ASL and English have different grammar, and in ASL the face carries grammar, not only feelings:

| Face / head | Meaning in ASL | Example |
|---|---|---|
| Raised eyebrows | yes/no question | `YOU HUNGRY` + brows up → *"Are you hungry?"* |
| Furrowed eyebrows | wh-question | `YOUR NAME WHAT` → *"What's your name?"* |
| Head shake | negation, even without a NOT sign | `ME LIKE` + shake → *"I don't like it."* |
| Head nod | affirmation / emphasis | *"I really do."* |
| Puffed cheeks | intensifier | *"a huge amount"* |
| "mm" / "th" mouth | easily / carelessly | *"drove carelessly"* |

Emotion changes word choice and delivery too. `FINISH WORK` with a smile becomes *"I'm finally done with work!"*, said brightly.

## Recognition models (no training needed)

| Model | What it does | Tested accuracy (unseen people) |
|---|---|---|
| **250 ASL words** | Isolated-sign recognizer from hand + face landmarks | **71.4% top-1 · 86.8% top-5**, and **90.4%** when it's confident |
| **Fingerspelling A–Z** | Letter classifier from 21 hand points | **96.4%** cross-validated · **72.7% top-1 / 84.5% top-3** on an independent photo set |
| **80 everyday objects** | MediaPipe EfficientDet-Lite2, with label voting + AI naming | (MediaPipe's published model) |

**Words.** The model is the 1st-place solution of Google's [Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs) Kaggle competition (Hoyeol Sohn, MIT licence, via [huggingface.co/sign](https://huggingface.co/sign/kaggle-asl-signs-1st-place)). It runs fully in the browser with [LiteRT.js](https://www.npmjs.com/package/@litertjs/core). The model has a dynamic input length that LiteRT.js can't resize, so `public/js/tflite-patch.js` writes each clip's frame count into the `.tflite` flatbuffer before compiling (~25 ms). It uses only lips, eyes, nose and both hands, so the app's Face + Hand landmarkers feed it directly.
I tested it on **374 clips from WLASL's test/val splits** (187 of its words; different signers and cameras from its training data) through the app's exact pipeline: 71.4% top-1, 83.5% top-3, 86.8% top-5. When its top guess is ≥50% (two-thirds of clips) it is right **90.4%** of the time, so the app only auto-adds a word above that bar. Tap a word chip to switch to the next guess. Re-run it at `/lab/eval-words.html`.

**Letters.** A small MLP trained for this project (`scripts/train-fingerspelling.mjs`, zero dependencies) on 6,749 hand-landmark samples from two sources: [asl-now-fingerspelling](https://huggingface.co/datasets/sid220/asl-now-fingerspelling), plus [an ASL alphabet photo set](https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03) run through MediaPipe (`/lab/extract-landmarks.html`), with rotated/flipped augmentation copies removed because direction matters in ASL (H vs U, P vs K). Features: wrist-relative coordinates, fingertip distances, joint bend angles, finger spread, palm direction. Results: 96.4% ± 0.5 in 5-fold cross-validation; ~76% when trained on one source and tested on the other; **72.7% top-1 / 84.5% top-3 on a third, independent photo set** (`/lab/eval-letters.html`). The hardest pairs are the ones people confuse too: N/M, Q/G, U/R, and H/U when the hand's direction is ambiguous.

## Features

**New in 0.2:** liquid-glass UI with a WebGL liquid-metal background and a 3D particle hand, 🔍 **object recognition**, **English / Hindi / Hinglish** output, and **expressive female voices**.

- **Live hand + face mesh** overlay, running fully in the browser on the GPU (MediaPipe Tasks).
- **No training needed**: sign 250 ASL words or fingerspell A–Z straight away (switch **Words / Letters / My signs** on the camera view).
- **Learn mode teaches you**: the alphabet with a drawn target hand, a live accuracy meter and finger-by-finger coaching ("Curl your index finger more"); 250 words with a "watch a signer" video link; progress is saved.
- **Train your own signs** in seconds, too. The classifier uses handshape, palm orientation, location relative to the face, and movement (four of the five ASL parameters). Non-manual markers, the fifth, go to the LLM.
- **Sentence building**: a sign is committed once it holds steady; drop your hands to finish the sentence and translate.
- **Emotion + facial-grammar detection**, with one-click neutral-face calibration.
- **LLM translation** with underlined AI edits, conversation context, and three backends:
  - **Groq** or **Google Gemini**: free cloud AI, set up inside the app in a minute
  - **Ollama**: local and private, if you have the disk space
  - **Claude**: paid, best quality
  - **Offline grammar**: English, Hindi and Hinglish with no AI, so the app always works
- **Emotion-matched voice**: ElevenLabs with per-tone voice settings, or the browser's voice as a free fallback.
- **🔍 Objects mode**: 80 everyday object types are detected live (MediaPipe EfficientDet), with label voting so boxes don't flicker. With an AI connected, steady objects get a precise ✨ name, and **What's in view?** names things outside those 80 classes. Tap any box to hear what it is, with 3 facts. Claude, or a vision model on Ollama, looks at the actual photo; the offline fallback uses Wikipedia.
- **English, हिन्दी, Hinglish**: translations and object descriptions in the language you pick. Reply captions listen in Indian English or Hindi.
- **Indian voices**: add a Sarvam AI key for natural Indian female voices (Priya, Neha, Kavya, Shreya, Ishita, Ritu) in Indian English, Hindi and Hinglish. Without a key, the browser's Indian-English female voice is preferred.
- **Expressive international voices**: pick Sarah, Aria, Jessica, Lily, Charlotte or Rachel. With `eleven_v3` the AI adds performance tags (`[excited]`, `[sighs]`, `[laughs softly]`) so the voice really acts. Without a key, the best female system voice is chosen automatically.
- **Pro UI**: glassmorphism, liquid buttons, animated aurora borders, word-by-word caption reveal, a live voice waveform, an emotion orb that morphs with your mood, 3D tilt cards. Respects reduced-motion settings.
- **Two-way conversation**: "Listen to reply" captions the hearing person's speech, so the Deaf user can read it.
- **Practice mode**: the app shows a word and checks your signing. Useful for learning, and for spotting weak signs.
- **Type signs instead**: test the translator by typing glosses.
- Transcript export, sign-set export/import (JSON), and keyboard shortcuts (`Enter` translate, `Backspace` undo, `Esc` clear, `R` record, `O` objects mode).

## Quick start

Requires **Node 20.12+** and Chrome/Edge (for webcam + speech recognition).

```bash
git clone https://github.com/Aayushsuthar/signbridge.git
cd signbridge
npm install
cp .env.example .env     # optional: add keys
npm start                # → http://localhost:5173
```

1. Click **Start camera** and allow access.
2. Open **Train signs**. Record a `_rest` pose (hands relaxed), then 5–10 signs, e.g. `ME`, `YOU`, `HUNGRY`, `GO`, `STORE`, `YESTERDAY`. Record each one 2–3 times.
3. Back on **Translate**, sign a sentence, then drop your hands. SignBridge translates and speaks it.
4. Optional: **Calibrate neutral face** so your resting expression isn't mistaken for a cue.

### Better translations (recommended, free)

**No install, free (recommended):** open ⚙ **Settings → AI brain** in the app and paste a key:
- **Groq**: sign up with email at [console.groq.com/keys](https://console.groq.com/keys). Fastest.
- **Google Gemini**: sign in with Google at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

The app checks the key, saves it to `.env` and switches over instantly, with no restart. Both handle Hindi and Hinglish and can see tapped objects.

**Works with no key at all:** the built-in offline grammar handles the starter vocabulary in all three languages, e.g. `STORE ME GO YESTERDAY` gives *I went to the store yesterday.* / *मैं कल दुकान गया।* / *Main kal dukaan gaya.*

**Local, private (Ollama, needs ~2 GB of disk):**
```bash
brew install ollama        # or download from ollama.com
ollama pull llama3.2
```
Keep Ollama running. SignBridge detects it automatically.

**Claude:** put `ANTHROPIC_API_KEY=...` in `.env` (default model `claude-opus-5`, set with `CLAUDE_MODEL`).

**Expressive voice:** put `ELEVENLABS_API_KEY=...` in `.env`. The default model `eleven_v3` performs emotion tags; if your plan doesn't include it, SignBridge falls back to `eleven_multilingual_v2` with emotion-tuned voice settings.

**Objects that the AI can see:** Claude sees the tapped object automatically. For Ollama, use a vision model: `ollama pull llama3.2-vision` and set `OLLAMA_MODEL=llama3.2-vision`.

Force a backend with `LLM=groq|gemini|ollama|claude|rules`.

## Project layout

```
server.mjs            HTTP server: static files, /api/translate, /api/describe, /api/speak, /api/status
lib/prompt.mjs        interpreter prompt, facial-grammar descriptions, JSON output schema
lib/rules.mjs         offline grammar: ASL gloss → English / Hindi / Hinglish
lib/wiki.mjs          Wikipedia summaries to ground object descriptions
public/models/        asl-signs.tflite (250 words), fingerspelling.json (A–Z, trained here)
public/js/asl-words.js   word model runtime + live segmentation (WordStream)
public/js/tflite-patch.js  sets the model's input length in the flatbuffer (LiteRT.js can't resize)
public/js/fingerspell.js   letter recognizer + Learn-mode coaching
public/lab/           reproducible evaluation + data-extraction pages
scripts/train-fingerspelling.mjs  trains and cross-validates the letter model
public/index.html     UI
public/js/bg-shader.js   WebGL liquid-metal background
public/js/hero3d.js   three.js particle hand for the landing screen
public/js/objects.js  object detection, smoothed tracking, crops for vision models
public/js/vision.js   MediaPipe setup, drawing, 138-dim hand feature vector
public/js/classifier.js  k-NN classifier, persistence, import/export
public/js/face.js     emotion + non-manual markers from blendshapes and head pose
public/js/speech.js   TTS playback and speech-to-text captions
public/js/diff.js     finds the words the AI changed (for underlining)
public/js/app.js      recognition loop, sentence building, training, practice
scripts/check.mjs     `npm run check`: tests for the camera-free logic
```

## Privacy

Video is processed in the browser and never uploaded. Only the recognised sign names and face-cue labels (e.g. `["ME","HUNGRY"]`, `brows-raised`, `happy`) go to the translator. When you tap an object, a small crop of that object is sent to the describing model (Claude, or Ollama locally). With Ollama, everything stays on your machine.

## Limitations & roadmap

- The word model knows 250 signs (a child-oriented everyday vocabulary). Signs outside it can be fingerspelled, or recorded under My signs.
- Real-world accuracy is lower than benchmark numbers, which is why every number above comes from people and cameras the models never saw. Good light and a plain background help most.
- Next steps: a larger word vocabulary (WLASL-2000 / ISL); a sequence model (LSTM/Transformer over landmark windows) trained on WLASL for dynamic signs; fingerspelling; ISL (Indian Sign Language) support; the reverse direction (English → sign via avatar).

## Credits

Inspired by [@cho_co_pie's prototype](https://www.instagram.com/p/DXJ0G8ADPc1/). Built with [MediaPipe](https://ai.google.dev/edge/mediapipe), [Ollama](https://ollama.com), [Claude](https://www.anthropic.com), and [ElevenLabs](https://elevenlabs.io).

MIT License.
