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

## Features

**New in 0.2:** liquid-glass UI with a WebGL liquid-metal background and a 3D particle hand, 🔍 **object recognition**, **English / Hindi / Hinglish** output, and **expressive female voices**.

- **Live hand + face mesh** overlay, running fully in the browser on the GPU (MediaPipe Tasks).
- **Train your own signs** in seconds. The classifier uses handshape, palm orientation, location relative to the face, and movement (four of the five ASL parameters). Non-manual markers, the fifth, go to the LLM.
- **Sentence building**: a sign is committed once it holds steady; drop your hands to finish the sentence and translate.
- **Emotion + facial-grammar detection**, with one-click neutral-face calibration.
- **LLM translation** with underlined AI edits, conversation context, and three backends:
  - **Ollama** (local, private, free), the default when running
  - **Claude**, when `ANTHROPIC_API_KEY` is set
  - **Offline rules**, so the app always works
- **Emotion-matched voice**: ElevenLabs with per-tone voice settings, or the browser's voice as a free fallback.
- **🔍 Objects mode**: 80 everyday object types are detected live (MediaPipe EfficientDet). Tap any box to hear what it is, with 3 facts. Claude, or a vision model on Ollama, looks at the actual photo; the offline fallback uses Wikipedia.
- **English, हिन्दी, Hinglish**: translations and object descriptions in the language you pick. Reply captions listen in Indian English or Hindi.
- **Expressive female voices**: pick Sarah, Aria, Jessica, Lily, Charlotte or Rachel. With `eleven_v3` the AI adds performance tags (`[excited]`, `[sighs]`, `[laughs softly]`) so the voice really acts. Without a key, the best female system voice is chosen automatically.
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

### Better translations (recommended)

**Local, private (Ollama):**
```bash
brew install ollama        # or download from ollama.com
ollama pull llama3.2
```
Keep Ollama running. SignBridge detects it automatically.

**Claude:** put `ANTHROPIC_API_KEY=...` in `.env` (default model `claude-opus-5`, set with `CLAUDE_MODEL`).

**Expressive voice:** put `ELEVENLABS_API_KEY=...` in `.env`. The default model `eleven_v3` performs emotion tags; if your plan doesn't include it, SignBridge falls back to `eleven_multilingual_v2` with emotion-tuned voice settings.

**Objects that the AI can see:** Claude sees the tapped object automatically. For Ollama, use a vision model: `ollama pull llama3.2-vision` and set `OLLAMA_MODEL=llama3.2-vision`.

Force a backend with `LLM=ollama|claude|rules`.

## Project layout

```
server.mjs            HTTP server: static files, /api/translate, /api/describe, /api/speak, /api/status
lib/prompt.mjs        interpreter prompt, facial-grammar descriptions, JSON output schema
lib/rules.mjs         offline fallback translator
lib/wiki.mjs          Wikipedia summaries to ground object descriptions
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

- Signs are learned from your own recordings (k-NN), so there is no universal vocabulary out of the box. Signs with long, complex motion are harder than static or short-motion ones.
- Next steps: a sequence model (LSTM/Transformer over landmark windows) trained on WLASL for dynamic signs; fingerspelling; ISL (Indian Sign Language) support; the reverse direction (English → sign via avatar).

## Credits

Inspired by [@cho_co_pie's prototype](https://www.instagram.com/p/DXJ0G8ADPc1/). Built with [MediaPipe](https://ai.google.dev/edge/mediapipe), [Ollama](https://ollama.com), [Claude](https://www.anthropic.com), and [ElevenLabs](https://elevenlabs.io).

MIT License.
