# VocalBlade

An AI-powered, text-based video editor: edit a video by editing its transcript, and let AI make the cuts sound and look seamless.

## About

VocalBlade is a final-year Computer Engineering capstone project by [Mohammad Salhab](https://github.com/MohammadSalhabEngineer) at Beirut Arab University (graded A+). It integrates speech transcription, a local LLM, and a suite of audio/video ML models into a single Descript-style editing workflow.

## How it works

1. **Upload** a video or audio file.
2. **Transcribe** — WhisperX generates a word-level, speaker-diarized transcript.
3. **Edit the transcript, edit the video** — delete, move, or copy-paste words and phrases in the transcript, and the underlying video is cut and reordered to match. Or describe the edit in plain language (e.g. *"remove anything related to politics"*, *"cut what this person said"*, *"remove filler words like umm, ooo"*) and a local LLM turns it into precise word-level edits.
4. **Preview live** — a virtual timeline skips deleted words during playback before you export anything.
5. **Export** — choose an audio continuity method and a video continuity method, and VocalBlade renders a seamlessly re-cut video.

## Features

### Text-based editing
Click/select/delete/reorder words directly in the transcript — the video follows. Undo/redo, copy/paste, multi-select, and full keyboard shortcuts (Del, Ctrl+Z/Shift+Z, Ctrl+C/V, Ctrl+A, Esc).

### AI-assisted editing
A dedicated local LLM (**Qwen 2.5 7B**, 4-bit quantized, runs on-GPU) turns natural-language instructions into validated word-level edit operations (delete/move/delete-range), checked against the real transcript so it can't hallucinate edits that don't exist. Literal requests ("delete the word coffee") are fast-pathed with regex; semantic/topical requests ("cut anything about coffee", filler-word removal, removing a whole speaker's lines) are routed to the LLM. A conversational AI chat assistant is also available for Q&A over the transcript.

### Audio continuity engine
Deleting words creates hard cuts — this engine makes them inaudible. Cut boundaries are first snapped to silence/zero-crossing points, then one of the following is applied:

| Method | Description |
|---|---|
| Crossfade | Fast time-domain equal-power crossfade |
| Smart Silence | Silence-aware cuts (recommended default) |
| Spectral | FFT-based phase-coherent crossfade |
| AI Inpainting | Neural mel-spectrogram audio inpainting |
| WSOLA | Pitch-aware speech blending |
| Room Tone | Room-tone replacement (recommended) |
| AI Regeneration | AI voice regeneration via F5-TTS |
| EnCodec | Meta EnCodec neural-codec latent blending |
| Neural Smooth | Combined room tone + face morph ("Smart Edit") |
| Smart Cut | FOMM video bridge + speech-crossfade audio bridge |
| SadTalker Cut | SadTalker audio-driven face + F5-TTS voice-clone bridge |
| Demucs | Vocal/background source separation + per-stem continuity, so music isn't glitched |
| Boundary Clone | F5-TTS clones boundary words, crossfades within the word |
| None | Raw concatenation, no processing |

### Visual continuity engine
Smooths the visual jump at a cut:

| Method | Description |
|---|---|
| None | Hard cut, no processing (default) |
| Blend | ffmpeg motion-compensated interpolation |
| RIFE | Neural optical-flow frame interpolation (vendored local weights) |
| Frame Pad | Freeze boundary frames + dissolve transition |

These compose with the audio-side Neural Smooth, Smart Cut, and SadTalker Cut methods above for full audio+visual bridging — including generating brand-new talking-head footage of the speaker (SadTalker) to cover a cut.

### Transcription foundation
Built on WhisperX: all Whisper model sizes, 98 supported languages, word-level timestamp alignment, and speaker diarization via `pyannote`.

## Architecture

```
vocal_blade_backend/   Django + Strawberry GraphQL backend (Python, managed by uv)
frontend-mvp/           React + TypeScript + Vite + Tailwind frontend
docs/                    API reference and setup guide
```

Single GraphQL endpoint (`/graphql/`). See [docs/SETUP.md](docs/SETUP.md) for a detailed setup walkthrough. [docs/API.md](docs/API.md) currently documents only the transcription mutations (`transcribeAudio`, `batchTranscribeAudio`) — it predates the editing features and doesn't yet cover `exportEditedMedia`, `semanticEdit`, or `aiChat`.

## Prerequisites

- Python 3.12, managed via [uv](https://github.com/astral-sh/uv)
- Node.js 20+
- ffmpeg
- An NVIDIA GPU is strongly recommended — transcription, the local LLM, and the ML-based continuity methods (RIFE, FOMM, SadTalker, F5-TTS, Demucs) all benefit from or require it. If you enable speaker diarization, `onnxruntime`/`pyannote` require CUDA 11 libraries alongside PyTorch's CUDA 12 — see [docs/SETUP.md](docs/SETUP.md) for how the CUDA 11/12 coexistence is handled.

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/MohammadSalhabEngineer/VocalBlade.git
cd VocalBlade

cd vocal_blade_backend
uv sync

cd ../frontend-mvp
npm install
```

### 2. Configure environment variables

```bash
cd vocal_blade_backend
cp .env.example .env
```

Fill in `.env`:
- `HF_TOKEN` — a Hugging Face access token with access to the [pyannote speaker-diarization](https://huggingface.co/pyannote/speaker-diarization) gated model. Required only if you use speaker diarization.
- `DJANGO_SECRET_KEY` — generate your own for anything beyond local development.

Export the variables into your shell before running the backend (e.g. `set -a; source .env; set +a`), since the app reads them via `os.environ`.

### 3. Run database migrations

```bash
cd vocal_blade_backend/src
uv run python manage.py migrate
```

### 4. Local model assets

A few large local-only assets are not tracked in this repo (see `.gitignore`) and must be downloaded/placed separately if you use the features that depend on them:
- `models/` — model weights (Qwen2.5-7B for AI-assisted editing, `vox-cpk.pth.tar` for FOMM-based Smart Cut, etc.).
- `third_party/` — vendored external projects (SadTalker, First-Order-Motion-Model) used by the visual continuity engine; clone the relevant upstream repos into this directory yourself if you need them.

Core transcription and text-based editing work without either of these — they're only needed for AI-assisted editing and the neural visual continuity methods.

## Running

```bash
./start.sh
```

Starts both the backend (`:8000`) and frontend (`:3000`), and stops any process already bound to those ports first. Logs go to `/tmp/backend.log` and `/tmp/frontend.log`.

Alternatively, run each service separately:

```bash
./start-backend.sh          # backend only, sets up CUDA 11/12 library paths
cd frontend-mvp && npm run dev   # frontend only
```

## License

MIT — see [LICENSE](LICENSE).
