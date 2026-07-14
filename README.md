# VocalBlade

GPU-accelerated audio transcription powered by [WhisperX](https://github.com/m-bain/whisperX) — word-level timestamps, alignment, and speaker diarization, served through a Django + Strawberry GraphQL backend with a React/TypeScript frontend.

## Features

- Transcription with WhisperX (all Whisper model sizes, 98 supported languages)
- Word-level timestamp alignment
- Speaker diarization (via `pyannote`)
- Single GraphQL API endpoint (`/graphql/`)
- React + TypeScript + Vite + Tailwind frontend with upload, model/language selection, and transcript display

## Architecture

```
vocal_blade_backend/   Django + Strawberry GraphQL backend (Python, managed by uv)
frontend-mvp/           React + TypeScript + Vite + Tailwind frontend
docs/                    API reference and setup guide
```

See [docs/API.md](docs/API.md) for the GraphQL schema and [docs/SETUP.md](docs/SETUP.md) for a detailed setup walkthrough.

## Prerequisites

- Python 3.12, managed via [uv](https://github.com/astral-sh/uv)
- Node.js 20+
- ffmpeg
- An NVIDIA GPU is strongly recommended for reasonable transcription speed. If you also enable speaker diarization, `onnxruntime`/`pyannote` require CUDA 11 libraries alongside PyTorch's CUDA 12 — see [docs/SETUP.md](docs/SETUP.md) for how the CUDA 11/12 coexistence is handled.

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
- `models/` — model weights (e.g. Qwen2.5-7B, `vox-cpk.pth.tar`) for optional features beyond core transcription.
- `third_party/` — vendored external projects used for optional features; clone the relevant upstream repos into this directory yourself if you need them.

Core WhisperX transcription works without either of these.

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
