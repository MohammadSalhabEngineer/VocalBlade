# VocalBlade Setup Guide

## Prerequisites
- GCP VM with NVIDIA GPU (L4/T4)
- SSH access via VS Code Remote-SSH

## Backend Setup

### 1. Install UV (Python package manager)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

### 2. Install dependencies
```bash
cd ~/vocalblade/vocal_blade_backend
uv sync
```
This installs Python 3.12, creates `.venv`, and installs all packages (Django, WhisperX, PyTorch with CUDA).

### 3. Run migrations
```bash
cd ~/vocalblade/vocal_blade_backend/src
uv run python manage.py migrate
```

### 4. Start backend server
```bash
uv run python manage.py runserver 0.0.0.0:8000
```

### 5. Verify
```bash
curl -X POST http://localhost:8000/graphql/ \
  -H "Content-Type: application/json" \
  -d '{"query": "{ availableModels }"}'
```

## Frontend Setup

### 1. Install Node.js 20
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### 2. Install dependencies
```bash
cd ~/vocalblade/frontend-mvp
npm install
```

### 3. Start dev server
```bash
npm run dev
```
Frontend runs on port 3000, proxies `/graphql` to backend on port 8000.

## Accessing via SSH Port Forwarding

In your local terminal:
```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 user@your-gcp-ip
```

Then open `http://localhost:3000` in your browser.

VS Code Remote-SSH auto-forwards ports — check the "Ports" tab.

## Project Structure
```
~/vocalblade/
├── vocal_blade_backend/     # Django + Strawberry GraphQL backend
│   ├── pyproject.toml       # Dependencies (managed by UV)
│   ├── src/
│   │   ├── manage.py        # Django entry point
│   │   ├── vocal_blade/     # Django project settings
│   │   ├── core/            # User auth models
│   │   └── transcriber/     # WhisperX transcription
│   └── .venv/               # Python virtual environment
├── frontend-mvp/            # React + TypeScript + Tailwind
│   ├── src/
│   │   ├── api/graphql.ts   # GraphQL client
│   │   ├── components/      # React components
│   │   └── App.tsx          # Main app
│   └── vite.config.ts       # Vite + proxy config
└── docs/                    # Documentation
```

## Troubleshooting

**Backend won't start:** Check that port 8000 isn't in use: `ss -tlnp | grep 8000`

**GPU not detected:** Verify with `nvidia-smi`. PyTorch CUDA comes bundled — no separate CUDA toolkit needed.

**Frontend can't reach backend:** Ensure backend is running on port 8000 and the Vite proxy is configured in `vite.config.ts`.
