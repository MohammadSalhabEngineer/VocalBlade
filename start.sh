#!/bin/bash
# VocalBlade — launch backend (:8000) + frontend (:3000)

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── Kill any existing processes on these ports ────────────────────────────
for PORT in 8000 3000; do
  PID=$(lsof -ti tcp:$PORT 2>/dev/null)
  if [ -n "$PID" ]; then
    echo "Stopping existing process on :$PORT (PID $PID)"
    kill "$PID" 2>/dev/null
    sleep 1
  fi
done

# ── Backend ───────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="$ROOT/vocal_blade_backend/.cudnn8/lib:$LD_LIBRARY_PATH"
source "$HOME/.local/bin/env"

echo "Starting backend  → http://localhost:8000"
(
  cd "$ROOT/vocal_blade_backend/src"
  uv run python manage.py runserver 0.0.0.0:8000 --noreload >> /tmp/backend.log 2>&1
) &
BACKEND_PID=$!

# ── Frontend ──────────────────────────────────────────────────────────────
echo "Starting frontend → http://localhost:3000"
(
  cd "$ROOT/frontend-mvp"
  npm run dev >> /tmp/frontend.log 2>&1
) &
FRONTEND_PID=$!

# ── Wait for both ports to open ───────────────────────────────────────────
echo "Waiting for services..."

for PORT in 8000 3000; do
  for i in $(seq 1 30); do
    if ss -tlnp | grep -q ":$PORT"; then
      echo "  :$PORT  ready"
      break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
      echo "  :$PORT  timed out — check /tmp/backend.log or /tmp/frontend.log"
    fi
  done
done

echo ""
echo "VocalBlade is running:"
echo "  Backend  → http://localhost:8000"
echo "  Frontend → http://localhost:3000"
echo ""
echo "Logs: /tmp/backend.log  |  /tmp/frontend.log"
echo "Press Ctrl+C to stop both services."

# ── Keep script alive; kill children on exit ─────────────────────────────
trap "echo 'Stopping...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
