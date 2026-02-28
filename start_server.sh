#!/usr/bin/env bash
# start_server.sh - Launch the FastAPI inference server
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "Starting RAN AI Lab inference server..."
echo "Health check: http://localhost:8000/health"
echo "API docs:     http://localhost:8000/docs"

# Production tuning advice:
# - Use --workers 2 for multi-core (but keep 1 for low-RAM systems)
# - Use gunicorn + uvicorn workers for production
# - Enable --limit-concurrency 10 to prevent OOM

exec uvicorn serving.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --timeout-keep-alive 30
