#!/usr/bin/env bash
# demo_run.sh - Quick demo: run experiment + server + dashboard
# Uses background processes. Ctrl+C to stop all.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

trap "kill 0; exit" SIGINT SIGTERM

echo "=== RAN AI Lab Demo ==="

# Run the data pipeline first
bash run_experiment.sh

# Start API server in background
echo "Starting API server on :8000..."
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --workers 1 --log-level warning &
API_PID=$!
sleep 3

# Test health
if curl -s http://localhost:8000/health | grep -q "ok"; then
    echo "API server is healthy"
else
    echo "WARNING: API server may not be ready"
fi

# Start Streamlit dashboard
echo "Starting Streamlit dashboard on :8501..."
echo "Open http://localhost:8501 in your browser"
streamlit run dashboard/app.py --server.port 8501 --server.headless true &
DASH_PID=$!

echo ""
echo "=== Demo running ==="
echo "API:       http://localhost:8000/health"
echo "Dashboard: http://localhost:8501"
echo "Press Ctrl+C to stop all"
wait
