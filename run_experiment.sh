#!/usr/bin/env bash
# run_experiment.sh - Run the full experiment pipeline
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=== RAN AI Lab - Full Experiment ==="
echo "Start time: $(date)"

# Step 1: Run simulated experiment (topology + traffic + telemetry)
echo ""
echo "[1/5] Running simulated experiment (60s data collection)..."
python -c "
import sys, os, time, logging, yaml
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

from simulator.topology import start_topology, stop_topology, load_config
from simulator.traffic import TrafficGenerator
from simulator.controllers import ControllerRunner
from telemetry.collector import TelemetryCollector

cfg = load_config()
# Force simulated mode for this script (safe on any OS)
cfg['experiment']['simulated_mode'] = True

net = start_topology(cfg)
tg = TrafficGenerator(net, cfg)
runner = ControllerRunner(net, cfg)
collector = TelemetryCollector(net, tg, runner, cfg)

tg.start_all()
runner.start()
collector.start()

duration = cfg['experiment']['duration']
print(f'Collecting telemetry for {duration}s...')
time.sleep(duration)

collector.stop()
runner.stop()
tg.stop()
stop_topology(net)
print(f'Done. {len(collector.get_records())} records collected.')
"

# Step 2: Build dataset
echo ""
echo "[2/5] Building ML dataset..."
python -c "
import sys; sys.path.insert(0, '.')
import logging; logging.basicConfig(level=logging.INFO)
from dataset.builder import build_dataset
df = build_dataset()
print(f'Dataset: {df.shape[0]} rows, {df.shape[1]} cols, conflict rate: {df[\"conflict\"].mean():.2%}')
"

# Step 3: Train models
echo ""
echo "[3/5] Training models..."
python -m ml.train

# Step 4: Optimize (ONNX export)
echo ""
echo "[4/5] Optimizing models..."
python -m ml.optimize

# Step 5: Print summary
echo ""
echo "[5/5] Experiment complete!"
echo "Artifacts:"
echo "  - Data:   data/telemetry.csv, data/dataset.parquet"
echo "  - Models: models/lgbm_model.joblib, models/mlp_traced.pt"
echo "  - Logs:   logs/telemetry.jsonl"
echo ""
echo "To start the API server:"
echo "  uvicorn serving.api:app --host 0.0.0.0 --port 8000"
echo ""
echo "To start the dashboard:"
echo "  streamlit run dashboard/app.py"
echo ""
echo "End time: $(date)"
