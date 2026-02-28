# Makefile for RAN AI Lab
# Works on Linux/WSL. Use `make <target>` to run.

SHELL := /bin/bash
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYRUN := $(VENV)/bin/python

.PHONY: env deps run-demo train serve test clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

env:  ## Create Python venv
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

deps: env  ## Install all dependencies
	bash setup.sh

run-demo:  ## Run full demo (experiment + server + dashboard)
	bash examples/demo_run.sh

train:  ## Build dataset and train models
	$(PYRUN) -c "import sys; sys.path.insert(0,'.'); from dataset.builder import build_dataset; build_dataset()"
	$(PYRUN) -m ml.train
	$(PYRUN) -m ml.optimize

serve:  ## Start FastAPI inference server
	bash start_server.sh

dashboard:  ## Start Streamlit dashboard
	$(VENV)/bin/streamlit run dashboard/app.py --server.port 8501 --server.headless true

test:  ## Run unit tests
	$(VENV)/bin/pytest tests/ -v --tb=short

experiment:  ## Run simulated experiment (data collection only)
	bash run_experiment.sh

clean:  ## Remove generated files (keeps code)
	rm -rf models/*.joblib models/*.pt models/*.onnx models/*.pth models/*.json
	rm -rf data/*.csv data/*.parquet
	rm -rf logs/*.jsonl logs/*.jsonl.*
	rm -rf __pycache__ */__pycache__ */*/__pycache__
