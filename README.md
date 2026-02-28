# IntelliRAN — 5G xApp Conflict Detection with Sub-Millisecond AI

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**Industry-oriented simulation of 5G NR / O-RAN near-RT RIC with sub-millisecond ML-based xApp conflict detection.**

> A full-stack AI/ML platform that simulates how near-Real-Time RAN Intelligent Controllers (near-RT RIC) detect and handle policy conflicts between competing xApps in O-RAN 5G networks. Built on 3GPP TS 38.xxx standards, O-RAN WG3 E2SM-KPM, and WG2 AI/ML workflows.

---

## Table of Contents

1. [What This Is](#what-this-is)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Standards Alignment](#standards-alignment)
5. [Components Deep Dive](#components-deep-dive)
6. [ML Models and Results](#ml-models-and-results)
7. [Inference Benchmark](#inference-benchmark)
8. [Quick Start](#quick-start)
9. [Full Pipeline](#full-pipeline)
10. [Testing](#testing)
11. [API Reference](#api-reference)
12. [Dashboard](#dashboard)
13. [Configuration Reference](#configuration-reference)
14. [Public Repository Safety](#public-repository-safety)
15. [Dependencies](#dependencies)
16. [Troubleshooting](#troubleshooting)
17. [Roadmap](#roadmap)
18. [Citation and Attribution](#citation-and-attribution)

---

## What This Is

This project simulates the AI/ML layer of an **O-RAN near-RT RIC** — the decision-making brain of a modern 5G base station network. Specifically it models the **xApp conflict detection** problem:

When multiple AI-driven xApps independently control the same radio cells (e.g. one xApp tries to save energy by reducing transmit power while another tries to boost QoS by increasing power), their actions **conflict**. A conflict detection ML model running inside the RIC must identify these in **< 1 ms** to allow the conflict mitigation function to arbitrate before the control actions reach the gNB.

### Why This Matters

- O-RAN disaggregation means **dozens of xApps** from different vendors can simultaneously issue E2 control commands to gNodeBs
- Undetected conflicts degrade real user experience: dropped calls, latency spikes, throughput collapse, energy waste
- 3GPP and O-RAN specs require near-RT RIC control loops at **10–100 ms** — the AI conflict detection inference must complete in **sub-millisecond** time to leave headroom for arbitration, E2 message encoding, and transport
- This project demonstrates the full pipeline: simulate radio network → collect 3GPP PM counters → engineer 125 features → train 5 ML models → optimize to ONNX → serve via REST API → explain predictions with LIME/SHAP/counterfactuals

### What It Is NOT

- Not real radio hardware or real over-the-air network traffic
- Not a production O-RAN RIC deployment (no real E2 interface, no O1 management plane, no real gNB)
- Not a network emulator (no packet-level simulation, no real PHY/MAC/RLC layers)
- Not affiliated with O-RAN Alliance, 3GPP, or any telecom vendor

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    O-RAN near-RT RIC (Simulated)                │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │TS xApp   │  │QoS xApp  │  │MLB xApp  │  │ES xApp       │   │
│  │Handover  │  │PRB/Power │  │Load Bal. │  │Energy Saving │   │
│  │Carrier   │  │MCS/Sched │  │CIO/PRB   │  │MIMO/DTX/DRX  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘   │
│       └──────────────┴──────────────┴───────────────┘            │
│                          │ XAppActions                            │
│              ┌───────────▼───────────┐                           │
│              │  Conflict Detector    │ ← ML model (this project) │
│              │  ONNX: 0.039ms mean   │                           │
│              │  p99:  0.116ms        │                           │
│              └───────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
         │ E2-like PM Counters (3GPP TS 28.552)
         ▼
┌─────────────────────────────────────────────────────────────────┐
│              RAN Simulator (3GPP TS 38.901 UMa)                 │
│                                                                  │
│   Cell-01 (n78 FR1 3.5GHz 100MHz 273PRBs)                      │
│   Cell-02 (n78 FR1 3.5GHz 100MHz 273PRBs)                      │
│   Cell-03 (n257 FR2 mmWave 28GHz 400MHz)                       │
│                                                                  │
│   500 UEs: 60 eMBB_premium, 120 eMBB_standard,                 │
│            20 URLLC, 300 mMTC_IoT                               │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Simulator (topology + traffic + controllers)
    │
    ├─→ 21 PM counters per cell per 10ms tick
    ├─→ xApp action counts per tick
    ├─→ Conflict counts per tick
    │
    ▼
Telemetry Collector (100 Hz)
    │
    ▼  data/telemetry.parquet  (705+ rows × 53 columns)
    │
Feature Engineering (telemetry/metrics.py)
    │
    ▼  125 derived features (rolling stats, spectral efficiency,
    │   MCS entropy, cross-cell differentials, action intensity)
    │
Dataset Builder (dataset/builder.py)
    │
    ▼  data/dataset.parquet  (405+ rows × 137 columns)
    │   5 conflict types + binary label + severity score
    │
ML Training Pipeline (ml/train.py)
    │
    ├─→ LightGBM       (AUC 0.781)
    ├─→ XGBoost         (AUC 0.990) ← best
    ├─→ Random Forest   (AUC 0.913)
    ├─→ Logistic Reg.   (AUC 0.832)
    ├─→ PyTorch MLP     (AUC 0.787)
    │
    ▼
ONNX Optimization (ml/optimize.py)
    │
    ├─→ mlp_model.onnx       (fp32)
    ├─→ mlp_model_quant.onnx (uint8 quantized)
    │
    ▼
FastAPI Server (serving/api.py)
    │
    ├─→ POST /predict        (single prediction + LIME + SHAP + counterfactual)
    ├─→ POST /predict/batch  (batch predictions)
    ├─→ GET  /health         (model status)
    ├─→ GET  /metrics        (latency percentiles)
    │
    ▼
Streamlit Dashboard (dashboard/app.py)
    │
    └─→ 5 interactive pages: KPIs, Conflicts, Models, Live Inference, Benchmarks
```

---

## Project Structure

```
intelliran/
├── config/
│   └── network.yaml            3GPP/O-RAN parameters: cells, UE profiles, xApps, ML config
│
├── simulator/
│   ├── topology.py             RANSimulator: gNB cells, UE attachment, 3GPP UMa path loss, handover
│   ├── traffic.py              TrafficModel: Ornstein-Uhlenbeck per-cell load, flash crowd events
│   └── controllers.py          4 xApps + conflict detection engine + ControllerRunner
│
├── telemetry/
│   ├── collector.py            E2SM-KPM PM counter collector running at 100 Hz
│   └── metrics.py              Feature engineering: 125 derived 5G KPI features
│
├── dataset/
│   └── builder.py              Multi-label conflict labeling + dataset assembly
│
├── ml/
│   ├── models.py               5 model implementations: LightGBM, XGBoost, RF, LogReg, MLP
│   ├── train.py                Training pipeline: temporal split, evaluation, model comparison
│   ├── inference.py            Sub-ms production inference: ONNX, warm-up, pre-allocated buffers
│   └── optimize.py             ONNX export, INT8 quantization, multi-format benchmark
│
├── explainability/
│   ├── lime_analysis.py        LIME tabular explainer for any model type
│   ├── shap_analysis.py        SHAP TreeExplainer (tree models) + KernelExplainer (MLP/LogReg)
│   └── counterfactual.py       Minimal feature perturbation to flip predictions
│
├── serving/
│   └── api.py                  FastAPI: /predict, /predict/batch, /health, /metrics
│
├── dashboard/
│   └── app.py                  Streamlit: 5-page interactive dashboard
│
├── tests/
│   └── test_basic.py           16 tests covering all layers
│
├── .gitignore                  Excludes generated data, model binaries, venv
├── requirements.txt            Pinned dependencies (CPU-only PyTorch)
├── Makefile                    Build automation
├── run_experiment.sh           Full pipeline runner (Linux/Mac)
├── setup.sh                    Environment setup (Linux/Mac)
├── start_server.sh             API + dashboard launcher
└── examples/
    └── demo_run.sh             Quick demo script
```

**Not committed (generated at runtime):**
```
data/                           telemetry.parquet, dataset.parquet, CSVs
models/                         .joblib, .onnx, .pt, .pth, .npz model files
logs/                           telemetry.jsonl
venv/                           Python virtual environment
```

---

## Standards Alignment

| Layer | Standard | What Is Implemented |
|-------|----------|---------------------|
| Cell configuration | 3GPP TS 38.104 | NR band n78 FR1 (3.5 GHz, 100 MHz BW, 273 PRBs, 30 kHz SCS) and n257 FR2 mmWave (28 GHz, 400 MHz BW) |
| Path loss model | 3GPP TR 38.901 | Urban Macro (UMa) NLOS: PL = 28 + 22·log₁₀(d) + 20·log₁₀(fc) with log-normal shadow fading (σ=4 dB) and Rayleigh fast fading |
| PM counters | 3GPP TS 28.552 | 21 counters per cell: DRB.UEThpDl/Ul, RRU.PrbUsedDl/Ul, L1M.RS-SINR/RSRP/RSRQ, DRB.RlcSduDelayDl, DRB.PacketLossRateDl/Ul, TB.TotNbrDl.{Qpsk,16Qam,64Qam,256Qam}, RRC.ConnEstabSucc/Att, HO.SuccOutInterEnb/AttOutInterEnb |
| CQI/MCS mapping | 3GPP TS 38.214 | Simplified CQI = (SINR+5)/2.7 mapping (Table 5.2.2.1-3 approximation), MCS = CQI × 1.9 |
| RRC procedures | 3GPP TS 38.331 | RRC connection establishment success/attempt counters, Cell Individual Offset (CIO) for handover |
| xApp E2 actions | O-RAN WG3 E2SM-RC v1.0 | Structured XAppAction with action_type, target_cell, dimension, value — covering handover, PRB reservation, power boost, MCS override, carrier aggregation/shutdown, MIMO layer control, scheduling weight, CIO adjust |
| Telemetry format | O-RAN WG3 E2SM-KPM v3.0 | Per-cell KPM indication report at 10 ms granularity (100 Hz) |
| AI/ML feature engineering | O-RAN WG2 AI/ML v2.0 | Rolling statistics, spectral efficiency, MCS entropy, cross-cell differentials — aligned with WG2 recommended KPI categories |
| Network slicing | 3GPP TS 28.530 | 4 UE profiles mapping to eMBB, URLLC, mMTC network slice types with differentiated QoS (QFI, min throughput, max latency) |

---

## Components Deep Dive

### Simulator Layer (`simulator/`)

**topology.py — RANSimulator**

Initializes a 3-cell 5G NR network with 500 UEs:
- Cell-01 and Cell-02: NR band n78 (FR1, 3.5 GHz), 100 MHz bandwidth, 273 PRBs, 30 kHz SCS, 43 dBm tx power, 64T64R antenna, 4 MIMO layers, positioned in a triangular grid (inter-site distance ~1 km)
- Cell-03: NR band n257 (FR2 mmWave, 28 GHz), 400 MHz bandwidth, 264 PRBs, 120 kHz SCS
- UE profiles: 60 eMBB_premium (50+ Mbps, <10ms, stationary), 120 eMBB_standard (10+ Mbps, <50ms, pedestrian), 20 URLLC (1+ Mbps, <1ms, vehicular), 300 mMTC_IoT (0.1 Mbps, <1000ms, stationary)
- Per-UE radio channel: 3GPP TR 38.901 UMa path loss with 4 dB shadow fading std, Rayleigh fast fading approximation, Doppler shift modeling
- Channel-to-KPI mapping: SINR → CQI (TS 38.214), CQI → MCS, MCS → BLER model, Shannon capacity with practical efficiency factor (0.6), rank adaptation from SINR
- Handover execution with 95% success probability, source/target cell counter updates
- `step(dt)` advances: UE mobility (random walk bounded to ±1 km), channel update for all connected UEs, cell-level counter aggregation (PRB usage proportional to connected UEs / capacity)

**traffic.py — TrafficModel**

Per-cell traffic load evolves as an Ornstein-Uhlenbeck stochastic process:
- Mean reversion: θ = 0.1, mean μ = 0.5 (50% average load)
- Volatility: σ = 0.08, bounded to [5%, 98%] utilization
- Flash crowd events: 0.1% probability per step of a 3–10 second burst on a random cell (load jumps to 98%)
- Per-UE throughput modulation: sharing factor = max(0.05, 1 - load × 0.8), latency increases with load
- Profile-specific buffer modeling: eMBB_premium (50 KB × load), URLLC (500 B × load), mMTC (sparse bursts with 1% duty cycle)

**controllers.py — 4 xApps + ControllerRunner**

Each xApp runs every control loop interval (configurable, default 10 ms) and independently decides actions based on cell snapshots:

| xApp | Trigger Conditions | Actions Generated |
|------|-------------------|-------------------|
| TrafficSteeringXApp (priority 1) | UE RSRP < -100 dBm | `handover` to best neighbor (with 3 dB hysteresis) |
| | eMBB_premium UE with SINR > 20 dB | `carrier_aggregation` (5% probability per eligible UE) |
| QoSOptimizationXApp (priority 2) | URLLC UEs present AND PRB util > 70% | `prb_reservation` (20 PRBs for URLLC) |
| | Edge UEs with SINR < 5 dB | `power_boost` (+3 dB) |
| | Cell BLER > 10% | `mcs_override` (-2 MCS steps) |
| | PRB util > 85% | `scheduling_weight` (1.5×) |
| MLBXApp (priority 3) | Cell load > avg + 20% | `cio_adjust` (+2 dB to push UEs away), `prb_cap` (80%) |
| | Cell load < avg - 20% | `cio_adjust` (-2 dB to attract UEs), `ho_threshold` reduction |
| EnergySavingXApp (priority 4) | PRB util < 25% | `mimo_layer_reduce`, `dtx_drx` enable |
| | PRB util < 10% | `carrier_shutdown` |
| | PRB util > 60% | `mimo_layer_reduce` (restore layers) |

`detect_action_conflicts()` compares every pair of actions from different xApps targeting the same cell. A conflict is detected when:
1. The action pair matches a known conflict dimension mapping (e.g. handover vs. cio_adjust = HO dimension conflict), OR
2. Both actions target the same resource dimension (e.g. both target "prb") with opposing values (one positive, one negative)

### Telemetry Layer (`telemetry/`)

**collector.py** — Runs in a background thread at 100 Hz:
1. Calls `sim.get_cell_snapshot(cid)` for each of the 3 cells → 21 PM counters + cell config state per cell
2. Calls `controller.get_action_summary()` → 12 xApp action count columns
3. Counts recent conflicts per cell from the controller's conflict log
4. Writes each record to CSV (for compatibility) and JSONL (for streaming)
5. At shutdown, consolidates all records into `data/telemetry.parquet` via PyArrow

Output schema: 53 columns per row = timestamp + step + cell_id + PCI + band + 21 PM counters + 6 cell config params + 12 xApp action counts + n_conflicts + cell_load

**metrics.py — Feature Engineering**

Transforms 53 raw PM columns into 125 ML-ready features:

| Feature Category | Count | Description |
|-----------------|-------|-------------|
| Rolling mean (MA) | 13 | 100-step rolling average for each of 13 KPIs |
| Rolling std | 13 | 100-step rolling standard deviation |
| Rate of change | 13 | First difference (velocity) |
| Acceleration | 13 | Second difference |
| Spectral efficiency DL | 1 | throughput / (PRBs × 12 subcarriers × 14 symbols × 30 kHz) |
| PRB utilization ratio | 1 | PrbUsedDl / PrbAvailDl |
| PRB headroom | 1 | 1 - utilization ratio |
| RRC success rate | 1 | ConnEstabSucc / ConnEstabAtt |
| HO success rate | 1 | HO.SuccOutInterEnb / HO.AttOutInterEnb |
| MCS entropy | 1 | Shannon entropy of [QPSK, 16QAM, 64QAM, 256QAM] distribution |
| Throughput burstiness | 1 | Rolling coefficient of variation |
| CQI deviation | 1 | Observed CQI - expected CQI from SINR |
| Delay jitter | 1 | Absolute first difference of DRB.RlcSduDelayDl |
| Total xApp actions | 1 | Sum of all 12 action columns |
| xApp action rolling sums | 12+ | Per-action-type rolling sum over window |
| PRB util vs. cell average | 1 | Per-cell differential from step mean |
| Throughput vs. cell average | 1 | Per-cell differential from step mean |
| Load imbalance index | 1 | Per-step std of cell_load across cells |

### Dataset Builder (`dataset/builder.py`)

Multi-label conflict labeling across 5 resource dimensions:

| Conflict Label | Logic | Meaning |
|---------------|-------|---------|
| `conflict_ho` | ts_handover > 0 AND (mlb_cio_adjust > 0 OR mlb_ho_threshold > 0) | TS and MLB both trying to control handover parameters simultaneously |
| `conflict_prb` | qos_prb_reservation > P75 AND mlb_prb_cap > P75 | QoS reserving PRBs while MLB caps them — directly contradictory |
| `conflict_power` | qos_power_boost > 0 AND es_mimo_reduce > 0 | QoS boosting power while ES reducing radio capability |
| `conflict_carrier` | ts_carrier_agg > 0 AND es_carrier_shutdown > 0 | TS adding carriers while ES shutting them down |
| `conflict_sched` | qos_sched_weight > P75 AND mlb_prb_cap > P75 | QoS increasing scheduling priority while MLB constraining resources |
| **`conflict`** (binary) | OR of all 5 above + KPI deterioration check | Master conflict flag for ML training |
| `conflict_severity` | Sum of active conflict dimensions | 0–5 continuous severity score |

KPI deterioration is detected when:
- Throughput drops > 20% below its rolling mean
- Latency spikes > 50% above its rolling mean
- BLER exceeds 3× its rolling mean

Warmup period (first `warmup_s / poll_interval_ms` steps) is excluded to avoid unstable initial transients.

### ML Layer (`ml/`)

**models.py — 5 Model Implementations**

| Model | Key Hyperparameters | Notes |
|-------|-------------------|-------|
| LightGBM | n_estimators=300, max_depth=6, lr=0.03, subsample=0.8, reg_alpha=0.5, reg_lambda=1.0 | Early stopping patience=30, `is_unbalance` auto |
| XGBoost | n_estimators=300, max_depth=6, lr=0.03, tree_method=hist, reg_alpha=0.5, reg_lambda=1.0 | `scale_pos_weight` for class imbalance |
| Random Forest | 200 trees, max_depth=8, min_samples_split=20, min_samples_leaf=10 | `class_weight='balanced'` not used (manual) |
| Logistic Regression | Pipeline(StandardScaler → LR(C=1.0, max_iter=1000, solver=lbfgs)) | L2 regularization |
| ConflictMLP | Linear(D→128)→BN→ReLU→Drop(0.15)→Linear(128→64)→BN→ReLU→Drop(0.15)→Linear(64→32)→BN→ReLU→Drop(0.15)→Linear(32→1)→Sigmoid | AdamW, CosineAnnealingLR, gradient clip=1.0, early stop=10, built-in StandardScaler |

**train.py — Training Pipeline**
- Loads dataset.parquet, applies temporal 80/20 split (not random — prevents data leakage from rolling features)
- Trains all 5 models sequentially with evaluation metrics: AUC, precision, recall, F1, confusion matrix, train-test gap
- Saves all model artifacts to `models/` directory
- Exports MLP to TorchScript + saves scaler parameters for inference

**inference.py — ConflictPredictor**
- Model loading priority: quantized ONNX → ONNX → LightGBM → XGBoost → TorchScript
- ONNX Runtime optimizations: `ORT_ENABLE_ALL` graph optimization, CPU memory arena, memory pattern, single-threaded (avoids context switching overhead for small models)
- Pre-allocated numpy input buffer (zero per-call allocation)
- 50-iteration warm-up on startup (fills instruction cache, JIT-compiles ONNX graph)
- Returns: probability, binary conflict flag, inference latency in ms, model type used

**optimize.py — ONNX Optimization**
- TorchScript → ONNX export (opset 13, `do_constant_folding=True`, dynamic batch axis)
- Dynamic quantization: weight matrices stored as uint8, dequantized at runtime
- Multi-format benchmark (n=1000): LightGBM, XGBoost, TorchScript, ONNX fp32, ONNX quantized
- Each benchmark includes 50-iteration warmup before measurement

### Explainability Layer (`explainability/`)

| Method | Implementation | Output |
|--------|---------------|--------|
| LIME | LimeTabularExplainer with 500-sample background, 200 perturbation samples | Top-k feature weights + local prediction |
| SHAP | TreeExplainer for LightGBM/XGBoost/RF (exact Shapley values), KernelExplainer for MLP/LogReg | SHAP values per feature + base value |
| Counterfactual | Iterative perturbation of top-importance features across min/max range | Minimal feature changes to flip prediction |

### Serving Layer (`serving/api.py`)

FastAPI server with automatic Swagger UI at `/docs`:
- `/predict`: single sample → probability + conflict flag + latency + LIME + SHAP + counterfactual
- `/predict/batch`: array of samples → array of predictions
- `/health`: model status, feature count, explainer availability
- `/metrics`: request count, average latency, p50/p95/p99 latency (rolling 1000-request window)

### Dashboard (`dashboard/app.py`)

Streamlit with 5 pages:
1. **Cell KPI Overview**: per-cell time series (PRB util, throughput, SINR, CQI, BLER), raw data table
2. **xApp Conflict Analysis**: conflict label distribution, severity histogram, rolling conflict rate
3. **Model Comparison**: AUC bar chart, metrics table, inference benchmark chart
4. **Live Inference**: pick any dataset row, run prediction, see probability/conflict/latency
5. **Benchmark Results**: per-format latency metrics with sub-ms pass/fail indicators

---

## ML Models and Results

Trained on 405 samples, 80/20 temporal split, 125 features, 34.8% conflict rate.

| Model | Test AUC | Train AUC | Overfit Gap | Precision | Recall | F1 | Train Time |
|-------|----------|-----------|-------------|-----------|--------|----|------------|
| **XGBoost** | **0.990** | 1.000 | +0.010 | 0.750 | 0.692 | **0.720** | 2.1s |
| Random Forest | 0.913 | 0.999 | +0.085 | 0.750 | 0.692 | 0.720 | 1.2s |
| Logistic Regression | 0.832 | 0.997 | +0.165 | 0.500 | 0.562 | 0.529 | 0.1s |
| MLP (BatchNorm) | 0.787 | 0.997 | +0.210 | 0.500 | 0.500 | 0.500 | 5.4s |
| LightGBM | 0.781 | 0.992 | +0.211 | 0.000 | 0.000 | 0.000 | 0.8s |

**Notes on results:**
- The small dataset (405 rows) causes overfitting in all models. XGBoost has the tightest gap (+0.010) due to its strong built-in L1/L2 regularization.
- LightGBM's F1=0 is caused by aggressive early stopping on a small validation set — it predicts all samples as non-conflict.
- Increasing simulation duration from 30s to 300s (via `experiment.duration_s` in config) produces ~15,000 rows and significantly improves all models.
- The temporal split ensures no data leakage from rolling features — a random split would artificially inflate metrics.

---

## Inference Benchmark

Measured on CPU, n=1000 iterations, single sample with 125 features, 50-iteration warmup:

| Format | Mean | P50 | P95 | P99 | Sub-ms (p99 < 1ms) |
|--------|------|-----|-----|-----|---------------------|
| **ONNX Runtime (fp32)** | **0.039 ms** | 0.036 ms | 0.054 ms | **0.116 ms** | **YES** |
| ONNX Quantized (uint8) | 0.064 ms | 0.054 ms | 0.121 ms | 0.343 ms | YES |
| TorchScript | 0.188 ms | 0.163 ms | 0.336 ms | 0.820 ms | YES |
| XGBoost (native) | 0.588 ms | 0.495 ms | 1.148 ms | 2.013 ms | NO (p99 > 1ms) |
| LightGBM (native) | 1.935 ms | 1.662 ms | 3.391 ms | 4.805 ms | NO |

**ONNX Runtime achieves p99 = 0.116 ms** — well within the O-RAN near-RT RIC 10 ms control loop budget. The O-RAN spec allocates 10–50 ms for the full RIC processing round trip; this model uses < 0.12% of that budget at the 99th percentile.

---

## Quick Start

### Prerequisites

- **Python 3.10** (strictly — Python 3.11+ may have PyTorch compatibility issues with CPU wheels)
- **pip** (bundled with Python)
- **Git** (for cloning and pushing)
- Windows 10/11 or Linux (tested on Windows 11 with Python 3.10.0)
- ~2 GB free disk space (venv + dependencies + model artifacts)

### Setup

```bash
git clone https://github.com/CodeforGood1/intelliran.git
cd intelliran

python3.10 -m venv venv

# Activate virtual environment:
# Windows PowerShell:
.\venv\Scripts\activate
# Windows CMD:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Step 1: Simulate 5G network and collect PM counter telemetry (30 seconds)
python -m telemetry.collector
# Output: data/telemetry.parquet (705+ rows, 53 columns)

# Step 2: Engineer features and label conflicts
python -m dataset.builder
# Output: data/dataset.parquet (405+ rows, 137 columns, ~35% conflict rate)

# Step 3: Train all 5 ML models
python -m ml.train
# Output: models/*.joblib, models/mlp_traced.pt, models/training_results.json

# Step 4: Export to ONNX, quantize, and benchmark
python -m ml.optimize
# Output: models/mlp_model.onnx, models/mlp_model_quant.onnx, models/benchmark_results.json

# Step 5: Start the inference API server
python -m serving.api
# → http://localhost:8000/health
# → http://localhost:8000/docs  (Swagger UI)

# Step 6: Start the dashboard (in a separate terminal)
# Windows — IMPORTANT: use the venv streamlit directly:
.\venv\Scripts\streamlit.exe run dashboard/app.py
# Linux/Mac:
streamlit run dashboard/app.py
# → http://localhost:8501
```

---

## Full Pipeline

```
telemetry.collector  →  data/telemetry.parquet   (705+ rows × 53 PM counter columns)
        │
dataset.builder      →  data/dataset.parquet     (405+ rows × 137 cols = 125 features + labels)
        │
ml.train             →  models/lgbm_model.joblib
                         models/xgb_model.joblib
                         models/rf_model.joblib
                         models/logreg_model.joblib
                         models/mlp_traced.pt        (TorchScript)
                         models/mlp_state.pth        (state dict)
                         models/mlp_scaler.npz       (StandardScaler mean/std)
                         models/feature_cols.json    (125 feature names)
                         models/training_results.json
        │
ml.optimize          →  models/mlp_model.onnx       (fp32 ONNX)
                         models/mlp_model_quant.onnx (uint8 quantized)
                         models/benchmark_results.json
        │
serving.api          ←  loads models + LIME + SHAP + counterfactual
        │
dashboard/app.py     ←  reads parquet files + model artifacts + results JSONs
```

---

## Testing

### Running the Full Test Suite

```bash
# From the project root directory with the venv activated:
python -m pytest tests/test_basic.py -v

# Or on Windows with explicit venv python:
.\venv\Scripts\python.exe -m pytest tests/test_basic.py -v
```

Expected output: **16 tests, all passing.**

### Test Coverage

| Test Class | Tests | What Is Verified |
|------------|-------|------------------|
| **TestRANSimulator** | 4 | Cell/UE initialization from config, simulation step execution, 3GPP PM counter snapshot generation, handover between cells |
| **TestTrafficModel** | 2 | Traffic model initialization, Ornstein-Uhlenbeck step (load stays in [0, 1.5] range) |
| **TestControllers** | 1 | All 4 xApps execute a control step, step counter increments |
| **TestTelemetry** | 2 | PM counter collection produces records with correct schema, feature engineering increases column count |
| **TestDatasetBuilder** | 1 | Multi-label conflict labeling produces `conflict` and `conflict_severity` columns |
| **TestMLPipeline** | 5 | Each of 5 models (LightGBM, XGBoost, RF, LogReg, MLP) trains and produces valid probability outputs |
| **TestInference** | 1 | ONNX Runtime is available and has CPU execution provider |

### Running Individual Test Classes

```bash
# Only simulator tests:
python -m pytest tests/test_basic.py::TestRANSimulator -v

# Only ML pipeline tests:
python -m pytest tests/test_basic.py::TestMLPipeline -v

# Only a single test:
python -m pytest tests/test_basic.py::TestMLPipeline::test_xgboost -v
```

### Running with Verbose Failure Output

```bash
python -m pytest tests/test_basic.py -v --tb=long
```

### Verifying the Full Pipeline End-to-End

After running the full pipeline (steps 1–4 from Quick Start), verify:

```bash
# Check telemetry was collected:
python -c "import pandas as pd; df=pd.read_parquet('data/telemetry.parquet'); print(f'Telemetry: {df.shape[0]} rows, {df.shape[1]} cols')"

# Check dataset was built with correct conflict rate:
python -c "import pandas as pd; df=pd.read_parquet('data/dataset.parquet'); print(f'Dataset: {df.shape[0]} rows, {df.shape[1]} cols, conflict rate: {df[\"conflict\"].mean():.2%}')"

# Check all models were trained:
python -c "import json; r=json.load(open('models/training_results.json')); [print(f'{k}: AUC={v[\"auc\"]:.3f}') for k,v in sorted(r.items(), key=lambda x: x[1].get('auc',0), reverse=True)]"

# Check ONNX benchmark:
python -c "import json; r=json.load(open('models/benchmark_results.json')); [print(f'{k}: mean={v[\"mean_ms\"]:.3f}ms p99={v[\"p99_ms\"]:.3f}ms') for k,v in sorted(r.items(), key=lambda x: x[1]['mean_ms'])]"

# Test the API server (start it first in another terminal):
python -c "import requests; r=requests.get('http://localhost:8000/health'); print(r.json())"
```

### Testing API Predictions

```bash
# Start the server:
python -m serving.api

# In another terminal, send a prediction request:
python -c "
import requests, json, pandas as pd
df = pd.read_parquet('data/dataset.parquet')
fc = json.load(open('models/feature_cols.json'))
cols = [c for c in fc if c in df.columns]
sample = {c: float(df[c].iloc[100]) for c in cols}
r = requests.post('http://localhost:8000/predict', json={'features': sample})
d = r.json()
print(f'Probability: {d[\"probability\"]:.4f}')
print(f'Conflict: {\"YES\" if d[\"conflict\"] else \"NO\"}')
print(f'Latency: {d[\"latency_ms\"]:.3f} ms')
print(f'Model: {d[\"model_type\"]}')
"
```

---

## API Reference

Base URL: `http://localhost:8000`

Interactive API documentation: `http://localhost:8000/docs` (Swagger UI)

### GET `/health`

Returns model loading status and available explainers.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "lgbm",
  "feature_count": 125,
  "explainers": {
    "lime": true,
    "shap": true,
    "counterfactual": true
  }
}
```

### GET `/metrics`

Returns p50/p95/p99 latency percentiles computed over the last 1000 requests.

```json
{
  "request_count": 42,
  "avg_latency_ms": 1.234,
  "p50_latency_ms": 1.100,
  "p95_latency_ms": 2.800,
  "p99_latency_ms": 3.500
}
```

### POST `/predict`

Single-sample prediction with full explainability.

**Request:**
```json
{
  "features": {
    "DRB.UEThpDl": 150000.0,
    "prb_util_dl": 0.652,
    "L1M.RS-SINR": 18.3,
    "...": "all 125 features"
  }
}
```

**Response:**
```json
{
  "probability": 0.374,
  "conflict": 0,
  "latency_ms": 1.86,
  "model_type": "lgbm",
  "top_explanations": {
    "lime": {
      "features": {"prb_util_dl_vs_avg": 0.12, "DRB.UEThpDl_roc": -0.05},
      "prediction": 0,
      "intercept": 0.32
    },
    "shap": {
      "shap_values": {"DRB.UEThpDl_roc": -0.08, "prb_util_dl_ma100": 0.03},
      "base_value": 0.32
    }
  },
  "counterfactual": {
    "found": true,
    "original_prediction": 0.374,
    "counterfactual_prediction": 0.523,
    "changes": {
      "prb_util_dl_vs_avg": {"from": 0.12, "to": 0.35, "delta": 0.23}
    }
  }
}
```

### POST `/predict/batch`

Batch prediction (no explainability, for throughput).

**Request:**
```json
{
  "instances": [
    {"features": {"...": "..."}},
    {"features": {"...": "..."}}
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {"probability": 0.12, "conflict": 0, "latency_ms": 0.04, "model_type": "lgbm"},
    {"probability": 0.87, "conflict": 1, "latency_ms": 0.03, "model_type": "lgbm"}
  ],
  "count": 2
}
```

---

## Dashboard

Launch command:
```bash
# Windows:
.\venv\Scripts\streamlit.exe run dashboard/app.py

# Linux/Mac:
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

**5 Pages:**

| Page | Shows |
|------|-------|
| Cell KPI Overview | Per-cell time series of PRB utilization, throughput, SINR, CQI, MCS, BLER with latest-value metric cards |
| xApp Conflict Analysis | Conflict label distribution table, severity histogram, rolling conflict rate with configurable window |
| Model Comparison | AUC bar chart, full metrics table (AUC/F1/gap), inference benchmark chart from benchmark_results.json |
| Live Inference | Row selection slider, feature display, click-to-predict with ConflictPredictor, shows probability/conflict/latency |
| Benchmark Results | Per-format expandable cards showing mean/p50/p95/p99, sub-ms pass/fail summary |

---

## Configuration Reference

All parameters are in `config/network.yaml`. Key sections:

### Cell Configuration
```yaml
ran:
  cells:
    - cell_id: cell-01
      pci: 1
      band: n78
      bandwidth_mhz: 100
      total_prbs: 273
      scs_khz: 30
      tx_power_dbm: 43
      antenna_ports: 64
      max_layers: 4
      max_ue_capacity: 200
```

### UE Profiles
```yaml
ue_profiles:
  - name: eMBB_premium
    count: 60
    qfi: 9
    min_throughput_kbps: 50000
    max_latency_ms: 10
    mobility: stationary
```

### Experiment Parameters
```yaml
experiment:
  duration_s: 30          # Simulation length in seconds
  warmup_s: 1             # Warmup period excluded from dataset
  doppler_hz: 5           # Doppler shift frequency

telemetry:
  poll_interval_ms: 10    # 100 Hz collection rate
  window_size: 100        # Rolling window for feature engineering
```

### ML Configuration
```yaml
ml:
  seed: 42
  test_size: 0.2
  conflict_kpi_threshold: 0.05
  lgbm:
    n_estimators: 500
    learning_rate: 0.05
    max_depth: 6
  xgb:
    n_estimators: 300
    max_depth: 6
  mlp:
    hidden_layers: [128, 64, 32]
    dropout: 0.15
    epochs: 50
    lr: 0.0005
```

**To generate a larger dataset:** increase `experiment.duration_s` to 300 (5-minute run, ~15,000 rows).

---

## Public Repository Safety

### Safe for Public GitHub

| Item | Reason |
|------|--------|
| All `.py` source files | Pure simulation and ML code, no credentials or secrets |
| `config/network.yaml` | Synthetic 3GPP radio parameters only |
| `requirements.txt` | Package dependency list only |
| `tests/test_basic.py` | Unit tests only |
| Shell scripts, Makefile | Automation commands only |
| `README.md`, `.gitignore` | Documentation and config |

### Not Committed (via .gitignore)

| Item | Reason |
|------|--------|
| `data/*.parquet`, `data/*.csv` | Generated synthetic data — anyone can regenerate in 30 seconds |
| `models/*.joblib`, `*.onnx`, `*.pt`, `*.pth`, `*.npz` | Binary model files — bloat git history, regeneratable |
| `logs/telemetry.jsonl` | Large generated log file |
| `venv/` | Python virtual environment |

### Security Notes

- **No secrets anywhere** — no API keys, tokens, passwords, or real network addresses in any file
- **No real data** — all telemetry is synthetically generated by the simulator
- **No PII** — no personal information of any kind
- **Dependencies** — all open-source, well-maintained packages from PyPI and PyTorch official index
- **ONNX models** — contain only the MLP neural network architecture and weights trained on synthetic data

---

## Dependencies

CPU-only PyTorch is used to minimize download size (~200 MB vs ~2 GB for CUDA wheels).

| Package | Version Constraint | Purpose |
|---------|-------------------|---------|
| numpy | >=1.24, <2.0 | Array operations |
| pandas | >=1.5, <2.1 | DataFrames, parquet I/O |
| scikit-learn | >=1.2 | RF, LogReg, metrics, preprocessing |
| lightgbm | >=3.3 | Gradient boosting |
| xgboost | (latest) | Gradient boosting (best AUC) |
| torch | >=2.0, <2.4 (CPU) | MLP training, TorchScript export |
| onnx | >=1.14 | ONNX model format |
| onnxruntime | >=1.15 | Sub-ms inference |
| lime | >=0.2 | LIME explanations |
| shap | >=0.42 | SHAP explanations |
| fastapi | >=0.100 | REST API server |
| uvicorn | >=0.23 | ASGI server |
| streamlit | >=1.25 | Dashboard |
| pyarrow | >=12.0 | Parquet I/O |
| pyyaml | >=6.0 | YAML config loading |
| pytest | >=7.4 | Testing |
| requests | >=2.31 | HTTP client (for API testing) |
| psutil | >=5.9 | System metrics |

---

## Troubleshooting

### Streamlit won't start on Windows

Use `.\venv\Scripts\streamlit.exe run dashboard/app.py` instead of `python -m streamlit run dashboard/app.py`. The latter may invoke the system Python instead of the venv.

### Port already in use

```bash
# Kill process on port 8000 (API):
# Windows:
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force

# Linux:
kill $(lsof -t -i:8000)
```

### "No trained model found" error

Run the full pipeline first: `python -m telemetry.collector`, then `python -m dataset.builder`, then `python -m ml.train`, then `python -m ml.optimize`.

### Test failures after code changes

Run `python -m pytest tests/test_basic.py -v --tb=long` for detailed failure output. Most test failures indicate a missing model file or dataset — regenerate by running the pipeline.

### XGBoost not found

Install separately: `pip install xgboost`.

---

## Roadmap

### Phase 1: Stronger ML
- Longer simulations for 10k+ row datasets
- Hyperparameter tuning with Optuna
- Sequence models (LSTM / Transformer) for temporal conflict patterns
- Class-imbalance handling with SMOTE or focal loss
- Cross-cell joint prediction

### Phase 2: Real O-RAN Integration
- Real E2 interface via SD-RAN / ONOS-RIC
- OpenAirInterface (OAI) or srsRAN as gNB
- O-RAN SC near-RT RIC container
- Docker + Kubernetes xApp deployment

### Phase 3: Production Hardening
- Model versioning with MLflow
- Feature drift detection
- Prometheus + Grafana observability
- CI/CD with GitHub Actions
- Model card documentation

### Phase 4: Research Extensions
- Graph neural networks on cell topology
- Federated learning across RIC instances
- Reinforcement learning conflict-aware xApps
- Integration with real 5G KPI datasets

---

## Citation and Attribution

This project is a research simulation. Relevant standards:
- 3GPP TS 28.552 — Management and orchestration; 5G performance measurements
- 3GPP TR 38.901 — Study on channel model for frequencies from 0.5 to 100 GHz
- 3GPP TS 38.104 — NR; Base Station (BS) radio transmission and reception
- 3GPP TS 38.214 — NR; Physical layer procedures for data
- 3GPP TS 38.331 — NR; Radio Resource Control (RRC)
- 3GPP TS 28.530 — Management and orchestration; Concepts, use cases and requirements
- O-RAN.WG3.E2SM-KPM-v03.00 — E2 Service Model, KPM
- O-RAN.WG3.E2SM-RC-v01.00 — E2 Service Model, RAN Control
- O-RAN.WG2.AIML-v01.03 — AI/ML workflow description and requirements

---

*Built as a learning and research platform. Not affiliated with O-RAN Alliance, 3GPP, or any telecom vendor.*