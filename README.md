# AdaptCast

Adaptive time series forecasting pipeline with concept drift detection, automated retraining, and a live monitoring dashboard.

---

## Overview

AdaptCast trains a LightGBM baseline on the ETTh1 electricity transformer dataset, streams new observations row-by-row, and uses three statistical drift detectors (ADWIN, Page-Hinkley, KSWIN) to decide when the data distribution has shifted. When drift is detected, an online learner (river's `HoeffdingAdaptiveTreeRegressor`) updates the model incrementally, or a full LightGBM retrain is triggered if the buffer is large enough. The new model is promoted to champion only if it achieves at least 5 % RMSE improvement. All experiments are tracked in a local MLflow server. A FastAPI service exposes predictions, and a Streamlit + Plotly dashboard gives live visibility into drift events, model versions, and forecast accuracy.

---

## Architecture

```
ETTh1.csv
    │
    ▼
src/data/preprocess.py   ←── lag features (1,24,168), rolling stats, train/val/test split
    │
    ▼
src/models/baseline.py   ←── LightGBM train + time-series CV + MLflow logging
    │
    ▼  champion alias
src/models/registry.py   ←── MLflow model registry  ──────────────────────────┐
    │                                                                           │
    ▼  row-by-row stream                                                        │
src/data/stream.py                                                              │
    │                                                                           │
    ▼  residuals                                                                │
src/drift/monitor.py     ←── ADWIN │ Page-Hinkley │ KSWIN                     │
    │  DriftEvent                                                               │
    ▼                                                                           │
src/drift/retrainer.py   ←── online update (river) or full retrain ────────────┘
    │
    ▼
src/serving/app.py       ←── FastAPI  (uvicorn, port 8000)
    │
    ▼
src/dashboard/app.py     ←── Streamlit + Plotly (port 8501)
```

---

## Quick Start

**Prerequisites:** Python 3.11+, pip.

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Download and preprocess data
python -m src.data.download
python -m src.data.preprocess

# 3. Train baseline model (logs to MLflow)
python -m src.models.baseline

# 4. Start MLflow UI (separate terminal)
mlflow ui --port 5000
# → http://localhost:5000

# 5. Start prediction API (separate terminal)
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs

# 6. Start dashboard (separate terminal)
streamlit run src/dashboard/app.py
# → http://localhost:8501
```

Or use the Makefile shortcuts:

```bash
make install
make data
make train
make mlflow   # terminal 1
make serve    # terminal 2
make dashboard  # terminal 3
```

---

## API Reference

Base URL: `http://localhost:8000`

| Method | Path            | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | `/health`       | Liveness check                       |
| POST   | `/predict`      | Single-step forecast                 |
| GET    | `/model/info`   | Current champion model metadata      |
| GET    | `/drift/status` | Latest drift detector readings       |
| POST   | `/drift/reset`  | Manually reset drift detectors       |

Interactive docs: `http://localhost:8000/docs`

---

## Running Tests

```bash
pytest -v
# Expected: 25 tests pass
```

---

## Project Structure

```
AdaptCast/
├── configs/
│   ├── drift.yaml          # detector thresholds
│   ├── model.yaml          # LightGBM hyperparameters
│   └── serving.yaml        # API port, MLflow URI, dashboard refresh
├── data/
│   ├── raw/ETTh1.csv
│   └── processed/          # train/val/test parquet files
├── src/
│   ├── data/               # download, preprocess, stream
│   ├── drift/              # detectors, monitor, retrainer
│   ├── models/             # baseline (LightGBM), online (river), registry
│   ├── serving/            # FastAPI app, routes, schemas
│   └── dashboard/          # Streamlit app + components
├── tests/                  # 25 pytest tests
├── notebooks/              # EDA, feature engineering, drift simulation
├── AGENT/                  # project docs (phases, decisions, concepts)
├── pyproject.toml
├── Makefile
└── README.md
```
