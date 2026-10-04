#!/usr/bin/env bash
# Reproduce every artefact from the raw CSV: data split, EDA, experiments, model, demo cases.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/prepare_data.py
python scripts/run_eda.py
python scripts/train_models.py
python scripts/select_demo_cases.py
python scripts/make_architecture_diagram.py
echo "Pipeline complete. Start the app with: uvicorn backend.main:app --port 8000"
