#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

if [ -z "$1" ]; then
    echo "Usage: $0 <site_name>"
    echo "Example: $0 Dielmissen"
    exit 1
fi

SITE="$1"
echo "=== Retraining models for: $SITE ==="

echo "[1/2] Training ensemble (RF + SVR)..."
python -c "
import asyncio
from glider_sites_app.analysis.ensemble import train_ensemble
asyncio.run(train_ensemble('$SITE', save=True))
"

echo "[2/2] Training Bayesian Network..."
python -c "
import asyncio
from glider_sites_app.analysis.bayes_network import flight_predictor
asyncio.run(flight_predictor('$SITE', save_model=True))
"

echo "Done. Models saved to glider_sites_app/analysis/models/"
