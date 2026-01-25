#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
python -m glider_sites_app.jobs.sync_weather_forecast >> logs/forecast_sync.log 2>&1