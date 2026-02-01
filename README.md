# paraglider-sites
Connect paragliding flights and weather data

## Background

More about the methodology in the [Background](./BACKGROUND.md)

## Features

This project uses **FastAPI** with machine learning capabilities:
- Random Forest models for classification and regression
- Bayesian Networks for probabilistic inference
- OpenAPI/Swagger documentation at `/docs`
- Modern async API design


## Quick start

1. **Clone the repository**
```bash
cd paraglider-sites
```

2. **Create virtual environment**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**
```powershell
pip install -r requirements.txt
```

4. **Run the application**

```powershell
uvicorn app:app --reload --host localhost --port 3979
```

The app will be available at `http://localhost:3979` with API docs at `http://localhost:3979/docs`


## Database

Initialize the database (creates `glider_sites_app.db` in the repository root):
```powershell
python -m glider_sites_app.db
```

To reset the database:
```powershell
rm glider_sites_app.db
python -m glider_sites_app.db
```

## Tools

### dhv_loader

Load flight data (sample used for Rammi NW):
```powershell
python -m glider_sites_app.tools.flights.dhv_loader
```

### openmeteo_loader

Load weather data (sample used for Rammi):
```powershell
python -m glider_sites_app.tools.weather.openmeteo_loader
```


## Analysis

### Random forest

Run classification and regression models:
```powershell
python -m glider_sites_app.analysis.random_forest
```

### Bayes network

Run probabilistic inference:
```powershell
python -m glider_sites_app.analysis.bayes_network
```