# paraglider-sites
Connect paragliding flights and weather data

## 🚀 FastAPI Version Available!

This project has been migrated to **FastAPI** with added machine learning capabilities:
- Random Forest models for classification and regression
- Bayesian Networks for probabilistic inference
- OpenAPI/Swagger documentation
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

4. **Configure environment**
Create a `.env` file with:
```
PORT=3979
```

5. **Running the applicaiton**

```powershell
uvicorn app:app --reload --host localhost --port 3979
```


## DB

Execute as module to init: `py -m glider_sites_app.db`

Complete reset by `rm glider_sites_app.db`

## Tools

### dhv_loader

Execute as a module `python -m glider_sites_app.tools.flights.dhv_loader`, sample used for Rammi NW

### openmeteo_loader

Execute as a module `py -m glider_sites_app.tools.weather.openmeteo_loader`, sample used for Rammi


## Analysis

### Random forest

Run with `py -m glider_sites_app.analysis.random_forest`

### Bayes network

Run with `py -m glider_sites_app.analysis.bayes_network`