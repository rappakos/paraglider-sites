# paraglider-sites
Connect paragliding flights and weather data

## 🚀 FastAPI Version Available!

This project has been migrated to **FastAPI** with added machine learning capabilities:
- Random Forest models for classification and regression
- Bayesian Networks for probabilistic inference
- OpenAPI/Swagger documentation
- Modern async API design

**See [README_FASTAPI.md](README_FASTAPI.md) for the new FastAPI version.**

---

## DB

Execute as module to init: `py -m glider_sites_app.db`

Complete reset by `rm glider_sites_app.db`

## Tools

### dhv_loader

Execute as a module `python -m glider_sites_app.tools.flights.dhv_loader`, sample used for Rammi NW

### openmeteo_loader

Execute as a module `py -m glider_sites_app.tools.weather.openmeteo_loader`, sample used for Rammi