import os
import pathlib
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
from .schemas import (
    SiteStats
)
from .services.site_service import get_site_data, get_all_sites, get_forecast_data

PROJECT_ROOT = pathlib.Path(__file__).parent
# Setup Jinja2 templates
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

prefix = os.environ.get('GLIDER_SITES_APP_PREFIX', '')

# Create API router
api_router = APIRouter(prefix=prefix+"/api", tags=["sites"])
# Create page router for HTML views
page_router = APIRouter(prefix=prefix,tags=["pages"])

@api_router.get("/", response_model=List[SiteStats])
async def api_get_all_sites():
    """Get all paraglider sites with statistics"""
    stats = await get_all_sites()
    return stats


@api_router.get("/{site_name}", response_model=SiteStats)
async def api_get_site_details(site_name: str):
    """Get detailed information for a specific site"""
    data = await get_site_data(site_name)
    if not data:
        raise HTTPException(status_code=404, detail="Site not found")    
    return data

@api_router.get("/{site_name}/forecast")
async def api_get_site_forecast(
    site_name: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD), defaults to today")
):
    """Get forecast data for a specific site (5-day forecast)"""
    # Default to current day if not provided
    if not start_date:
        start_date = datetime.now().strftime('%Y-%m-%d')
    
    # Calculate end_date as start_date + 5 days
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = (start_dt + timedelta(days=5)).strftime('%Y-%m-%d')
    
    data = await get_forecast_data(site_name, start_date, end_date)
    if not data:
        raise HTTPException(status_code=404, detail="Site not found")    

    return data


# HTML page routes
@page_router.get("/")
async def index(request: Request):
    """Home page with all sites"""
    data = await get_all_sites()
    return templates.TemplateResponse("index.html", {"request": request, "data": data})


@page_router.get("/forecast")
async def forecast_page(
    request: Request,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD), defaults to today")
):
    """Forecast page (5-day forecast)"""
    # Default to current day if not provided
    if not start_date:
        start_date = datetime.now().strftime('%Y-%m-%d')
    
    # Calculate end_date as start_date + 5 days
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = (start_dt + timedelta(days=5)).strftime('%Y-%m-%d')
    
    site_data = await get_all_sites()

    for site in site_data:
        # Get full site data which includes has_model
        site_details = await get_site_data(site['site_name'])
        if site_details and site_details.get('has_model'):
            forecast = await get_forecast_data(site['site_name'], start_date, end_date)
            site['has_model'] = True
            site['forecast'] = forecast
        else:
            site['has_model'] = False

    site_data.sort(key=lambda x: x.get('main_direction', ''), reverse=True)            
    return templates.TemplateResponse("forecast.html", {"request": request,"start_date": start_date,  "data": site_data})

@page_router.get("/site/{site_name}")
async def site_details(request: Request, site_name: str):
    """Site details page"""
    data = await get_site_data(site_name)
    if not data:
        raise HTTPException(status_code=404, detail="Site not found")
    # Add flights data if needed
    return templates.TemplateResponse("site.html", {
        "request": request,
        "name": site_name,
        "data": [data]
    })




def setup_routes(app):
    """Setup all routes for the FastAPI app"""
    app.mount("/static", StaticFiles(directory="glider_sites_app/static"), name="static")
    app.include_router(api_router)
    app.include_router(page_router)

