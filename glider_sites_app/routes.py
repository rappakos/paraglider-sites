import pathlib
from fastapi import APIRouter, HTTPException, Request
from typing import List

from .schemas import (
    SiteStats, SiteDetail, FlightRecord,
    RefreshFlightsRequest, RefreshWeatherRequest,
    DataLoaderResponse
)
from . import db
from .tools.flights.dhv_loader import refresh_flight_list
from .openmeteo_loader import refresh_weather_data

PROJECT_ROOT = pathlib.Path(__file__).parent


# Create API router
api_router = APIRouter(prefix="/api", tags=["sites"])


async def get_site_data(site_name: str):
    """Helper function to get site data"""
    all_stats = await db.get_stats()
    current = all_stats[all_stats['site_name'] == site_name]
    if current.empty:
        raise HTTPException(status_code=404, detail=f"Site {site_name} not found")
    return current.to_dict('records')


@api_router.get("/", response_model=List[SiteStats])
async def get_all_sites():
    """Get all paraglider sites with statistics"""
    stats = await db.get_stats()
    return stats.to_dict('records')


@api_router.get("/{site_name}", response_model=SiteDetail)
async def get_site_details(site_name: str):
    """Get detailed information for a specific site"""
    data = await get_site_data(site_name)
    flights = await db.get_flights(site_name)
    
    return {
        'name': site_name,
        'data': data,
        'flights': flights.sort_values(by='FlightStartTime', ascending=False).head(10).to_dict('records')
    }


def setup_routes(app):
    """Setup all routes for the FastAPI app"""
    
    app.include_router(api_router)


