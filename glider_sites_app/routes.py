import pathlib
from fastapi import APIRouter, HTTPException, Request
from typing import List
from .schemas import (
    SiteStats
)
from .services.site_service import get_site_data, get_all_sites

PROJECT_ROOT = pathlib.Path(__file__).parent


# Create API router
api_router = APIRouter(prefix="/api", tags=["sites"])


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


def setup_routes(app):
    """Setup all routes for the FastAPI app"""
    
    app.include_router(api_router)


