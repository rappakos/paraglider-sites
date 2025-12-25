# schemas.py
from pydantic import BaseModel
from typing import Optional


class SiteBase(BaseModel):
    site_name: str
    dhv_site_id: Optional[int] = None
    geo_latitude: float
    geo_longitude: float
    elevation: Optional[float] = None


class SiteStats(SiteBase):
    last_flight_date: Optional[str] = None
    flight_count: int


class FlightRecord(BaseModel):
    IDFlight: int
    FlightDate: str
    FlightStartTime: Optional[str] = None
    FKPilot: Optional[int] = None 
    Glider: Optional[str] = None
    GliderClassification: Optional[str] = None
    FlightDuration: Optional[float] = None
    BestTaskPoints: Optional[float] = None
    BestTaskType: Optional[str] = None




