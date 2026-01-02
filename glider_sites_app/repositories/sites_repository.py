# repositories/sites_repository.py
from sqlalchemy import create_engine, text
from ..db import DB_NAME
from pandas import read_sql_query, DataFrame


async def get_stats() -> DataFrame:  
    engine = create_engine(f'sqlite:///{DB_NAME}?charset=utf8')
    try:
        with engine.connect() as db:
            param = {}
            df  = read_sql_query(text(f"""
    SELECT 
        s.site_name, s.dhv_site_id, s.geo_latitude, s.geo_longitude, s.elevation, s.main_direction,
        f.last_flight_date, f.flight_count, f.total_flight_days,
        w.last_weather_time
    FROM sites s
    LEFT JOIN (
        SELECT site_name, MAX(FlightDate) as last_flight_date, COUNT(DISTINCT IDFlight) as flight_count
            , COUNT(DISTINCT DATE(FlightDate)) as total_flight_days
        FROM dhv_flights
            --WHERE FlightDate > '2022-01-01'
        GROUP BY site_name
    ) f ON f.site_name = s.site_name
    LEFT JOIN (
        SELECT site_name, MAX(time) as last_weather_time
        FROM weather_data
        GROUP BY site_name
    ) w ON w.site_name = s.site_name                               
                    """), db, params=param)
            return df
    finally:
        engine.dispose()

async def get_main_direction(site_name: str) -> int:
    engine = create_engine(f'sqlite:///{DB_NAME}?charset=utf8')
    try:
        with engine.connect() as db:
            param = {'site_name': site_name}
            result = db.execute(
                text("SELECT main_direction FROM sites WHERE site_name = :site_name"),
                param
            ).fetchone()
            if result:
                return result[0]
            else:
                raise ValueError(f"Site {site_name} not found")
    finally:
        engine.dispose()