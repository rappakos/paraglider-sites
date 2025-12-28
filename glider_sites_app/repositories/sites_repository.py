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
        f.last_flight_date, f.flight_count,
        w.last_weather_time
    FROM sites s
    LEFT JOIN (
        SELECT site_name, MAX(FlightDate) as last_flight_date, COUNT(DISTINCT IDFlight) as flight_count
        FROM dhv_flights
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