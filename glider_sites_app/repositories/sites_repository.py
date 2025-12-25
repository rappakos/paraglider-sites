# repositories/sites_repository.py
from sqlalchemy import create_engine, text
from ..db import DB_NAME
from pandas import read_sql_query, DataFrame


async def get_stats():  
    engine = create_engine(f'sqlite:///{DB_NAME}?charset=utf8')
    with engine.connect() as db:
        param = {}
        df  = read_sql_query(text(f"""
                        SELECT 
                            s.site_name, s.dhv_site_id, s.geo_latitude, s.geo_longitude, s.elevation, max(f.FlightDate) [last_flight_date], count(f.IDFlight) [flight_count]
                        FROM sites s
                        left join dhv_flights f on f.site_name=s.site_name
                        GROUP BY s.site_name, s.dhv_site_id, s.geo_latitude, s.geo_longitude, s.elevation                                  
                    """), db, params=param)
        return df