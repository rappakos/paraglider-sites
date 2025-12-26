# repositories/flights_repository.py
from ..db import DB_NAME
import aiosqlite
from pandas import read_sql_query, DataFrame
from sqlalchemy import create_engine, text

async def load_flight_counts(site_name: str) -> DataFrame:
    """Load daily flight counts for a site"""
    engine = create_engine(f'sqlite:///{DB_NAME}')
    try:
        with engine.connect() as db:
            df = read_sql_query(
                text("""
                    SELECT 
                        site_name,
                        FlightDate as date,
                        COUNT(*) as flight_count,
                        MAX(BestTaskPoints) as max_daily_score
                        --get the type of the flight with MAX(BestTaskPoints)
                    FROM dhv_flights
                    WHERE site_name = :site_name
                    GROUP BY site_name, FlightDate
                """),
                db,
                params={'site_name': site_name}
            )
        return df
    finally:
        engine.dispose()


async def get_flights(site_name:str):
    engine = create_engine(f'sqlite:///{DB_NAME}')
    try:
        with engine.connect() as db:
            param = {'site_name':site_name}
            df  = read_sql_query(text(f"""
                            SELECT 
                                [IDFlight], [FlightDate], [FlightStartTime], [FKPilot], [Glider], [GliderClassification], [FlightDuration], [BestTaskPoints], [BestTaskType]
                            FROM dhv_flights f 
                            WHERE f.site_name=:site_name
                        """), db, params=param)
            return df
    finally:
        engine.dispose()

async def save_dhv_flights(df_flights:DataFrame):
    columns_to_save = ['IDFlight', 'FlightDate', 'FlightStartTime', 'FKPilot', 
                       'Glider', 'GliderClassification', 'FlightDuration', 
                       'BestTaskPoints', 'BestTaskType', 'site_name']

    async with aiosqlite.connect(DB_NAME) as db:
        await db.executemany("""
            INSERT OR IGNORE INTO dhv_flights 
            (IDFlight, 
            FlightDate, 
            FlightStartTime, 
            FKPilot, 
            Glider,
            GliderClassification,
            FlightDuration, 
            BestTaskPoints, 
            BestTaskType, 
            site_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, df_flights[columns_to_save].values.tolist())
        await db.commit()