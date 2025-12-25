# repositories/flights_repository.py
from sqlalchemy import create_engine, text
from ..db import DB_NAME
import aiosqlite
from pandas import read_sql_query, DataFrame

    
async def get_flights(site_name:str):
    engine = create_engine(f'sqlite:///{DB_NAME}')
    with engine.connect() as db:
        param = {'site_name':site_name}
        df  = read_sql_query(text(f"""
                        SELECT 
                            [IDFlight], [FlightDate], [FlightStartTime], [FKPilot], [Glider], [GliderClassification], [FlightDuration], [BestTaskPoints], [BestTaskType]
                        FROM dhv_flights f 
                        WHERE f.site_name=:site_name
                    """), db, params=param)
        return df

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