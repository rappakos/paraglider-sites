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
                        MAX(BestTaskPoints) as max_daily_score,
                        (SELECT FKPilot 
                                FROM dhv_flights f2 
                                WHERE f2.site_name = f.site_name 
                                AND f2.FlightDate = f.FlightDate 
                                ORDER BY BestTaskPoints DESC 
                         LIMIT 1) as best_pilot_id,
                         (SELECT AVG(FlightDuration) FROM 
                            (SELECT FlightDuration
                                FROM dhv_flights f3
                                WHERE f3.site_name = f.site_name 
                                AND f3.FlightDate = f.FlightDate 
                                ORDER BY FlightDuration DESC 
                                -- TODO decide whether we want to aggregate by pilot
                                LIMIT 3))/60.0 as avg_flight_duration  -- in minutes                  
                    FROM dhv_flights f
                    WHERE site_name = :site_name
                    GROUP BY site_name, FlightDate
                """),
                db,
                params={'site_name': site_name}
            )
        return df
    finally:
        engine.dispose()

async def get_xcontest_flight_counts(site_name: str) -> DataFrame:
    """Load daily xcontest flight counts for a site"""
    engine = create_engine(f'sqlite:///{DB_NAME}')
    try:
        with engine.connect() as db:
            df = read_sql_query(
                text("""
                    SELECT 
                        site_name,
                        flight_date as date,
                        COUNT(*) as flight_count,
                        MAX(points) as max_daily_score,
                        (SELECT pilot_id 
                         FROM xcontest_flights f2 
                         WHERE f2.site_name = f.site_name 
                           AND f2.flight_date = f.flight_date 
                         ORDER BY points DESC 
                         LIMIT 1) as best_pilot_id
                    FROM xcontest_flights f
                    WHERE site_name = :site_name
                        AND NOT EXISTS (SELECT 1 FROM dhv_flights d 
                                        WHERE d.site_name = f.site_name and d.FlightDate = f.flight_date)
                     /* Rammi NW vs Rammi SW */
                     /* and f.flight_date not in (
                                '2025-12-30' -- NW
                        , '2025-08-09' -- ? NW but light W/SW ...
                        , '2025-05-20' -- NW
                        , '2025-04-25' -- actually NE
                        , '2025-03-18', '2024-10-22', '2024-07-11', '2024-04-16', '2024-03-25'
                        , '2023-12-02', '2023-07-09'
                        , '2023-11-30', '2023-05-01', '2023-04-30', '2023-03-11', '2022-08-20','2022-08-05','2022-07-29'
                        , '2022-06-21' -- actually NE
                        , '2022-07-05'
                        , '2021-12-02', '2021-10-07'
                        , '2021-09-04' -- actually NE
                        , '2021-07-22', '2021-07-15', '2021-04-01', '2021-01-15', '2021-01-01', '2020-11-08'
                        , '2020-11-05' -- actually SW !
                        , '2020-06-20', '2020-06-19', '2020-06-07', '2020-05-10', '2020-02-07'
                        , '2020-05-28', '2019-10-29', '2020-05-20', '2020-05-12', '2020-05-08'
                        , '2019-09-14', '2019-03-21', '2018-08-26', '2018-07-20', '2018-07-15'
                        , '2018-04-17', '2018-03-25','2018-03-21'                     
                     ) */

                    GROUP BY site_name, flight_date
                """),
                db,
                params={'site_name': site_name}
            )
        return df
    finally:
        engine.dispose()


async def pilot_stats() -> DataFrame:
    """Load pilot statistics"""
    engine = create_engine(f'sqlite:///{DB_NAME}')
    try:
        with engine.connect() as db:
            df  = read_sql_query(text(f"""
                            SELECT 
                                FKPilot,
                                MAX(BestTaskPoints) as best_score
                            FROM dhv_flights
                            GROUP BY FKPilot
                            ORDER BY best_score DESC
                        """), db)
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

async def get_last_xcontest_flight_date(site_name: str):
    engine = create_engine(f'sqlite:///{DB_NAME}')
    try:
        with engine.connect() as db:
            param = {'site_name': site_name}
            result = db.execute(
                text("SELECT MAX(flight_date) FROM xcontest_flights WHERE site_name = :site_name"),
                param
            ).fetchone()
            if result and result[0]:
                return result[0]
            else:
                return None
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

async def save_xcontest_flights(df_flights:DataFrame):
    # Rename DataFrame columns to match database schema
    df_flights = df_flights.rename(columns={'date': 'flight_date', 'time': 'flight_time'})
    
    columns_to_save = ['flight_id', 'flight_date', 'flight_time', 'pilot_id', 
                       'pilot_name', 'takeoff_name', 'flight_type', 
                       'distance_km', 'points', 'glider_category', 'glider_name', 
                       'flight_details', 'site_name']

    async with aiosqlite.connect(DB_NAME) as db:
        await db.executemany("""
            INSERT OR IGNORE INTO xcontest_flights 
            (flight_id, 
            flight_date, 
            flight_time, 
            pilot_id, 
            pilot_name,
            takeoff_name,
            flight_type,
            distance_km, 
            points, 
            glider_category,
            glider_name,
            flight_details,
            site_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, df_flights[columns_to_save].values.tolist())
        await db.commit()