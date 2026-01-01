# repositories/weather_repository.py
from glider_sites_app.tools.weather.constants import HOURLY_PARAMS_ARCHIVE,HOURLY_PARAMS
from ..db import DB_NAME
import aiosqlite
from pandas import DataFrame, read_sql_query
from sqlalchemy import create_engine, text

async def load_weather_data(site_name: str) -> DataFrame:
    """Load weather data for a site"""
    engine = create_engine(f'sqlite:///{DB_NAME}')
    try:
        with engine.connect() as db:
            df = read_sql_query(
                text("""SELECT 
                        site_name,
                        DATE(time) as date,
                        time,
                        wind_speed_10m,
                        wind_direction_10m,
                        wind_gusts_10m,
                        sunshine_duration,
                        precipitation,
                        cloud_cover_low,
                        -- physics
                        boundary_layer_height, -- pre 2022 ?
                        temperature_2m, -- post 2022 ?
                        dew_point_2m,
                        temperature_850hPa,
                        wind_speed_850hPa
                    FROM weather_data
                    WHERE site_name = :site_name
                        -- and DATE(time) >= '2022-01-01'  """),
                db,
                params={'site_name': site_name}
            )
        return df
    finally:
        engine.dispose()

async def fix_temperature_850hPa_pre2020(df_weather: DataFrame):
    """Fix temperature_850hPa for pre-2020 data using some external source (placeholder)"""
    
    # Prepare data: (temperature, site_name, date)
    updates = df_weather[['temperature_850hPa', 'site_name', 'date']].values.tolist()

    sql = f"""UPDATE weather_data
            SET temperature_850hPa = ?
            WHERE (temperature_850hPa IS NULL OR temperature_850hPa = 0)
              AND site_name = ?
              AND DATE(time) = ?"""

    async with aiosqlite.connect(DB_NAME) as db:
        rows = await db.executemany(sql, updates)
        print(f"Updated {rows.rowcount} rows with fixed temperature_850hPa for {df_weather['site_name'].iloc[0]}")
        await db.commit()


async def save_weather_data(df_weather: DataFrame):
    columns_to_save = ["site_name","time"]

    if df_weather['time'].max().year < 2022:
        columns_to_save += HOURLY_PARAMS_ARCHIVE
    else: 
        columns_to_save += HOURLY_PARAMS

    df_to_save = df_weather[columns_to_save].copy()
    df_to_save['time'] = df_to_save['time'].dt.strftime('%Y-%m-%d %H:%M:%S')        
    # special case
    df_to_save[['boundary_layer_height']] = df_to_save[['boundary_layer_height']].fillna(0).infer_objects(copy=False)


    sql =f"""INSERT OR IGNORE INTO weather_data
            ({', '.join(columns_to_save)})
            VALUES ({','.join(['?' for _ in columns_to_save])})"""

    async with aiosqlite.connect(DB_NAME) as db:
        await db.executemany(sql, df_to_save.values.tolist())
        await db.commit()
