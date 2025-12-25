# repositories/weather_repository.py
from glider_sites_app.tools.weather.constants import HOURLY_PARAMS_ARCHIVE,HOURLY_PARAMS
from ..db import DB_NAME
import aiosqlite
from pandas import DataFrame, read_sql_query
from sqlalchemy import create_engine, text

async def load_weather_data(site_name: str) -> DataFrame:
    """Load weather data for a site"""
    engine = create_engine(f'sqlite:///{DB_NAME}')
    with engine.connect() as db:
        df = read_sql_query(
            text("""
                SELECT 
                    site_name,
                    DATE(time) as date,
                    AVG(wind_speed_10m) as avg_wind_speed,
                    AVG(wind_direction_10m) as avg_wind_direction,
                    SUM(sunshine_duration) as total_sunshine,
                    SUM(precipitation) as total_precipitation
                FROM weather_data
                WHERE site_name = :site_name
                GROUP BY site_name, DATE(time)
            """),
            db,
            params={'site_name': site_name}
        )
    return df


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
