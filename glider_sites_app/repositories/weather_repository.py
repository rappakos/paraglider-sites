# repositories/weather_repository.py
from glider_sites_app.tools.weather.constants import HOURLY_PARAMS_ARCHIVE,HOURLY_PARAMS
from ..db import DB_NAME
import aiosqlite
from pandas import DataFrame


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
