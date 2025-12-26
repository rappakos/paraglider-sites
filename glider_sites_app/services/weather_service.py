import logging
import numpy as np
from datetime import datetime, timedelta
from pandas import DataFrame, to_datetime

from glider_sites_app.repositories.sites_repository import get_stats
from glider_sites_app.repositories.weather_repository import load_weather_data, save_weather_data
from glider_sites_app.tools.weather.constants import MIN_DATE
from glider_sites_app.tools.weather.openmeteo_loader import refresh_weather_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_start_date(max_date:str) -> str:
    if not max_date:
        return MIN_DATE
    max_date_obj = datetime.strptime(max_date[:10],'%Y-%m-%d')
    start_date_obj = min(
        max_date_obj + timedelta(days=1), 
        datetime.now() + timedelta(days=-5) 
    )
    return start_date_obj.strftime('%Y-%m-%d')

def get_end_date(start_date:str) -> str:
    end_date_obj = min(
            datetime(datetime.strptime(start_date,'%Y-%m-%d').year,12,31),
            datetime.now() + timedelta(days=-5)
    )
    return end_date_obj.strftime('%Y-%m-%d')

def get_blh(row) -> float:
    if row['date'] < '2022-01-01' or row['boundary_layer_height'] > 100:
        return row['boundary_layer_height']
    else:
        spread = (row['temperature_2m'] - row['dew_point_2m'])
        if spread < 0:
            return 0.
        # do not add elevation!
        return spread * 125.


def aggregate_weather(raw_weather_df: DataFrame, main_direction: int) -> DataFrame:
    """
    Aggregate hourly weather data to daily metrics
    
    Args:
        raw_weather_df: DataFrame with hourly weather data including time, wind_speed_10m, 
                       wind_direction_10m, wind_gusts_10m, sunshine_duration, precipitation
        main_direction: Primary launch direction in degrees (0-360)
    
    Returns:
        DataFrame with daily aggregated weather metrics
    """
    if raw_weather_df.empty:
        return DataFrame()
    
    # Calculate wind direction alignment with main direction
    # cos(main_direction - wind_direction) gives 1 when aligned, -1 when opposite
    wind_dir_diff = np.radians(main_direction - raw_weather_df['wind_direction_10m'])
    raw_weather_df['wind_alignment'] = np.cos(wind_dir_diff)
    
    #
    raw_weather_df['blh'] =raw_weather_df.apply(get_blh,axis=1)
    raw_weather_df['lability']=raw_weather_df['temperature_2m'] - raw_weather_df['temperature_850hPa']

    # Aggregate by date
    daily_weather = raw_weather_df.groupby('date').agg({
        'wind_speed_10m': ['mean', 'min'],  # AVG and MIN wind strength
        'wind_gusts_10m': 'max',            # MAX wind gust
        'wind_alignment': 'mean',           # AVG wind alignment with main direction
        'precipitation': 'sum',             # SUM precipitation
        'sunshine_duration': 'sum',         # SUM sunshine
        'cloud_cover_low': 'mean',
        'wind_speed_850hPa': 'mean',
        'blh': 'max',
        'lability': 'max'
    }).reset_index()
    
    # Flatten multi-index columns
    daily_weather.columns = [
        'date',
        'avg_wind_speed',
        'min_wind_speed',
        'max_wind_gust',
        'avg_wind_alignment',
        'total_precipitation',
        'total_sunshine',
        'avg_cloud_cover',
        'wind_speed_850hPa',
        'max_boundary_layer_height',
        'max_lapse_rate'
    ]
    
    logger.debug(f"Aggregated {len(raw_weather_df)} hourly records to {len(daily_weather)} daily records")
    
    return daily_weather


async def load_agg_weather_data(site_name:str, main_direction: int) -> DataFrame:
    raw_weather_df = await load_weather_data(site_name)
    weather_df = aggregate_weather(raw_weather_df,main_direction)
    
    # Add day-of-week features
    weather_df['date'] = to_datetime(weather_df['date'])
    weather_df['day_of_week'] = weather_df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    weather_df['is_weekend'] = (weather_df['day_of_week'] >= 5).astype(int)  # 1 for Sat/Sun, 0 for weekdays
    
    # Filter out invalid data
    weather_df = weather_df[
        (weather_df['avg_wind_speed'].notna()) &
        (weather_df['total_sunshine'].notna())
    ]

    return weather_df
    

async def sync_weather(site_name:str):
    infos = await get_stats()
    logger.info(infos.head())

    site_info = infos[infos['site_name'] == site_name]
    if site_info.empty:
        logger.error(f"Site {site_name} not found in stats")
        return

    lat, lng, elev = site_info.iloc[0]['geo_latitude'], site_info.iloc[0]['geo_longitude'], site_info.iloc[0]['elevation']
    start_date = get_start_date(site_info.iloc[0]['last_weather_time'])
    end_date = get_end_date(start_date)

    df = await refresh_weather_data(lat, lng, elev, start_date, end_date)
    if df is None or df.empty:
        return 

    df['site_name'] = site_name
    await save_weather_data(df)


if __name__=='__main__':
    import asyncio
    asyncio.run(sync_weather('Börry'))