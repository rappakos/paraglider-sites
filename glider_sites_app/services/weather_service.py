import logging
import numpy as np
from datetime import datetime, timedelta
from pandas import DataFrame, to_datetime, cut

from glider_sites_app.repositories.sites_repository import get_stats, get_main_direction
from glider_sites_app.repositories.weather_repository import fix_temperature_850hPa_pre2020, load_weather_data, save_weather_data
from glider_sites_app.tools.weather.constants import MIN_DATE
from glider_sites_app.tools.weather.era5_reanalysis import extract_temperature_from_era5
from glider_sites_app.tools.weather.openmeteo_loader import refresh_weather_data, load_new_forecast_data

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
            #datetime(datetime.strptime(start_date,'%Y-%m-%d').year,12,31),
            datetime.strptime(start_date,'%Y-%m-%d') + timedelta(days=180),
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
    
    # Add day-of-week features
    daily_weather['date'] = to_datetime(daily_weather['date'])
    daily_weather['day_of_week'] = daily_weather['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    daily_weather['is_weekend'] = (daily_weather['day_of_week'] >= 5).astype(int)  # 1 for Sat/Sun, 0 for weekdays
    
    # Filter out invalid data
    daily_weather = daily_weather[
        (daily_weather['avg_wind_speed'].notna()) &
        (daily_weather['total_sunshine'].notna())
    ]

    return daily_weather

async def load_forecast_weather(site_name:str, start_date:str, end_date:str) -> DataFrame:
    if start_date > end_date:
        return DataFrame()
    
    main_direction =  await get_main_direction(site_name)
    if main_direction is None:
        return DataFrame()

    if datetime.strptime(start_date, '%Y-%m-%d') >= datetime.now() + timedelta(days=-2):
        infos = await get_stats()
        site_info = infos[infos['site_name'] == site_name]
        if site_info.empty:
            logger.error(f"Site {site_name} not found in stats")
            return

        lat, lng, elev = site_info.iloc[0]['geo_latitude'], site_info.iloc[0]['geo_longitude'], site_info.iloc[0]['elevation']        
        raw_weather_df = await load_new_forecast_data(lat, lng, elev)
        raw_weather_df['site_name'] = site_name
        logger.debug(raw_weather_df.head())
    else:
        raw_weather_df = await load_weather_data(site_name)
        filter = (raw_weather_df['date'] >= start_date) & (raw_weather_df['date'] <= end_date)
        raw_weather_df = raw_weather_df[filter]

    weather_df = aggregate_weather(raw_weather_df,main_direction)

    return weather_df


async def load_agg_weather_data(site_name:str) -> DataFrame:
    main_direction =  await get_main_direction(site_name)
    raw_weather_df = await load_weather_data(site_name)
    weather_df = aggregate_weather(raw_weather_df,main_direction)
    
    return weather_df
    

async def sync_weather(site_name:str):
    infos = await get_stats()
    logger.info(infos.head(10))

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

def gust_factor(avg_wind, max_gust):
    """Calculate wind gust factor based on average wind speed and maximum wind gust
        Using contours of  max_gust = b(g) + (C(g) - b(g)) * avg_wind / (C(g) - b(g) + avg_wind )
        And b(g) = 2 v0 g - v0, C(g) = Vmax - v0*n + v0*g, with n=4, v0=4.5 km/h and Vmax = 36 km/h
    """
    V0 = 4.5
    n = 4
    Vmax = 36.0
    def b(g):
        return 2 * V0 * g - V0
    def C(g):
        return Vmax - V0 * n + V0 * g
    def Y(g, avg_wind):
        return b(g) + (C(g) - b(g)) * avg_wind / (C(g) - b(g) + avg_wind )
    
    if max_gust > Y(4, avg_wind):
        return 4.0
    if max_gust > Y(3, avg_wind):
        return 3.0
    if max_gust > Y(2, avg_wind):
        return 2.0
    if max_gust > Y(1, avg_wind):
        return 1.0
    return 0.0


async def fix_temp_850hpa_pre2020():

    infos = await get_stats()  
    # DONE
    for year in []:
        ds = extract_temperature_from_era5(year)
        for index, row in infos.iterrows():
            site_name, lat, lng = row['site_name'], row['geo_latitude'], row['geo_longitude']
            logging.info(f"Processing site {site_name}...")

            # Interpolate to site coordinates
            site_data = ds.interp(latitude=lat, longitude=lng, method='linear')
            
            # Convert to DataFrame (now just time series for this location)
            site_df = site_data.to_dataframe().reset_index()
            
            print(f"{site_name}: {len(site_df)} records")

            site_df['site_name'] = site_name
            site_df['date'] = site_df['valid_time'].dt.strftime('%Y-%m-%d')
            site_df['temperature_850hPa'] = site_df['t'].astype(float) - 273.15  # Convert from Kelvin to Celsius

            site_df = site_df[['site_name', 'date', 'temperature_850hPa']].groupby(['site_name','date']).mean().reset_index()

            logging.info(site_df.head())

            await fix_temperature_850hPa_pre2020(site_df)


if __name__=='__main__':
    import asyncio
    import plotly.express as px
    #asyncio.run(sync_weather('Porta'))
    #asyncio.run(sync_weather('Brunsberg'))
    site_name = 'Börry'
    #weather = asyncio.run(load_agg_weather_data(site_name))
    #weather['gust_factor'] = weather.apply(lambda row: gust_factor(row['avg_wind_speed'], row['max_wind_gust']), axis=1)

    #fig = px.scatter(weather,x='avg_wind_speed', y='max_wind_gust', color='gust_factor', title=f'Wind Gust Factor for {site_name}')
    #fig.update_xaxes(range=[0,30])
    #fig.update_yaxes(range=[0,40])
    #fig.write_image("wind_gusts.png")
    
    #for avg_wind in [8,16,24]:
    #    for max_gust in [16,24,32,38] if avg_wind<=16 else [24,32,36]:
    #        gust_fac= gust_factor(avg_wind, max_gust)
    #       print(f"{avg_wind:2} | {max_gust:2} | {gust_fac} ")

    #turbulence= cut(
    #    weather['gust_factor'], 
    #    bins=[-np.inf, 1.,2., 3., np.inf], 
    #    labels=['Smooth','OK', 'Gusty', 'Dangerous']
    #)
    #print(turbulence.value_counts())

    asyncio.run(fix_temp_850hpa_pre2020())  