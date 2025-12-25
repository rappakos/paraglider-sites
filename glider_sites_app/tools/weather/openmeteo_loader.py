
import logging
import requests
import pandas as pd

from glider_sites_app.schemas import SiteBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


URL_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"  # PRE 2022-01-01 - does not include pressure level data
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"  # FROM 2022-01-01, include pressure level data


TIME_ZONE = "Europe/Berlin"
START_HOUR = 10
END_HOUR = 18

HOURLY_PARAMS_ARCHIVE = [
        "temperature_2m",
        "dew_point_2m",
        "precipitation",
        "weather_code",
        "cloud_cover_low",
        "surface_pressure", 
        "wind_speed_10m", 
        "wind_direction_10m", 
        "wind_gusts_10m", 
        "sunshine_duration",
        "boundary_layer_height",
        "direct_radiation",
        "diffuse_radiation"
]

HOURLY_PARAMS = ["temperature_2m", # in archive
                 "dew_point_2m", # in archive
                 "precipitation", # in archive
                 "weather_code", # in archive
                 "cloud_cover_low",   # Detects Stratus/Fog ,  in archive              
                 "surface_pressure",  # in archive
                 "wind_speed_10m",  # in archive
                 "wind_direction_10m",  # in archive
                 "wind_gusts_10m",  # in archive
                 "sunshine_duration",  # in archive
                 "boundary_layer_height",  # in archive
                 "direct_radiation",  # in archive
                 "diffuse_radiation", #  Detects Cirrus/Haze ,  in archive 
                 "cape", 
                 "lifted_index",                  
                 "temperature_950hPa",
                 "temperature_850hPa",
                 "wind_speed_950hPa",
                 "wind_direction_950hPa",             
                 "wind_speed_850hPa", 
                 "wind_direction_850hPa", 
                 "geopotential_height_850hPa"]

async def refresh_weather_data(geo_lat: float,geo_long:float, elevation:float, start_date:str, end_date: str ):
    if not start_date:
        start_date = '2018-01-01'
        end_date = '2018-12-31'

    use_archive = end_date < '2022-01-01'

    params = {
	    "latitude": geo_lat,
	    "longitude": geo_long,
        "elevation": elevation,
	    "start_date": start_date,
	    "end_date": end_date,
	    "hourly": HOURLY_PARAMS_ARCHIVE if use_archive else HOURLY_PARAMS,
    }

    openmeteo_url = f'{URL_ARCHIVE if use_archive else URL}?latitude={params["latitude"]}&longitude={params["longitude"]}&elevation={params["elevation"]}&start_date={params["start_date"]}&end_date={params["end_date"]}&hourly={",".join(params["hourly"])}&timezone={TIME_ZONE}'
    res = requests.get(openmeteo_url)
    if res.status_code == 200:
        data = res.json()
        logger.info(f"Weather data fetched for lat={geo_lat}, lon={geo_long} from {start_date} to {end_date}")
        logger.info(f"Data keys: {list([k for k in data.keys() if k not in HOURLY_PARAMS])}")
        logger.info(f"Timezone: {data.get('timezone', 'N/A')}, offset: {data.get('utc_offset_seconds', 'N/A')}")
        df = pd.DataFrame(data['hourly'], columns= data['hourly_units'])
        df['time'] = pd.to_datetime(df['time'])
        filter_mask = df['time'].dt.hour.between(START_HOUR, END_HOUR)

    return df[filter_mask]


if __name__ == "__main__":
    import asyncio
    # test - Rammi
    rammi_nw = SiteBase(site_name='Rammelsberg NW', dhv_site_id=9427, geo_latitude=51.889874886365874, geo_longitude=10.43097291843072, elevation=610 )
    d = '2018-01-01'
    weather_data = asyncio.run(refresh_weather_data(rammi_nw.geo_latitude, rammi_nw.geo_longitude, rammi_nw.elevation, d,d))
    print(weather_data.tail(10))    
    d = '2022-01-01'
    weather_data = asyncio.run(refresh_weather_data(rammi_nw.geo_latitude, rammi_nw.geo_longitude, rammi_nw.elevation, d,d))
    print(weather_data.tail(10))
