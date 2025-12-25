
import logging
import requests
import pandas as pd

from glider_sites_app.schemas import SiteBase
from glider_sites_app.tools.weather.constants import END_HOUR, HOURLY_PARAMS, HOURLY_PARAMS_ARCHIVE, START_HOUR, TIME_ZONE, URL, URL_ARCHIVE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.debug(f"Data keys: {list([k for k in data.keys() if k not in HOURLY_PARAMS])}")
        logger.debug(f"Timezone: {data.get('timezone', 'N/A')}, offset: {data.get('utc_offset_seconds', 'N/A')}")
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
    print(weather_data.tail())    
    d = '2022-01-01'
    weather_data = asyncio.run(refresh_weather_data(rammi_nw.geo_latitude, rammi_nw.geo_longitude, rammi_nw.elevation, d,d))
    print(weather_data.tail())
