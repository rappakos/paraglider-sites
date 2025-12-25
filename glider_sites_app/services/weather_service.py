import logging
from datetime import datetime, timedelta
from glider_sites_app.repositories.sites_repository import get_stats
from glider_sites_app.repositories.weather_repository import save_weather_data
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