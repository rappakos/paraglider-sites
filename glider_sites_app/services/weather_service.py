import logging
from datetime import datetime, timedelta
from glider_sites_app.repositories.sites_repository import get_stats
from glider_sites_app.tools.weather.openmeteo_loader import refresh_weather_data

MIN_DATE = '2018-01-01'


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    start_date = site_info.iloc[0]['last_weather_time'] or MIN_DATE
    end_date = get_end_date(start_date)

    df = await refresh_weather_data(lat, lng, elev, start_date, end_date)

    # TODO save
    #logger.info(df.head())

    return df


if __name__=='__main__':
    import asyncio
    asyncio.run(sync_weather('Rammelsberg NW'))