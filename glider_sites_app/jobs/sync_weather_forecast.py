# glider_sites_app/jobs/sync_forecast.py

import logging

from datetime import datetime, timedelta

from glider_sites_app.repositories.weather_repository import save_daily_forecast_data
from glider_sites_app.services.site_service import get_all_sites
from glider_sites_app.services.weather_service import load_forecast_weather

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()    

    async def main():
        """
        Checking in linux:
        sqlite3 glider_sites.db "SELECT * FROM weather_forecasts ORDER BY forecast_date DESC, site_name, target_date LIMIT 10"
        Checking in Windows:
        python -c "import sqlite3; conn = sqlite3.connect('glider_sites.db'); print('\n'.join([str(row) for row in conn.execute('SELECT * FROM weather_forecasts GROUP BY site_name, forecast_date ORDER BY forecast_date DESC LIMIT 10').fetchall()]))"
        """

        startdate = datetime.now().strftime('%Y-%m-%d')
        enddate = (datetime.strptime(startdate, '%Y-%m-%d') + timedelta(days=7)).strftime('%Y-%m-%d')
        try:
            sites = await get_all_sites()
            logger.info(f"Loaded {len(sites)} sites")
        
            for site in sites:
                site_name = site['site_name']
                logger.info(f"Loading forecast weather for site: {site_name}")
                weather_df = await load_forecast_weather(site_name, startdate, enddate)
                logger.info(f"Loaded forecast weather for site: {site_name}, data shape: {weather_df.shape}")

                await save_daily_forecast_data(site_name, weather_df, startdate)

        except Exception as e:
            logger.error(f"Error in fetching weather forecast: {e}", exc_info=True)
           

    asyncio.run(main())