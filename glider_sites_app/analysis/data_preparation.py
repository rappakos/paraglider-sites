# analysis/data_preparation.py

import logging
import pandas as pd
from glider_sites_app.repositories.flights_repository import load_flight_counts
from glider_sites_app.services.weather_service import load_agg_weather_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def prepare_training_data(site_name: str, main_direction: int) -> pd.DataFrame:
    """Merge flight and weather data"""
    logger.info(f"Loading data for {site_name}")

    flights_df = await load_flight_counts(site_name)
    weather_df = await load_agg_weather_data(site_name, main_direction)  
    
    logger.debug(f"Weather data: {len(weather_df)} days")
    logger.debug(f"Flight data: {len(flights_df)} days")
    
    # Ensure date columns are the same type before merge
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date
    flights_df['date'] = pd.to_datetime(flights_df['date']).dt.date
    
    # Merge on date - LEFT JOIN to keep all weather days
    merged_df = weather_df.merge(
        flights_df[['date', 'flight_count', 'max_daily_score']], 
        on='date', 
        how='left'
    )

    # Fill days with no flights as 0
    merged_df['flight_count'] = merged_df['flight_count'].fillna(0)
    merged_df['max_daily_score'] = merged_df['max_daily_score'].fillna(0)

    # Keep only weekends OR weekdays with flights
    filter = (merged_df['is_weekend'] == 1) | (merged_df['flight_count'] > 0)
    merged_df = merged_df[filter]   

    
    logger.debug(f"Merged data: {len(merged_df)} days with complete data")
    logger.debug(f"Flight days: {(merged_df['flight_count'] > 0).sum()}")
    logger.debug(f"Weekend days: {merged_df['is_weekend'].sum()}, Weekdays: {(1-merged_df['is_weekend']).sum()}")
    logger.debug(f"Rainy days: {(merged_df['total_precipitation']>0).sum()}")
    
    return merged_df