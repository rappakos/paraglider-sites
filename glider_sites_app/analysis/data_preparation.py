# analysis/data_preparation.py

import logging
import holidays
import pandas as pd
from datetime import date

from glider_sites_app.services.weather_service import load_agg_weather_data
from glider_sites_app.services.flight_service import load_flight_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Site-specific minimum training date (e.g. launch opened for paragliders later)
SITE_MIN_DATE = {
    'Porta': date(2020, 1, 1),
}

# Competitor sites: when these sites had flights but ours didn't,
# it's likely a site-choice issue rather than non-flyable conditions.
# Feature: total flight count at competitor sites on that day.
COMPETITOR_SITES = {
    'Dielmissen': ['Börry', 'Königszinne'],
    #'Königszinne': ['Dielmissen', 'Rammelsberg NW'], did not seem to cause noise
}

async def prepare_training_data(site_name: str, use_workingdays: bool = False, FKPilotID: str = None) -> pd.DataFrame:
    """Merge flight and weather data"""
    logger.info(f"Loading data for {site_name}")

    flights_df = await load_flight_data(site_name, FKPilotID=FKPilotID)
    weather_df = await load_agg_weather_data(site_name)  
    
    logger.debug(f"Weather data: {len(weather_df)} days")
    logger.debug(f"Flight data: {len(flights_df)} days")
    
    # Ensure date columns are the same type before merge
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date
    flights_df['date'] = pd.to_datetime(flights_df['date']).dt.date

    # Apply site-specific minimum date filter
    min_date = SITE_MIN_DATE.get(site_name)
    if min_date:
        weather_df = weather_df[weather_df['date'] >= min_date]
        flights_df = flights_df[flights_df['date'] >= min_date]
        logger.info(f"Filtered {site_name} to dates >= {min_date}")

    # Merge on date - LEFT JOIN to keep all weather days
    merged_df = weather_df.merge(
        flights_df[['date', 'flight_count', 'max_daily_score','best_score','avg_flight_duration']], # best_score = best global score of the best pilot on that day
        on='date', 
        how='left'
    )

    # Fill days with no flights as 0
    merged_df['flight_count'] = merged_df['flight_count'].fillna(0)
    merged_df['best_score'] = merged_df['best_score'].fillna(0)
    merged_df['max_daily_score'] =merged_df['max_daily_score'].fillna(0)

    # Add competitor flights feature
    competitors = COMPETITOR_SITES.get(site_name, [])
    if competitors:
        competitor_dfs = []
        for comp_site in competitors:
            comp_flights = await load_flight_data(comp_site)
            comp_flights['date'] = pd.to_datetime(comp_flights['date']).dt.date
            competitor_dfs.append(comp_flights[['date', 'flight_count']].rename(
                columns={'flight_count': f'flights_{comp_site}'}
            ))
        for comp_df in competitor_dfs:
            merged_df = merged_df.merge(comp_df, on='date', how='left')
        comp_cols = [c for c in merged_df.columns if c.startswith('flights_')]
        merged_df[comp_cols] = merged_df[comp_cols].fillna(0)
        merged_df['competitor_flights'] = merged_df[comp_cols].sum(axis=1)
        merged_df.drop(columns=comp_cols, inplace=True)
    else:
        merged_df['competitor_flights'] = 0

    # Keep only weekends OR weekdays with flights
    if not use_workingdays:
        filter = (merged_df['is_weekend'] == 1) | (merged_df['flight_count'] > 0)
        merged_df = merged_df[filter]

    # Remove ambiguous zero-flight days: good alignment + competitors flew
    # These are likely flyable days where pilots chose a different site.
    if competitors:
        ambiguous = (
            (merged_df['flight_count'] == 0) &
            (merged_df['competitor_flights'] > 0) &
            (merged_df['avg_wind_alignment'] > 0.6)
        )
        n_removed = ambiguous.sum()
        if n_removed > 0:
            merged_df = merged_df[~ambiguous].reset_index(drop=True)
            logger.info(f"Removed {n_removed} ambiguous zero-flight days (good alignment + competitor flights)")

    de_holidays = set(holidays.Germany(prov='NI'))
    merged_df['is_workingday'] =(
        (merged_df['is_weekend'] == 0) & 
        (~merged_df['date'].isin(de_holidays))
    ).astype(int)

    
    logger.debug(f"Merged data: {len(merged_df)} days with complete data")
    logger.debug(f"Flight days: {(merged_df['flight_count'] > 0).sum()}")
    logger.debug(f"Weekend days: {merged_df['is_weekend'].sum()}, Weekdays: {(1-merged_df['is_weekend']).sum()}")
    logger.debug(f"Rainy days: {(merged_df['total_precipitation']>0).sum()}")
    
    return merged_df