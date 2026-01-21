# services/site_service.py
import asyncio
import logging
from datetime import datetime, timedelta
from glider_sites_app.repositories.sites_repository import get_stats
from glider_sites_app.analysis.model_loader import load_site_model, load_bayesian_model
from glider_sites_app.analysis.bayes_network import predict_from_raw_weather
from glider_sites_app.services.weather_service import load_forecast_weather

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple in-memory cache with TTL
_forecast_cache = {}
CACHE_TTL = timedelta(hours=6)

def _is_cache_valid(cache_entry):
    """Check if cache entry is still valid"""
    if cache_entry is None:
        return False
    return datetime.now() - cache_entry['timestamp'] < CACHE_TTL

async def get_all_sites():
    """Get all paraglider sites with statistics"""
    stats = await get_stats()
    return stats.to_dict('records')

async def get_site_data(site_name: str):
    """Helper function to get site data"""
    all_stats = await get_stats()
    current = all_stats[all_stats['site_name'] == site_name]
    if current.empty:
        return None
    
    result =  current.to_dict('records')[0]

    # Run blocking I/O in thread pool to avoid blocking event loop
    rf_model = await asyncio.to_thread(load_site_model, site_name, type='classifier')
    if rf_model is not None:
        result['has_model'] = True
        result['rf_model'] = {'feature_importances': rf_model['feature_importance'].to_dict('records')}
    else:
        result['has_model'] = False

    return result

async def get_forecast_data(site_name: str, start_date: str = '2025-06-01', end_date: str = '2025-06-07'):
    """Helper function to get forecast data for a site with caching"""
    
    # Check cache
    cache_key = f"{site_name}:{start_date}:{end_date}"
    cache_entry = _forecast_cache.get(cache_key)
    
    if _is_cache_valid(cache_entry):
        logger.info(f"Cache hit for {cache_key}")
        return cache_entry['data']
    
    logger.info(f"Cache miss for {cache_key}, fetching fresh data")
    
    all_stats = await get_stats()
    current = all_stats[all_stats['site_name'] == site_name]
    if current.empty:
        return None

    # Run blocking I/O in thread pool to avoid blocking event loop
    rf_model = await asyncio.to_thread(load_site_model, site_name, type='classifier')
    if rf_model is None:
        return None

    weather_df = await load_forecast_weather(site_name, start_date, end_date)
    X = weather_df[rf_model['features']]
    # Run blocking predict in thread pool
    rf_prediction = await asyncio.to_thread(rf_model['model'].predict, X)
    weather_df['rf_prediction'] = rf_prediction
    # Fraction of trees voting for the predicted class
    rf_proba = await asyncio.to_thread(rf_model['model'].predict_proba, X)    
    weather_df['rf_confidence'] = rf_proba.max(axis=1)

    # Load Bayesian model and add predictions
    bayesian_model_data = await asyncio.to_thread(load_bayesian_model, site_name)
    if bayesian_model_data is not None:
        # Run Bayesian prediction in thread pool
        bayesian_predictions = await asyncio.to_thread(
            predict_from_raw_weather, 
            bayesian_model_data['model'], 
            weather_df
        )
        
        # Merge predictions into weather_df
        weather_df['is_flyable'] = bayesian_predictions['predicted_flyable']
        weather_df['is_flyable_prob'] = bayesian_predictions.apply(lambda row: row['is_flyable_prob'] if row['is_flyable_prob'] > 0.5 else 1 - row['is_flyable_prob'], axis=1)
        # weather_df['xc_potential'] = bayesian_predictions.apply(
        #     lambda row: max(
        #         (row['xc_local_prob'], 'Local'),
        #         (row['xc_xc_prob'], 'XC'),
        #         (row['xc_hammer_prob'], 'Hammer')
        #     )[1] if row['is_flyable_prob'] > 0.5 else 'Sled',
        #     axis=1
        # )
        weather_df['abgleiter'] = bayesian_predictions['abgleiter']
        weather_df['9min_prob'] = bayesian_predictions['9min_prob']
        weather_df['45min_prob'] = bayesian_predictions['45min_prob']
        weather_df['120min_prob'] = bayesian_predictions['120min_prob']

    logging.debug(weather_df.columns.values)

    forecast_data = {"forecast": weather_df.to_dict('records')}
    
    # Cache the result
    _forecast_cache[cache_key] = {
        'data': forecast_data,
        'timestamp': datetime.now()
    }
    
    return forecast_data


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    
    async def main():
        sites = await get_all_sites()
        print(f"Loaded {len(sites)} sites")
        
        site_name = sites[0]['site_name']
        site_data = await get_site_data(site_name)
        print(f"Site data for {site_name}: {site_data}")
        
        forecast_data = await get_forecast_data(site_name, '2026-01-21', '2026-01-27')
        print(f"Forecast data for {site_name}: {forecast_data}")

    asyncio.run(main())