# services/site_service.py
import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from glider_sites_app.analysis.probabilities import load_site_params_and_fit_weights
from glider_sites_app.repositories.sites_repository import get_stats
from glider_sites_app.analysis.model_loader import load_site_model, load_bayesian_model
from glider_sites_app.analysis.bayes_network import DURATION_BIN_EDGES, predict_from_raw_weather
from glider_sites_app.services.weather_service import load_forecast_weather, load_agg_weather_data

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


def estimate_flight_times(site_name, row):
    """Estimate expected flight time based on probabilities"""

    if row.get('is_flyable',False) == False:
        # shortcut
        return 0.


    edges = DURATION_BIN_EDGES
    probs = [
        row.get('sleddie_prob', 0),
        row.get('establishing_prob', 0),
        row.get('soaring_prob', 0),
        row.get('extended_prob', 0),
        row.get('epic_prob', 0)
    ]
    res = load_site_params_and_fit_weights(site_name,edges, probs)

    logger.info(f"Estimated flight time params for site {site_name}  is {res}")

    return 0.


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

        weather_df['expected_flight_time'] = bayesian_predictions.apply(lambda row: estimate_flight_times(site_name, row), axis=1)

        weather_df['sleddie_prob'] = bayesian_predictions['sleddie_prob']
        weather_df['establishing_prob'] = bayesian_predictions['establishing_prob']
        weather_df['soaring_prob'] = bayesian_predictions['soaring_prob']
        weather_df['extended_prob'] = bayesian_predictions['extended_prob']
        weather_df['epic_prob'] = bayesian_predictions['epic_prob']

    logging.debug(weather_df.columns.values)

    forecast_data = {"forecast": weather_df.to_dict('records')}
    
    # Cache the result
    _forecast_cache[cache_key] = {
        'data': forecast_data,
        'timestamp': datetime.now()
    }
    
    return forecast_data


# 16-point compass labels in clockwise order starting at N (index = round(deg/22.5) % 16)
_COMPASS_LABELS_CW = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
# Display order: CCW from N — indices into _COMPASS_LABELS_CW
_DISPLAY_ORDER = [0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
_DISPLAY_LABELS = [_COMPASS_LABELS_CW[i] for i in _DISPLAY_ORDER]

_RF_FEATURES = ['avg_wind_speed', 'avg_wind_alignment', 'max_wind_gust', 'min_wind_speed',
                'total_sunshine', 'total_precipitation', 'max_lapse_rate']
_MIN_DAYS_FOR_STATS = 10


async def get_direction_stats(site_name: str) -> dict | None:
    """Return 16-bucket compass histogram data for the site details page.

    Returns a dict with keys:
        buckets       – list of 16 compass labels in display order (CCW from N)
        baseline      – count of filtered historical days per bucket
        rf_overlay    – count of RF-predicted-flyable days per bucket, or None
        total_days    – total filtered days included in baseline
        rf_overlay_days – total days in overlay, or None
    Returns None when there is insufficient historical data.
    """
    weather_df = await load_agg_weather_data(site_name)
    if weather_df.empty:
        return None

    # Wind-speed band filter
    filtered = weather_df[
        (weather_df['avg_wind_speed'] >= 4) &
        (weather_df['avg_wind_speed'] <= 40)
    ].copy()

    # Drop rows with NaN in any RF feature or the direction column
    required_cols = _RF_FEATURES + ['avg_wind_direction']
    filtered.dropna(subset=required_cols, inplace=True)

    if len(filtered) < _MIN_DAYS_FOR_STATS:
        return None

    # Assign each row to a compass bucket (0 = N, 1 = NNE, …, 15 = NNW clockwise)
    filtered['bucket'] = (filtered['avg_wind_direction'] / 22.5).round().astype(int) % 16

    # Baseline counts per bucket
    bucket_counts = filtered['bucket'].value_counts()
    baseline_cw = [int(bucket_counts.get(i, 0)) for i in range(16)]
    baseline = [baseline_cw[i] for i in _DISPLAY_ORDER]

    # RF overlay
    rf_overlay = None
    rf_overlay_days = None
    rf_model = await asyncio.to_thread(load_site_model, site_name, type='classifier')
    if rf_model is not None:
        overlay_df = filtered.copy()
        overlay_df['avg_wind_alignment'] = 0.9
        features = rf_model.get('features', _RF_FEATURES)
        predictions = await asyncio.to_thread(rf_model['model'].predict, overlay_df[features])
        flyable_df = overlay_df[predictions]
        flyable_counts = flyable_df['bucket'].value_counts()
        overlay_cw = [int(flyable_counts.get(i, 0)) for i in range(16)]
        rf_overlay = [overlay_cw[i] for i in _DISPLAY_ORDER]
        rf_overlay_days = int(sum(rf_overlay))

    return {
        'buckets': _DISPLAY_LABELS,
        'baseline': baseline,
        'rf_overlay': rf_overlay,
        'total_days': int(len(filtered)),
        'rf_overlay_days': rf_overlay_days,
    }


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