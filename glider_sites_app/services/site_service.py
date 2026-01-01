# services/site_service.py
import asyncio
from glider_sites_app.repositories.sites_repository import get_stats
from glider_sites_app.analysis.model_loader import load_site_model, load_bayesian_model
from glider_sites_app.analysis.bayes_network import predict_from_raw_weather
from glider_sites_app.services.weather_service import load_forecast_weather


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
    """Helper function to get forecast data for a site (placeholder)"""
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
        weather_df['is_flyable_prob'] = bayesian_predictions['is_flyable_prob']
        weather_df['xc_potential'] = bayesian_predictions.apply(
            lambda row: max(
                (row['xc_local_prob'], 'Local'),
                (row['xc_xc_prob'], 'XC'),
                (row['xc_hammer_prob'], 'Hammer')
            )[1] if row['is_flyable_prob'] > 0.5 else 'Sled',
            axis=1
        )

    forecast_data = {"forecast": weather_df.to_dict('records')}  
    return forecast_data