# services/site_service.py
from glider_sites_app.repositories.sites_repository import get_stats
from glider_sites_app.analysis.model_loader import load_site_model
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

    rf_model = load_site_model(site_name, type='classifier')
    if rf_model is not None:
        result['has_model'] = True
        result['rf_model'] = {'feature_importances': rf_model['feature_importance'].to_dict('records')}
    else:
        result['has_model'] = False

    return result

async def get_forecast_data(site_name: str):
    """Helper function to get forecast data for a site (placeholder)"""
    all_stats = await get_stats()
    current = all_stats[all_stats['site_name'] == site_name]
    if current.empty:
        return None

    rf_model = load_site_model(site_name, type='classifier')
    if rf_model is None:
        return None
    
    # get weather data
    start_date = '2025-06-01'
    end_date = '2025-06-07'

    weather_df = await load_forecast_weather(site_name, start_date, end_date)
    X = weather_df[rf_model['features']]
    weather_df['predictions'] = rf_model['model'].predict(X)

    forecast_data = {"forecast": weather_df.to_dict('records')}  
    return forecast_data