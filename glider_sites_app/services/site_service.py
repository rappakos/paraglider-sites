# services/site_service.py
from glider_sites_app.repositories.sites_repository import get_stats
from glider_sites_app.analysis.model_loader import load_site_model


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
        current = current.copy()
        result['has_model'] = True
        result['rf_model'] = {'feature_importances': rf_model['feature_importance'].to_dict('records')}
    else:
        result = current.copy()
        result['has_model'] = False


    print(result)

    return result