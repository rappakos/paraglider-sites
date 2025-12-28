# services/flight_service.py
import logging
from glider_sites_app.repositories.sites_repository import get_stats
from glider_sites_app.schemas import SiteBase
from ..tools.flights.dhv_loader import refresh_flight_list
from ..repositories.flights_repository import save_dhv_flights, pilot_stats, load_flight_counts

MIN_DATE = '2018-01-01'

TOL = 0.002787333064432527 # Rammi NW to SW half-distance, ca 150m

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def pilot_statistics():
    """Get pilot statistics from the repository"""
    
    res = await pilot_stats()

    logger.debug(f"Top pilots:\n{res.head()}")
    percentiles = res['best_score'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    logger.debug(f"\nBest Score Percentiles:")
    logger.debug(f"25th percentile: {percentiles[0.25]:.2f}")
    logger.debug(f"50th percentile (median): {percentiles[0.5]:.2f}")
    logger.debug(f"75th percentile: {percentiles[0.75]:.2f}")
    logger.debug(f"90th percentile: {percentiles[0.9]:.2f}")


    return res

async def load_flight_data(site_name: str):
    res = await load_flight_counts(site_name)
    pilot_stats = await pilot_statistics()
    res = res.merge(
        pilot_stats,
        how='left',
        left_on='best_pilot_id',
        right_on='FKPilot',
        suffixes=('', '_pilot')
    )
    logger.debug(res.head())

    return res

async def sync_dhv_flights(site_name: str):
    """Sync DHV flights for a given site"""

    # get site info: dhv_site_id, last_flight_date
    # note: same dhv_site_id can have multiple site_names (e.g., different launch directions)
    infos = await get_stats()
    logger.info(infos.head())

    site_info = infos[infos['site_name'] == site_name]
    if site_info.empty:
        logger.error(f"Site {site_name} not found in stats")
        return
    dhv_site_id, last_date = site_info.iloc[0]['dhv_site_id'],  site_info.iloc[0]['last_flight_date']
    if dhv_site_id is None:
        logger.error(f"Site {site_name} does not have a DHV site ID")
        return
    if last_date is None:
        last_date = MIN_DATE

    # TODO loop until no more new flights
    flights = await refresh_flight_list(dhv_site_id, last_date)
    if flights.empty:
        logger.info(f"No new flights for site {site_name} since {last_date}")
        return
    
    # check if there are multiple site_names for the same dhv_site_id
    same_id_sites = infos[infos['dhv_site_id'] == dhv_site_id]
    if len(same_id_sites) > 1:
        from ..tools.flights.dhv_loader import classify_site
        site_list = [SiteBase(**site_info) for _, site_info in same_id_sites.iterrows()]
        flights['site_name'] = flights.apply(
            lambda row: classify_site(
                sites=site_list,
                lat=row['FirstLat'],
                lon=row['FirstLng'],
                tol=TOL
            ),
            axis=1
        )
    else:
        flights['site_name'] = site_name

    logger.info(flights.tail())

    # save 
    await save_dhv_flights(flights[flights['site_name'].notna()])


if __name__ == "__main__":
    import asyncio
    #asyncio.run(sync_dhv_flights('Porta'))
    asyncio.run(load_flight_data('Porta'))