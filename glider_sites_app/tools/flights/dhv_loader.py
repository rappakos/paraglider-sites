# dhv_loader.py

from typing import List
import requests 
import urllib
import pandas as pd
from ...schemas import SiteBase

PAGE_SIZE=500

async def refresh_flight_list(dhv_site_id:int, last_flight_date: str) -> pd.DataFrame:
    """Fetch flight list from DHV-XC API for a given site since last_flight_date"""
    if last_flight_date is None:
        raise ValueError("last_flight_date cannot be None")
    #if last_flight_date
    

    query = {"navpars":{"start":0,"limit":PAGE_SIZE,"sort":[{"field":"FlightDate"}]}}
    decoded_url = f"https://de.dhv-xc.de/api/fli/flights?d0={last_flight_date}&fkcat%5B%5D=1&fkto%5B%5D={dhv_site_id}&{urllib.parse.urlencode(query,quote_via=urllib.parse.quote_plus).replace('%27', '%22').replace('+', '')}"
    r = requests.get(decoded_url)
    if r.status_code==200:
        response = r.json()
        df = pd.DataFrame(response['data'])
        #print(df.columns.values)
        return df[['IDFlight','FlightDate','FlightStartTime','FirstLat','FirstLng','FKPilot','Glider','GliderClassification' ,'FlightDuration','BestTaskPoints','BestTaskType']]
    
def classify_site(sites:List[SiteBase], lat: float, lon: float, tol: float) -> str:
    """Classify site based on latitude and longitude"""
    if (len(sites) == 0):
        raise ValueError("No sites provided for classification")   

    distances = [(site.site_name, (site.geo_latitude - lat)**2 + (site.geo_longitude - lon)**2) for site in sites]
    within_tolerance = [_ for _ in distances if _[1] < tol**2]
    
    if not within_tolerance:
        return None
    
    closest_site = min(within_tolerance, key=lambda x: x[1])
    return closest_site[0]

if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from glider_sites_app.schemas import SiteBase
    
    # test - Rammi
    rammi_nw = SiteBase(site_name='Rammelsberg NW', dhv_site_id=9427, geo_latitude=51.889874886365874, geo_longitude=10.43097291843072, elevation=610 )
    rammi_sw = SiteBase(site_name='Rammelsberg SW', dhv_site_id=9427, geo_latitude=51.8873305210640, geo_longitude=10.429834748486158, elevation=610 )
    r = (rammi_nw.geo_latitude - rammi_sw.geo_latitude)**2 + (rammi_nw.geo_longitude - rammi_sw.geo_longitude)**2
    print(f"Distance between Rammi NW and SW: {r**0.5}")

    flights = asyncio.run(refresh_flight_list(9427, '2023-01-01'))
    print(flights.tail(5))
    # map site
    flights['site_name'] = flights.apply(lambda row: classify_site([rammi_nw, rammi_sw], row['FirstLat'], row['FirstLng'], tol=r**0.5), axis=1)
    #print(flights[flights['site_name']=='Rammelsberg SW'][['IDFlight','FlightDate','site_name']].tail(5))
    print(flights.groupby('site_name', dropna=False).size())