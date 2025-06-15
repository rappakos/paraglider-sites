# dhv_loader.py

import requests 
import urllib
import pandas as pd


MIN_DATE = '2018-01-01'
PAGE_SIZE=500


async def refresh_flight_list(dhv_site_id:int, last_flight_date):

    print(dhv_site_id, last_flight_date)

    query = {"navpars":{"start":0,"limit":PAGE_SIZE,"sort":[{"field":"FlightDate"}]}}
    decoded_url = f"https://de.dhv-xc.de/api/fli/flights?d0={last_flight_date if last_flight_date else MIN_DATE}&fkcat%5B%5D=1&fkto%5B%5D={dhv_site_id}&{urllib.parse.urlencode(query,quote_via=urllib.parse.quote_plus).replace('%27', '%22').replace('+', '')}"
    r = requests.get(decoded_url)
    if r.status_code==200:
        response = r.json()
        df = pd.DataFrame(response['data'])
        #print(df.columns.values)
        return df[['IDFlight','FlightDate','FlightStartTime','FKPilot','Glider','GliderClassification' ,'FlightDuration','BestTaskPoints','BestTaskType']]