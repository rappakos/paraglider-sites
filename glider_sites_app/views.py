# views.py
import aiohttp_jinja2
from aiohttp import web
from . import db
from .dhv_loader import refresh_flight_list


def redirect(router, route_name, site_name = None):
    location = router[route_name].url_for(site_name=site_name)
    return web.HTTPFound(location)

async def get_site_data(site_name:str):
    # lazy
    all = await db.get_stats()
    current = all[all['site_name']==site_name]
    return current.to_dict('records')

@aiohttp_jinja2.template('index.html')
async def index(request):
    return {'data': (await db.get_stats()).to_dict('records')}
 
@aiohttp_jinja2.template('site.html')
async def site_details(request):
    site_name = request.match_info.get('site_name', None)

    data = await get_site_data(site_name)
    print(data)
    flights = await db.get_flights(site_name)
    #print(flights.sort_values(by='FlightStartTime').head(10))

    return {
            'name':site_name,
            'data': data,
            'flights': flights.sort_values(by='FlightStartTime', ascending=False).head(10).to_dict('records')
    }


async def refresh_flights(request):
    site_name = request.match_info.get('site_name', None)    
    if request.method == 'POST':
        [current] = await get_site_data(site_name)
       
        flights = await refresh_flight_list(current['dhv_site_id'], current['last_flight_date'])
        print(f"{len(flights)} flights loaded")
        #print(flights[['IDFlight','FlightStartTime']].head(20))
        await db.save_dhv_flights(site_name, flights)

        raise redirect(request.app.router, 'site_details', site_name=site_name)
    else:
        raise NotImplementedError("invalid?")
