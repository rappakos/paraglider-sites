# views.py
import aiohttp_jinja2
from aiohttp import web
from . import db

MIN_DATE = '2018-01-01'


def redirect(router, route_name, site_name = None):
    location = router[route_name].url_for(site_name=site_name)
    return web.HTTPFound(location)

@aiohttp_jinja2.template('index.html')
async def index(request):
    return {'data': (await db.get_stats()).to_dict('records')}
 
@aiohttp_jinja2.template('site.html')
async def site_details(request):
    site_name = request.match_info.get('site_name', None)
    # lazy
    all = await db.get_stats()
    current = all[all['site_name']==site_name]

    return {
            'name':site_name,
            'data': current.to_dict('records'),
            'flights': []
    }


async def load_flights(request):
    site_name = request.match_info.get('site_name', None)    
    if request.method == 'POST':

        print('TODO implement')


        raise redirect(request.app.router, 'site_details', site_name=site_name)
    else:
        raise NotImplementedError("invalid?")
