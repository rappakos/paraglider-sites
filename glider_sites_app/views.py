# views.py
import aiohttp_jinja2
from aiohttp import web
from . import db

MIN_DATE = '2018-01-01'


def redirect(router, route_name, org = None):
    location = router[route_name].url_for(org=org)
    return web.HTTPFound(location)

@aiohttp_jinja2.template('index.html')
async def index(request):
    return {'data': (await db.get_stats()).to_dict('records')}
 