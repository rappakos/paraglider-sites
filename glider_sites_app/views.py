# views.py
import os
import aiohttp_jinja2
from aiohttp import web


MIN_DATE = '2018-01-01'


def redirect(router, route_name, org = None):
    location = router[route_name].url_for(org=org)
    return web.HTTPFound(location)

@aiohttp_jinja2.template('index.html')
async def index(request):    
    return {'data': 'test'}
 