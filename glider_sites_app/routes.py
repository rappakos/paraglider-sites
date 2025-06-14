import pathlib

from .views import index,site_details,load_flights

PROJECT_ROOT = pathlib.Path(__file__).parent


def setup_routes(app):
    app.router.add_get('/', index)
    app.router.add_get('/{site_name}',site_details, name='site_details')
    app.router.add_post('/{site_name}', load_flights, name='load_flights')
