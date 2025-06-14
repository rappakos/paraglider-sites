# db.py
import aiosqlite
from sqlalchemy import create_engine,text
from pandas import DataFrame, read_sql_query

DB_NAME = './glider_sites.db'
INIT_SCRIPT = './glider_sites_app/init_db.sql'
START_DATE = '2018-01-01' # do not load earlier flights


async def setup_db(app):
    app['DB_NAME'] = DB_NAME
    async with aiosqlite.connect(DB_NAME) as db:
        # only test
        async with db.execute("SELECT 1") as cursor:
            async for row in cursor:
                print(row[0])

        #
        with open(INIT_SCRIPT, 'r') as sql_file:
            sql_script = sql_file.read()
            await db.executescript(sql_script)
            await db.commit()


async def get_stats():
    import pandas as pd
    
    engine = create_engine(f'sqlite:///{DB_NAME}')
    with engine.connect() as db:
        param = {}
        df  = pd.read_sql_query(text(f"""
                        SELECT 
                            site_name, dhv_site_id, geo_latitude, geo_longitude
                        FROM sites 
                    """), db, params=param)
        return df