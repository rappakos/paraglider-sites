# db.py
import aiosqlite
from sqlalchemy import create_engine,text
from pandas import DataFrame, read_sql_query

DB_NAME = './glider_sites.db'
INIT_SCRIPT = './glider_sites_app/init_db.sql'
#START_DATE = '2018-01-01' # do not load earlier flights


async def setup_db(app):
    app['DB_NAME'] = DB_NAME
    async with aiosqlite.connect(DB_NAME) as db:
        # only test
        async with db.execute('SELECT site_name, count(IDFlight) FROM dhv_flights GROUP BY site_name ') as cursor:
            async for row in cursor:
                print(row)

        #
        with open(INIT_SCRIPT, 'r') as sql_file:
            sql_script = sql_file.read()
            await db.executescript(sql_script)
            await db.commit()


async def get_stats():  
    engine = create_engine(f'sqlite:///{DB_NAME}')
    with engine.connect() as db:
        param = {}
        df  = read_sql_query(text(f"""
                        SELECT 
                            s.site_name, s.dhv_site_id, s.geo_latitude, s.geo_longitude, max(f.FlightDate) [last_flight_date], count(f.IDFlight) [flight_count]
                        FROM sites s
                        left join dhv_flights f on f.site_name=s.site_name
                    """), db, params=param)
        return df
    
async def get_flights(site_name:str):
    engine = create_engine(f'sqlite:///{DB_NAME}')
    with engine.connect() as db:
        param = {'site_name':site_name}
        df  = read_sql_query(text(f"""
                        SELECT 
                            [IDFlight], [FlightDate], [FlightStartTime], [FKPilot], [Glider], [GliderClassification], [FlightDuration], [BestTaskPoints], [BestTaskType]
                        FROM dhv_flights f 
                        WHERE f.site_name=:site_name
                    """), db, params=param)
        return df

async def save_dhv_flights(site_name:str, df_flights):
    df_flights['site_name'] = df_flights.apply(lambda row: site_name,axis=1)
    async with aiosqlite.connect(DB_NAME) as db:
        for params in df_flights.itertuples(index=False):            
            #print(params)
            res = await db.execute_insert("""
                            INSERT OR IGNORE INTO dhv_flights ([IDFlight], [FlightDate], [FlightStartTime], [FKPilot], [Glider], [GliderClassification], [FlightDuration], [BestTaskPoints], [BestTaskType],[site_name])
                            SELECT :IDFlight, :FlightDate, :FlightStartTime, :FKPilot, :Glider, :GliderClassification, :FlightDuration, :BestTaskPoints, :BestTaskType,:site_name
                        """, params)
            #print(res)
        await db.commit()