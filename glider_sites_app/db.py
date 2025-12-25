# db.py
import aiosqlite
from sqlalchemy import create_engine, text
from pandas import read_sql_query

DB_NAME = './glider_sites.db'
INIT_SCRIPT = './glider_sites_app/init_db.sql'


async def setup_db():
    """Initialize database on app startup"""
    #app.state.db_name = DB_NAME
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("PRAGMA encoding = 'UTF-8'")
        # Test connection
        #async with db.execute('SELECT site_name, count(IDFlight) FROM dhv_flights GROUP BY site_name ') as cursor:
        #    async for row in cursor:
        #        print(row)

        # Initialize schema
        with open(INIT_SCRIPT, 'r', encoding='utf-8') as sql_file:
            sql_script = sql_file.read()
            await db.executescript(sql_script)
            await db.commit()


async def close_db(app):
    """Close database connections on app shutdown"""
    # With SQLite, connections are short-lived, so nothing to do here
    pass

if __name__ == "__main__":
    import asyncio

    async def test_db():
        await setup_db()
        engine = create_engine(f'sqlite:///{DB_NAME}')
        with engine.connect() as db:
            df = read_sql_query(text("SELECT name FROM sqlite_master WHERE type='table';"), db)
            print("Tables in the database:")
            print(df)

    asyncio.run(test_db())