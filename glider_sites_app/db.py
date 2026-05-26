# db.py
import logging
import aiosqlite
from sqlalchemy import create_engine, text
from pandas import read_sql_query

logger = logging.getLogger(__name__)

DB_NAME = './glider_sites.db'
INIT_SCRIPT = './glider_sites_app/init_db.sql'

# Shared connect kwargs for aiosqlite
AIOSQLITE_TIMEOUT = 30  # seconds


async def setup_db(app=None):
    """Initialize database on app startup"""
    if app:
        app.state.db_name = DB_NAME
    try:
        async with aiosqlite.connect(DB_NAME, timeout=AIOSQLITE_TIMEOUT) as db:
            await db.execute("PRAGMA busy_timeout = 5000")
            await db.execute("PRAGMA encoding = 'UTF-8'")

            # Initialize schema
            with open(INIT_SCRIPT, 'r', encoding='utf-8') as sql_file:
                sql_script = sql_file.read()
                await db.executescript(sql_script)
                await db.commit()
        logger.info("Database initialized successfully")
    except FileNotFoundError:
        logger.error(f"Init script not found: {INIT_SCRIPT}")
        raise
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


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