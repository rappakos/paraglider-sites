import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config import DefaultConfig
from glider_sites_app.routes import setup_routes
from glider_sites_app.db import setup_db, close_db

CONFIG = DefaultConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database
    await setup_db(app)
    yield
    # Shutdown: Close database connections
    await close_db(app)


def create_app():
    app = FastAPI(
        title="Paraglider Sites API",
        description="API for paraglider site data analysis and modeling",
        version="2.0.0",
        lifespan=lifespan
    )

    # Store config
    app.state.config = CONFIG
    app.state.db_name = None  # Will be set by setup_db

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup routes
    setup_routes(app)

    return app


app = create_app()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "app:app",
        host='localhost',
        port=CONFIG.PORT,
        reload=True
    )