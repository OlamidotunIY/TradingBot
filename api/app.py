"""
FastAPI Application - Main API Application

This module creates and configures the FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .routes import trading, strategies, backtest, account

logger = logging.getLogger('trading_bot')

# Application state (will be set during startup)
app_state = {
    'mt5_handler': None,
    'strategy_manager': None,
    'risk_manager': None,
    'database': None,
    'config': None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("API starting up...")
    yield
    logger.info("API shutting down...")


def create_app(config: dict = None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        config: Application configuration

    Returns:
        FastAPI: Configured application
    """
    application = FastAPI(
        title="MT5 Trading Bot API",
        description="REST API for MT5 algorithmic trading bot",
        version="1.0.0",
        lifespan=lifespan
    )

    # Configure CORS
    cors_origins = ["http://localhost:5173", "http://localhost:3000"]
    if config and 'api' in config:
        cors_origins = config['api'].get('cors_origins', cors_origins)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    application.include_router(trading.router, prefix="/api", tags=["Trading"])
    application.include_router(strategies.router, prefix="/api", tags=["Strategies"])
    application.include_router(backtest.router, prefix="/api", tags=["Backtest"])
    application.include_router(account.router, prefix="/api", tags=["Account"])

    @application.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "MT5 Trading Bot API",
            "version": "1.0.0",
            "status": "running"
        }

    @application.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    # Store config
    if config:
        app_state['config'] = config

    return application


# Default application instance
app = create_app()
