"""
Database connection and session management
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base
from ..config import get_settings

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[sessionmaker] = None


async def init_database() -> AsyncEngine:
    """Initialize the database engine and create tables."""
    global _engine, _session_factory

    settings = get_settings()

    if not settings.database_url:
        raise ValueError("DATABASE_URL not configured")

    # Create async engine
    _engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_pre_ping=True,
        # For SQLite in memory/testing
        poolclass=StaticPool if "sqlite" in settings.database_url else None,
        connect_args={"check_same_thread": False}
        if "sqlite" in settings.database_url
        else {},
    )

    # Create session factory
    _session_factory = sessionmaker(
        bind=_engine, class_=AsyncSession, expire_on_commit=False
    )

    # Create tables
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized successfully")
    return _engine


async def close_database():
    """Close database connections."""
    global _engine, _session_factory

    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed")


def get_engine() -> AsyncEngine:
    """Get the database engine."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _engine


def get_session_factory() -> sessionmaker:
    """Get the session factory."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _session_factory


@asynccontextmanager
async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_database_dependency() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency to get database session."""
    async with get_database() as session:
        yield session
