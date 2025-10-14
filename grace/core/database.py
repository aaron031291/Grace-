"""
Database configuration and session management for Grace system.
Provides async SQLAlchemy setup with support for SQLite and PostgreSQL.
"""

import os
from typing import AsyncGenerator
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Database configuration from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./grace_data.db",  # Default to async SQLite
)

# Support sync SQLite fallback for development
SYNC_DATABASE_URL = os.getenv("SYNC_DATABASE_URL", "sqlite:///./grace_data.db")


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


# Async engine and session factory
async_engine = create_async_engine(
    DATABASE_URL, echo=bool(os.getenv("SQL_ECHO", False)), future=True
)

async_session_factory = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

# Sync engine for migrations and utilities
sync_engine = create_engine(
    SYNC_DATABASE_URL, echo=bool(os.getenv("SQL_ECHO", False)), future=True
)

sync_session_factory = sessionmaker(bind=sync_engine)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides async database session.
    Use with FastAPI's Depends() for route dependencies.
    """
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_session():
    """
    Context manager that provides sync database session.
    Used for migrations and utility scripts.
    """
    return sync_session_factory()


async def init_database():
    """Initialize database tables."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_database():
    """Close database connections."""
    await async_engine.dispose()


class DatabaseConfig:
    """Database configuration class."""

    def __init__(self):
        self.async_url = DATABASE_URL
        self.sync_url = SYNC_DATABASE_URL
        self.echo = bool(os.getenv("SQL_ECHO", False))

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return "sqlite" in self.async_url.lower()

    @property
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL database."""
        return "postgresql" in self.async_url.lower()


# Global database config instance
db_config = DatabaseConfig()
