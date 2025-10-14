"""Database session helper for Grace.

Provides a convenience function to create SQLAlchemy Engine and Session objects using
`DATABASE_URL` environment variable. Falls back to SQLite file if not provided.
"""
from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL = os.environ.get("DATABASE_URL")

# Default local SQLite file
_sqlite_file = os.environ.get("GRACE_SQLITE_PATH", "./grace_dev.sqlite3")


def get_engine(echo: bool = False):
    if DATABASE_URL:
        return create_engine(DATABASE_URL, echo=echo)
    # sqlite fallback
    return create_engine(f"sqlite:///{_sqlite_file}", echo=echo)


def get_session(echo: bool = False) -> Session:
    engine = get_engine(echo=echo)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
