"""
Database configuration and initialization
"""

from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator, Optional
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

# Global engine and session maker
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def init_db(database_url: Optional[str] = None) -> None:
    """Initialize database with given URL"""
    global _engine, _SessionLocal
    
    if database_url is None:
        from grace.config import get_settings
        settings = get_settings()
        database_url = settings.database.url
    
    _engine = create_engine(
        database_url,
        echo=False,
        pool_pre_ping=True
    )
    
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    
    # Create all tables
    Base.metadata.create_all(bind=_engine)
    
    logger.info(f"Database initialized: {database_url}")


def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    if _SessionLocal is None:
        init_db()
    
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized")
    
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


__all__ = ['Base', 'init_db', 'get_db']
