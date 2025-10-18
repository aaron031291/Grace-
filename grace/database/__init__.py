"""
Database configuration and initialization
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

# Will be initialized with actual URL
engine = None
SessionLocal = None


def init_db(database_url: str = None):
    """Initialize database with given URL"""
    global engine, SessionLocal
    
    if database_url is None:
        from grace.config import get_settings
        settings = get_settings()
        database_url = settings.database.url
    
    engine = create_engine(
        database_url,
        echo=False,
        pool_pre_ping=True
    )
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    logger.info(f"Database initialized: {database_url}")


def get_db():
    """Get database session"""
    if SessionLocal is None:
        init_db()
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


__all__ = ['Base', 'engine', 'SessionLocal', 'init_db', 'get_db']
