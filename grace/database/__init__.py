"""
Database configuration and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import os
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./grace_data.db")

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=False
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Database session dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables and create default data"""
    from grace.auth.models import Base, User, Role
    from grace.auth.security import get_password_hash
    from datetime import datetime, timezone
    import uuid
    
    logger.info(f"Initializing database: {DATABASE_URL}")
    
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")
    
    db = SessionLocal()
    try:
        # Create default roles
        user_role = db.query(Role).filter(Role.name == "user").first()
        if not user_role:
            user_role = Role(
                id=str(uuid.uuid4()),
                name="user",
                description="Default user role"
            )
            db.add(user_role)
            logger.info("Created 'user' role")
        
        admin_role = db.query(Role).filter(Role.name == "admin").first()
        if not admin_role:
            admin_role = Role(
                id=str(uuid.uuid4()),
                name="admin",
                description="Administrator role"
            )
            db.add(admin_role)
            logger.info("Created 'admin' role")
        
        superuser_role = db.query(Role).filter(Role.name == "superuser").first()
        if not superuser_role:
            superuser_role = Role(
                id=str(uuid.uuid4()),
                name="superuser",
                description="Superuser with full access"
            )
            db.add(superuser_role)
            logger.info("Created 'superuser' role")
        
        db.commit()
        
        # Create default admin user
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            admin_user = User(
                id=str(uuid.uuid4()),
                username="admin",
                email="admin@grace-ai.local",
                hashed_password=get_password_hash("Admin123!"),
                full_name="System Administrator",
                is_active=True,
                is_verified=True,
                is_superuser=True,
                password_changed_at=datetime.now(timezone.utc)
            )
            admin_user.roles.append(superuser_role)
            db.add(admin_user)
            db.commit()
            logger.info("Created default admin user (username: admin, password: Admin123!)")
        
        logger.info("Database initialization complete")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        db.rollback()
        raise
    finally:
        db.close()


__all__ = ['engine', 'SessionLocal', 'get_db', 'init_db']
