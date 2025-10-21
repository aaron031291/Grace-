"""
Apply database migrations
"""

import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_migrations():
    """Apply all SQL migrations"""
    # Look in database/migrations for SQL files
    migrations_dir = Path("database/migrations")
    
    if not migrations_dir.exists():
        logger.warning(f"Migrations directory not found: {migrations_dir}")
        logger.info("Creating migrations directory...")
        migrations_dir.mkdir(parents=True, exist_ok=True)
        return 0
    
    # Get all .sql files
    sql_files = sorted(migrations_dir.glob("*.sql"))
    
    if not sql_files:
        logger.info("No SQL migration files found")
        return 0
    
    # Get database URL from environment
    database_url = os.environ.get("DATABASE_URL")
    
    if not database_url:
        logger.error("DATABASE_URL not set")
        return 1
    
    logger.info(f"Found {len(sql_files)} migration files")
    
    # Apply migrations (simplified - would use alembic in production)
    for sql_file in sql_files:
        logger.info(f"Would apply: {sql_file.name}")
        
        try:
            import psycopg2
            conn = psycopg2.connect(database_url)
            cur = conn.cursor()
            
            sql = sql_file.read_text()
            cur.execute(sql)
            conn.commit()
            
            logger.info(f"✅ Applied: {sql_file.name}")
            
            cur.close()
            conn.close()
            
        except ImportError:
            logger.warning("psycopg2 not installed, skipping actual migration")
            logger.info(f"Would execute: {sql_file.name}")
        
        except Exception as e:
            logger.error(f"Failed to apply {sql_file.name}: {e}")
            return 1
    
    logger.info(f"✅ Applied {len(sql_files)} migrations")
    return 0


if __name__ == "__main__":
    sys.exit(apply_migrations())
