"""
Wait for database to be ready
"""

import sys
import time
import os
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError


def wait_for_db(max_retries=30, delay=2):
    """Wait for database to be ready"""
    database_url = os.getenv("DATABASE_URL", "postgresql://localhost/grace_dev")
    
    print(f"Waiting for database at {database_url}")
    
    for i in range(max_retries):
        try:
            engine = create_engine(database_url)
            connection = engine.connect()
            connection.close()
            print("✅ Database is ready!")
            return True
        except OperationalError:
            print(f"⏳ Attempt {i+1}/{max_retries}: Database not ready, waiting {delay}s...")
            time.sleep(delay)
    
    print("❌ Database did not become ready in time")
    return False


if __name__ == "__main__":
    if not wait_for_db():
        sys.exit(1)
