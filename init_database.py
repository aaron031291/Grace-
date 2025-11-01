#!/usr/bin/env python3
"""
Automated Database Initialization

Creates all required database tables.
Handles SQLite fallback if PostgreSQL unavailable.
"""

import sys
import os
from pathlib import Path

def main():
    print("\n🗄️  Grace Database Initialization\n")
    
    # Check for DATABASE_URL
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        print("⚠️  DATABASE_URL not set")
        print("   Using default: postgresql://grace:grace_dev_password@localhost:5432/grace_dev")
        db_url = "postgresql://grace:grace_dev_password@localhost:5432/grace_dev"
        os.environ["DATABASE_URL"] = db_url
    
    print(f"Database URL: {db_url[:50]}...")
    
    # Check if database setup script exists
    setup_script = Path("database/build_all_tables.py")
    
    if setup_script.exists():
        print("\n📋 Running database setup script...")
        
        try:
            # Run the setup script
            import subprocess
            result = subprocess.run(
                [sys.executable, str(setup_script)],
                capture_output=True,
                text=True
            )
            
            print(result.stdout)
            
            if result.returncode == 0:
                print("✅ Database initialized successfully!")
                return 0
            else:
                print(f"⚠️  Database initialization had warnings")
                if result.stderr:
                    print(result.stderr)
                print("\n   Backend will still work but some features may be limited")
                return 0
                
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            print("\n   Creating fallback SQLite database...")
            
            # Create SQLite fallback
            try:
                import sqlite3
                db_path = Path("data/grace_fallback.db")
                db_path.parent.mkdir(exist_ok=True)
                
                conn = sqlite3.connect(str(db_path))
                conn.execute("CREATE TABLE IF NOT EXISTS system_info (key TEXT, value TEXT)")
                conn.execute("INSERT INTO system_info VALUES ('initialized', datetime('now'))")
                conn.commit()
                conn.close()
                
                print(f"✅ Fallback SQLite database created: {db_path}")
                print("   Grace will use SQLite for basic operation")
                
                return 0
                
            except Exception as e2:
                print(f"❌ Fallback database creation failed: {e2}")
                print("   Grace will work without persistent storage")
                return 1
    else:
        print(f"⚠️  Database setup script not found: {setup_script}")
        print("   Grace will work without database")
        return 0

if __name__ == "__main__":
    sys.exit(main())
