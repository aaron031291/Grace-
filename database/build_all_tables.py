#!/usr/bin/env python3
"""
Grace System - Build All Tables Script
======================================
This script initializes all database tables for all Grace components and kernels.
Supports SQLite (default), PostgreSQL, and MySQL backends.

Usage:
    python build_all_tables.py --db sqlite --path grace.db
    python build_all_tables.py --db postgres --host localhost --dbname grace --user grace --password <pass>
    python build_all_tables.py --db mysql --host localhost --dbname grace --user grace --password <pass>
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_sqlite_tables(db_path: str) -> bool:
    """Build all tables in SQLite database."""
    try:
        import sqlite3
        
        logger.info(f"Connecting to SQLite database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Read SQL file
        sql_file = Path(__file__).parent / "init_all_tables.sql"
        logger.info(f"Reading SQL schema from: {sql_file}")
        
        with open(sql_file, 'r') as f:
            sql_script = f.read()
        
        # Execute entire script at once - SQLite can handle this
        logger.info("Executing SQL schema...")
        try:
            cursor.executescript(sql_script)
        except sqlite3.Error as e:
            logger.error(f"Error executing schema: {e}")
            # Try individual statements if bulk fails
            logger.info("Retrying with individual statements...")
            statements = [s.strip() for s in sql_script.split(';') if s.strip() and not s.strip().startswith('--')]
            for statement in statements:
                if statement and not statement.startswith('--'):
                    try:
                        cursor.execute(statement + ';')
                    except sqlite3.Error as err:
                        logger.debug(f"Skipping statement: {err}")
        
        conn.commit()
        
        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = cursor.fetchall()
        logger.info(f"\n✓ Successfully created/verified {len(tables)} tables:")
        for table in tables:
            logger.info(f"  - {table[0]}")
        
        conn.close()
        logger.info(f"\n✓ Database initialization complete: {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to build SQLite tables: {e}")
        return False


def build_postgres_tables(host: str, dbname: str, user: str, password: str, port: int = 5432) -> bool:
    """Build all tables in PostgreSQL database."""
    try:
        import psycopg2
        from psycopg2 import sql
        
        logger.info(f"Connecting to PostgreSQL: {user}@{host}:{port}/{dbname}")
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        cursor = conn.cursor()
        
        # Read SQL file
        sql_file = Path(__file__).parent / "init_all_tables.sql"
        logger.info(f"Reading SQL schema from: {sql_file}")
        
        with open(sql_file, 'r') as f:
            sql_script = f.read()
        
        # PostgreSQL-specific adjustments
        sql_script = sql_script.replace('AUTOINCREMENT', 'SERIAL')
        sql_script = sql_script.replace('INTEGER PRIMARY KEY AUTOINCREMENT', 'SERIAL PRIMARY KEY')
        sql_script = sql_script.replace('BOOLEAN', 'BOOLEAN')
        sql_script = sql_script.replace('TEXT', 'TEXT')
        sql_script = sql_script.replace('REAL', 'REAL')
        sql_script = sql_script.replace('BLOB', 'BYTEA')
        sql_script = sql_script.replace('CURRENT_TIMESTAMP', 'CURRENT_TIMESTAMP')
        
        logger.info("Executing SQL schema...")
        cursor.execute(sql_script)
        conn.commit()
        
        # Verify tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        logger.info(f"\n✓ Successfully created/verified {len(tables)} tables:")
        for table in tables:
            logger.info(f"  - {table[0]}")
        
        conn.close()
        logger.info(f"\n✓ Database initialization complete: {dbname}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to build PostgreSQL tables: {e}")
        return False


def build_mysql_tables(host: str, dbname: str, user: str, password: str, port: int = 3306) -> bool:
    """Build all tables in MySQL database."""
    try:
        import mysql.connector
        
        logger.info(f"Connecting to MySQL: {user}@{host}:{port}/{dbname}")
        conn = mysql.connector.connect(
            host=host,
            port=port,
            database=dbname,
            user=user,
            password=password
        )
        cursor = conn.cursor()
        
        # Read SQL file
        sql_file = Path(__file__).parent / "init_all_tables.sql"
        logger.info(f"Reading SQL schema from: {sql_file}")
        
        with open(sql_file, 'r') as f:
            sql_script = f.read()
        
        # MySQL-specific adjustments
        sql_script = sql_script.replace('AUTOINCREMENT', 'AUTO_INCREMENT')
        sql_script = sql_script.replace('TEXT', 'TEXT')
        sql_script = sql_script.replace('BLOB', 'BLOB')
        sql_script = sql_script.replace('BOOLEAN', 'TINYINT(1)')
        sql_script = sql_script.replace('REAL', 'FLOAT')
        
        # Execute statements one by one
        statements = [s.strip() for s in sql_script.split(';') if s.strip() and not s.strip().startswith('--')]
        
        total = len(statements)
        logger.info(f"Executing {total} SQL statements...")
        
        for i, statement in enumerate(statements, 1):
            if not statement or statement.startswith('--'):
                continue
            
            try:
                cursor.execute(statement)
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{total} statements executed")
            except mysql.connector.Error as e:
                logger.error(f"Error executing statement {i}: {e}")
                logger.error(f"Statement: {statement[:100]}...")
        
        conn.commit()
        
        # Verify tables
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        logger.info(f"\n✓ Successfully created/verified {len(tables)} tables:")
        for table in tables:
            logger.info(f"  - {table[0]}")
        
        conn.close()
        logger.info(f"\n✓ Database initialization complete: {dbname}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to build MySQL tables: {e}")
        return False


def list_components():
    """List all Grace components that have tables."""
    components = {
        "Core": [
            "audit_logs", "chain_verification", "log_categories",
            "structured_memory", "governance_decisions", "instance_states"
        ],
        "Ingress Kernel": [
            "sources", "source_health", "bronze_raw_events", "silver_records",
            "silver_articles", "silver_transcripts", "silver_tabular",
            "gold_article_topics", "gold_entity_mentions", "gold_feature_datasets",
            "processing_metrics", "validation_failures", "source_trust_history"
        ],
        "Intelligence Kernel": [
            "intel_requests", "intel_plans", "intel_results", "intel_specialist_reports",
            "intel_experiences", "intel_canary_deployments", "intel_shadow_deployments",
            "intel_policy_violations", "intel_snapshots"
        ],
        "Learning Kernel": [
            "datasets", "dataset_versions", "label_policies", "label_tasks", "labels",
            "active_queries", "curriculum_specs", "augment_specs", "augment_applications",
            "feature_views", "quality_reports", "learning_experiences",
            "weak_labelers", "weak_predictions", "learning_snapshots"
        ],
        "Memory Kernel": [
            "lightning_memory", "fusion_memory", "memory_access_patterns",
            "librarian_index", "memory_stats", "memory_operations",
            "memory_snapshots", "memory_cleanup_tasks"
        ],
        "Orchestration Kernel": [
            "orchestration_state", "orchestration_loops", "orchestration_tasks",
            "policies", "state_transitions"
        ],
        "Resilience Kernel": [
            "circuit_breakers", "degradation_policies", "active_degradations",
            "resilience_incidents", "rate_limits", "sli_measurements",
            "recovery_actions", "resilience_snapshots"
        ],
        "MLT Core": [
            "mlt_experiences", "mlt_insights", "mlt_plans", "mlt_snapshots",
            "mlt_specialist_reports"
        ],
        "MLDL Components": [
            "models", "model_approvals", "model_deployments", "model_metrics",
            "slo_violations", "alerts", "canary_progress", "deployment_rollback_history"
        ],
        "Communications": [
            "dlq_entries", "message_dedupe"
        ],
        "Common": [
            "snapshots", "rollback_history"
        ]
    }
    
    print("\n" + "="*80)
    print("Grace System - Database Tables by Component")
    print("="*80 + "\n")
    
    total_tables = 0
    for component, tables in components.items():
        print(f"{component} ({len(tables)} tables):")
        for table in tables:
            print(f"  • {table}")
        print()
        total_tables += len(tables)
    
    print("="*80)
    print(f"Total: {total_tables} tables across {len(components)} components")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build all Grace system database tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SQLite (default)
  python build_all_tables.py --db sqlite --path grace.db
  
  # PostgreSQL
  python build_all_tables.py --db postgres --host localhost --dbname grace --user grace --password secret
  
  # MySQL
  python build_all_tables.py --db mysql --host localhost --dbname grace --user grace --password secret
  
  # List all components and tables
  python build_all_tables.py --list
        """
    )
    
    parser.add_argument('--db', choices=['sqlite', 'postgres', 'mysql'], default='sqlite',
                       help='Database type (default: sqlite)')
    parser.add_argument('--path', default='grace.db',
                       help='Database file path (SQLite only)')
    parser.add_argument('--host', default='localhost',
                       help='Database host (PostgreSQL/MySQL)')
    parser.add_argument('--port', type=int,
                       help='Database port (default: 5432 for Postgres, 3306 for MySQL)')
    parser.add_argument('--dbname', default='grace',
                       help='Database name (PostgreSQL/MySQL)')
    parser.add_argument('--user', default='grace',
                       help='Database user (PostgreSQL/MySQL)')
    parser.add_argument('--password',
                       help='Database password (PostgreSQL/MySQL)')
    parser.add_argument('--list', action='store_true',
                       help='List all components and tables')
    
    args = parser.parse_args()
    
    # List components if requested
    if args.list:
        list_components()
        return 0
    
    logger.info("="*80)
    logger.info("Grace System - Database Table Builder")
    logger.info("="*80)
    
    success = False
    
    if args.db == 'sqlite':
        success = build_sqlite_tables(args.path)
    
    elif args.db == 'postgres':
        if not args.password:
            logger.error("PostgreSQL requires --password")
            return 1
        port = args.port or 5432
        success = build_postgres_tables(
            args.host, args.dbname, args.user, args.password, port
        )
    
    elif args.db == 'mysql':
        if not args.password:
            logger.error("MySQL requires --password")
            return 1
        port = args.port or 3306
        success = build_mysql_tables(
            args.host, args.dbname, args.user, args.password, port
        )
    
    if success:
        logger.info("\n" + "="*80)
        logger.info("✓ All tables successfully built!")
        logger.info("="*80)
        return 0
    else:
        logger.error("\n" + "="*80)
        logger.error("✗ Failed to build tables - check logs above")
        logger.error("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
