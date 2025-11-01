#!/usr/bin/env python3
"""
Build Complete Grace Database with ALL Schema Files
====================================================
This script builds a complete database including all schema files:
- init_all_tables.sql (base 82 tables)
- init_security_selfawareness_tables.sql (security & consciousness)
- init_meta_loop_tables.sql (meta-learning & OODA loop)
"""

import sqlite3
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_complete_database(db_path: str = "database/grace_complete.sqlite3") -> bool:
    """Build complete Grace database with all schema files."""
    try:
        # Remove existing database
        db_file = Path(db_path)
        if db_file.exists():
            db_file.unlink()
            logger.info(f"Removed existing database: {db_path}")
        
        # Create new database
        logger.info(f"Creating complete Grace database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Schema files in execution order
        schema_files = [
            Path("database/init_all_tables.sql"),
            Path("database/init_security_selfawareness_tables.sql"),
            Path("database/init_meta_loop_tables.sql")
        ]
        
        total_executed = 0
        
        for schema_file in schema_files:
            if not schema_file.exists():
                logger.warning(f"Schema file not found: {schema_file}")
                continue
                
            logger.info(f"Executing schema: {schema_file}")
            
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    sql_content = f.read()
                
                # Split into individual statements and execute
                statements = [s.strip() for s in sql_content.split(';') if s.strip() and not s.strip().startswith('--')]
                
                for i, statement in enumerate(statements):
                    if not statement or statement.startswith('--'):
                        continue
                    
                    try:
                        cursor.execute(statement + ';')
                        total_executed += 1
                    except sqlite3.Error as e:
                        if "already exists" not in str(e):
                            logger.debug(f"Statement {i+1} in {schema_file.name}: {e}")
                
                conn.commit()
                logger.info(f"✓ Successfully executed {schema_file.name}")
                
            except Exception as e:
                logger.error(f"Error executing {schema_file}: {e}")
                return False
        
        # Get final table count
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        logger.info(f"\n{'='*80}")
        logger.info("✓ COMPLETE GRACE DATABASE CREATED")
        logger.info(f"✓ Total tables: {len(tables)}")
        logger.info(f"✓ SQL statements executed: {total_executed}")
        logger.info(f"✓ Database file: {db_path}")
        logger.info(f"{'='*80}")
        
        # Show table breakdown by category
        logger.info("\nTable Categories:")
        
        categories = {
            "Core": ["audit_logs", "chain_verification", "log_categories", "structured_memory", "governance_decisions", "instance_states"],
            "Ingress": [t for t in table_names if any(prefix in t for prefix in ["source", "bronze_", "silver_", "gold_", "processing_", "validation_"])],
            "Intelligence": [t for t in table_names if t.startswith("intel_")],
            "Learning": [t for t in table_names if any(prefix in t for prefix in ["dataset", "label_", "active_queries", "curriculum", "augment", "feature_", "quality_", "learning_", "weak_"])],
            "Memory": [t for t in table_names if any(prefix in t for prefix in ["lightning_", "fusion_", "memory_", "librarian_"])],
            "Orchestration": [t for t in table_names if any(prefix in t for prefix in ["orchestration_", "policies", "state_"])],
            "Resilience": [t for t in table_names if any(prefix in t for prefix in ["circuit_", "degradation", "resilience_", "rate_", "sli_", "recovery_"])],
            "MLT": [t for t in table_names if t.startswith("mlt_")],
            "MLDL": [t for t in table_names if any(prefix in t for prefix in ["model", "slo_", "alerts", "canary_", "deployment_"])],
            "Security": [t for t in table_names if any(prefix in t for prefix in ["crypto_", "api_key", "parliament_", "quorum_"])],
            "Consciousness": [t for t in table_names if any(prefix in t for prefix in ["self_", "system_", "goal_", "value_", "consciousness_", "uncertainty_"])],
            "Meta-Loop": [t for t in table_names if any(prefix in t for prefix in ["observation", "orientation", "decision", "action", "evaluation", "reflection", "evolution_", "trust_", "ethics_", "knowledge_", "meta_loop_"])],
            "Communications": [t for t in table_names if any(prefix in t for prefix in ["dlq_", "message_"])],
            "Common": [t for t in table_names if t in ["snapshots", "rollback_history", "immutable_entries"]]
        }
        
        categorized = set()
        for category, patterns in categories.items():
            matching = [t for t in table_names if t in patterns or any(t.startswith(p.rstrip('_')) for p in patterns if p.endswith('_'))]
            if matching:
                logger.info(f"  {category}: {len(matching)} tables")
                categorized.update(matching)
        
        # Show uncategorized tables
        uncategorized = [t for t in table_names if t not in categorized]
        if uncategorized:
            logger.info(f"  Other: {len(uncategorized)} tables - {', '.join(uncategorized)}")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to build complete database: {e}")
        return False

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "database/grace_complete.sqlite3"
    success = build_complete_database(db_path)
    sys.exit(0 if success else 1)