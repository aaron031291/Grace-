#!/usr/bin/env python3
"""
Verify Grace Database Schema
============================
Quick verification script to check database integrity and list all tables.
"""

import sqlite3
import sys
from pathlib import Path

def verify_database(db_path: str):
    """Verify database structure and integrity."""
    print(f"\n{'='*80}")
    print(f"Grace Database Verification: {db_path}")
    print(f"{'='*80}\n")
    
    if not Path(db_path).exists():
        print(f"❌ Database file not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"✓ Found {len(tables)} tables\n")
        
        # Group tables by component
        components = {
            "Core": ["audit_logs", "chain_verification", "log_categories", 
                    "structured_memory", "governance_decisions", "instance_states"],
            "Ingress": ["sources", "source_health", "bronze_raw_events", "silver_records",
                       "silver_articles", "silver_transcripts", "silver_tabular",
                       "gold_article_topics", "gold_entity_mentions", "gold_feature_datasets",
                       "processing_metrics", "validation_failures", "source_trust_history"],
            "Intelligence": ["intel_requests", "intel_plans", "intel_results",
                           "intel_specialist_reports", "intel_experiences",
                           "intel_canary_deployments", "intel_shadow_deployments",
                           "intel_policy_violations", "intel_snapshots"],
            "Learning": ["datasets", "dataset_versions", "label_policies", "label_tasks",
                        "labels", "active_queries", "curriculum_specs", "augment_specs",
                        "augment_applications", "feature_views", "quality_reports",
                        "learning_experiences", "weak_labelers", "weak_predictions",
                        "learning_snapshots"],
            "Memory": ["lightning_memory", "fusion_memory", "memory_access_patterns",
                      "librarian_index", "memory_stats", "memory_operations",
                      "memory_snapshots", "memory_cleanup_tasks"],
            "Orchestration": ["orchestration_state", "orchestration_loops",
                            "orchestration_tasks", "policies", "state_transitions"],
            "Resilience": ["circuit_breakers", "degradation_policies", "active_degradations",
                         "resilience_incidents", "rate_limits", "sli_measurements",
                         "recovery_actions", "resilience_snapshots"],
            "MLT": ["mlt_experiences", "mlt_insights", "mlt_plans", "mlt_snapshots",
                   "mlt_specialist_reports"],
            "MLDL": ["models", "model_approvals", "model_deployments", "model_metrics",
                    "slo_violations", "alerts", "canary_progress", "deployment_rollback_history"],
            "Communications": ["dlq_entries", "message_dedupe"],
            "Common": ["snapshots", "rollback_history"]
        }
        
        # Check each component
        all_ok = True
        for component, expected_tables in components.items():
            present = [t for t in expected_tables if t in tables]
            missing = [t for t in expected_tables if t not in tables]
            
            status = "✓" if len(missing) == 0 else "⚠"
            print(f"{status} {component}: {len(present)}/{len(expected_tables)} tables")
            
            if missing:
                print(f"  Missing: {', '.join(missing)}")
                all_ok = False
        
        print()
        
        # Get row counts for key tables
        print("Row counts (sample):")
        key_tables = ["audit_logs", "sources", "intel_requests", "datasets", 
                     "orchestration_loops", "circuit_breakers"]
        for table in key_tables:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  {table}: {count} rows")
        
        print()
        
        # Check database integrity
        print("Integrity checks:")
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        if result == "ok":
            print("  ✓ Database integrity: OK")
        else:
            print(f"  ❌ Database integrity: {result}")
            all_ok = False
        
        # Check foreign key integrity
        cursor.execute("PRAGMA foreign_key_check")
        fk_errors = cursor.fetchall()
        if not fk_errors:
            print("  ✓ Foreign key integrity: OK")
        else:
            print(f"  ❌ Foreign key violations: {len(fk_errors)}")
            for error in fk_errors[:5]:  # Show first 5
                print(f"    {error}")
            all_ok = False
        
        # Database statistics
        print("\nDatabase statistics:")
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        size_bytes = cursor.fetchone()[0]
        size_mb = size_bytes / (1024 * 1024)
        print(f"  Size: {size_mb:.2f} MB")
        
        conn.close()
        
        print(f"\n{'='*80}")
        if all_ok:
            print("✓ Database verification PASSED")
        else:
            print("⚠ Database verification completed with warnings")
        print(f"{'='*80}\n")
        
        return all_ok
        
    except Exception as e:
        print(f"\n❌ Error verifying database: {e}\n")
        return False

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "grace_system.db"
    success = verify_database(db_path)
    sys.exit(0 if success else 1)
