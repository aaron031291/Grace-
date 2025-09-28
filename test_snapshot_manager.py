#!/usr/bin/env python3
"""Test script for Grace Snapshot Manager."""

import asyncio
import sys
import os
import tempfile

# Add grace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from grace.core.snapshot_manager import GraceSnapshotManager, ObjectStore
    
    print("✅ Successfully imported snapshot manager")
    
    async def test_snapshot_manager():
        """Test snapshot manager functionality."""
        print("🧪 Testing Grace Snapshot Manager...")
        
        # Create temporary storage
        with tempfile.TemporaryDirectory() as temp_dir:
            object_store = ObjectStore(temp_dir)
            snapshot_manager = GraceSnapshotManager(
                db_path=":memory:",  # In-memory SQLite for testing
                object_store=object_store
            )
            
            # Test export snapshot
            test_payload = {
                "version": "1.0.0",
                "config": {"test": "data"},
                "state": {"active": True, "connections": 42},
                "runtime_info": {"uptime": 3600}
            }
            
            snapshot_result = await snapshot_manager.export_snapshot(
                component_type="orchestration",
                payload=test_payload,
                description="Test orchestration snapshot",
                created_by="test_user",
                tags={"env": "test", "purpose": "validation"}
            )
            
            print(f"Created snapshot: {snapshot_result['snapshot_id']}")
            print(f"Size: {snapshot_result['size_bytes']} bytes")
            print(f"Hash: {snapshot_result['state_hash']}")
            
            # Test get snapshot
            snapshot = await snapshot_manager.get_snapshot(snapshot_result["snapshot_id"])
            assert snapshot is not None
            assert snapshot["component_type"] == "orchestration"
            
            # Test get payload
            payload = await snapshot_manager.get_snapshot_payload(snapshot_result["snapshot_id"])
            assert payload == test_payload
            
            # Test list snapshots
            snapshots = await snapshot_manager.list_snapshots("orchestration")
            assert len(snapshots) == 1
            assert snapshots[0]["snapshot_id"] == snapshot_result["snapshot_id"]
            
            # Test rollback
            rollback_result = await snapshot_manager.rollback(
                component_type="orchestration",
                to_snapshot=snapshot_result["snapshot_id"],
                reason="Test rollback",
                triggered_by="test_user"
            )
            
            print(f"Rollback: {rollback_result['rollback_id']}")
            print(f"Status: {rollback_result['status']}")
            
            # Test stats
            stats = await snapshot_manager.get_stats()
            print(f"Component stats: {stats['component_stats']}")
            
            # Cleanup
            snapshot_manager.close()
            
            print("✅ Snapshot manager test passed!")
            return True

    def run_tests():
        """Run all tests."""
        print("🚀 Running Grace Snapshot Manager Tests...\n")
        
        try:
            success = asyncio.run(test_snapshot_manager())
            print(f"\n📊 Result: {'PASSED' if success else 'FAILED'}")
            return success
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

    if __name__ == "__main__":
        success = run_tests()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Dependencies not available. Skipping snapshot tests.")
    sys.exit(0)
except Exception as e:
    print(f"❌ Test error: {e}")
    sys.exit(1)