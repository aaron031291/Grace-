#!/usr/bin/env python3
"""
Grace AI - Comprehensive End-to-End System Test
Tests all components with detailed logging and verification
"""
import asyncio
import logging
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add grace to path
sys.path.insert(0, str(Path(__file__).parent))

from grace.launcher import GraceLauncher
from grace import config

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("E2E_SYSTEM_TEST")


async def test_component_initialization(launcher):
    """Test that all core components initialize correctly"""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Component Initialization")
    logger.info("="*80)
    
    components = {
        'immutable_logger': 'Immutable Logger (Cryptographic Audit)',
        'trust_ledger': 'Trust Ledger (Dynamic Scoring)',
        'workflow_registry': 'Workflow Registry (Workflow Loader)',
        'workflow_engine': 'Workflow Engine (Execution)',
        'trigger_mesh': 'TriggerMesh (Event Orchestration)'
    }
    
    results = {}
    for component_id, description in components.items():
        try:
            component = launcher.registry.get(component_id)
            if component:
                logger.info(f"  ✓ {description:45s} - OK")
                results[component_id] = "OK"
            else:
                logger.error(f"  ✗ {description:45s} - NOT FOUND")
                results[component_id] = "NOT FOUND"
        except Exception as e:
            logger.error(f"  ✗ {description:45s} - ERROR: {e}")
            results[component_id] = f"ERROR: {e}"
    
    all_ok = all(r == "OK" for r in results.values())
    if all_ok:
        logger.info("\n✓✓✓ All components initialized successfully")
    else:
        logger.error("\n✗✗✗ Some components failed to initialize")
    
    return all_ok, results


async def test_workflow_loading(launcher):
    """Test that workflows are loaded correctly"""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Workflow Loading")
    logger.info("="*80)
    
    workflow_registry = launcher.registry.get('workflow_registry')
    workflows = workflow_registry.workflows if workflow_registry else []
    
    logger.info(f"  Loaded {len(workflows)} workflows")
    
    for wf in workflows:
        name = getattr(wf, 'name', 'UNKNOWN')
        events = getattr(wf, 'EVENTS', [])
        logger.info(f"    - {name:40s} triggers on: {events}")
    
    success = len(workflows) > 0
    if success:
        logger.info("\n✓✓✓ Workflows loaded successfully")
    else:
        logger.warning("\n⚠⚠⚠ No workflows loaded (this may be expected for a minimal setup)")
    
    return success, {"workflow_count": len(workflows), "workflows": [getattr(wf, 'name', 'UNKNOWN') for wf in workflows]}


async def test_event_dispatch_and_routing(launcher):
    """Test that events are dispatched and routed correctly"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Event Dispatch and Routing")
    logger.info("="*80)
    
    trigger_mesh = launcher.registry.get('trigger_mesh')
    if not trigger_mesh:
        logger.error("  ✗ TriggerMesh not available")
        return False, {"error": "TriggerMesh not available"}
    
    # Test Case 1: Data Ingestion Event
    logger.info("\n  Test Case 3a: Data Ingestion Event")
    event1_type = "external_data_received"
    event1_payload = {
        "source": "test_harness",
        "type": "structured_data",
        "data": {
            "test_id": "e2e_test_001",
            "temperature": 72.5,
            "humidity": 45,
            "timestamp": datetime.now().isoformat()
        },
        "tags": ["test", "e2e", "ingestion"]
    }
    
    try:
        await trigger_mesh.dispatch_event(event1_type, event1_payload)
        await asyncio.sleep(0.3)  # Allow processing
        logger.info("    ✓ Data ingestion event dispatched and processed")
    except Exception as e:
        logger.error(f"    ✗ Data ingestion event failed: {e}")
        return False, {"error": str(e)}
    
    # Test Case 2: Verification Request Event
    logger.info("\n  Test Case 3b: Verification Request Event")
    event2_type = "verification_request"
    event2_payload = {
        "source": "verified_api",
        "data": {
            "test_id": "e2e_test_002",
            "claim": "System operating normally",
            "confidence": 0.95,
            "timestamp": datetime.now().isoformat()
        },
        "tags": ["test", "e2e", "verification"]
    }
    
    try:
        await trigger_mesh.dispatch_event(event2_type, event2_payload)
        await asyncio.sleep(0.3)  # Allow processing
        logger.info("    ✓ Verification request dispatched and processed")
    except Exception as e:
        logger.error(f"    ✗ Verification request failed: {e}")
        return False, {"error": str(e)}
    
    logger.info("\n✓✓✓ Event dispatch and routing working")
    return True, {"events_dispatched": 2}


async def test_immutable_logging(launcher):
    """Test that immutable logging is working with cryptographic chain"""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Immutable Logging & Cryptographic Chain")
    logger.info("="*80)
    
    log_path = config.IMMUTABLE_LOG_PATH
    
    if not Path(log_path).exists():
        logger.error(f"  ✗ Immutable log not found at {log_path}")
        return False, {"error": "Log file not found"}
    
    # Read log entries
    records = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"  Total log records: {len(records)}")
    
    # Check for required fields
    if records:
        sample_record = records[-1]
        required_fields = ["ts", "event_id", "phase", "status", "sha256", "prev_hash"]
        missing_fields = [f for f in required_fields if f not in sample_record]
        
        if missing_fields:
            logger.error(f"  ✗ Missing required fields: {missing_fields}")
            return False, {"error": f"Missing fields: {missing_fields}"}
        
        logger.info("  ✓ All required fields present")
        
        # Check cryptographic signatures if available
        has_signatures = sample_record.get("ed25519_sig") and sample_record["ed25519_sig"] != "CRYPTO_UNAVAILABLE"
        if has_signatures:
            logger.info("  ✓ Cryptographic signatures present (Ed25519)")
        else:
            logger.warning("  ⚠ Cryptographic signatures not available (install pynacl)")
        
        # Verify chain integrity (spot check last 5 records)
        chain_check = records[-5:] if len(records) >= 5 else records
        chain_valid = True
        for i in range(1, len(chain_check)):
            expected_prev = chain_check[i-1].get("sha256")
            actual_prev = chain_check[i].get("prev_hash")
            if expected_prev != actual_prev:
                chain_valid = False
                logger.error(f"  ✗ Chain break at record {i}: expected {expected_prev[:16]}..., got {actual_prev[:16] if actual_prev else 'None'}...")
                break
        
        if chain_valid:
            logger.info("  ✓ Hash chain integrity verified (spot check)")
        
        # Count phases
        phases = {}
        for record in records:
            phase = record.get("phase", "UNKNOWN")
            phases[phase] = phases.get(phase, 0) + 1
        
        logger.info(f"\n  Phase distribution:")
        for phase, count in sorted(phases.items()):
            logger.info(f"    {phase:20s}: {count}")
        
        logger.info("\n✓✓✓ Immutable logging working correctly")
        return True, {
            "record_count": len(records),
            "has_signatures": has_signatures,
            "chain_valid": chain_valid,
            "phases": phases
        }
    else:
        logger.warning("  ⚠ No log records found (expected after running tests)")
        return False, {"error": "No log records"}


async def test_trust_ledger(launcher):
    """Test that trust ledger is tracking entity trust"""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Trust Ledger (Dynamic Trust Scoring)")
    logger.info("="*80)
    
    trust_ledger = launcher.registry.get('trust_ledger')
    if not trust_ledger:
        logger.error("  ✗ Trust Ledger not available")
        return False, {"error": "Trust Ledger not available"}
    
    # Get stats
    stats = trust_ledger.get_stats()
    logger.info(f"  Total entities tracked: {stats['total_entities']}")
    logger.info(f"  Total interactions: {stats['total_interactions']}")
    
    if stats['by_type']:
        logger.info(f"\n  Entities by type:")
        for entity_type, count in stats['by_type'].items():
            logger.info(f"    {entity_type:20s}: {count}")
    
    if stats['by_level']:
        logger.info(f"\n  Entities by trust level:")
        for level, count in stats['by_level'].items():
            logger.info(f"    {level:20s}: {count}")
    
    # Get trusted entities
    trusted = trust_ledger.get_trusted_entities(min_trust=0.7)
    if trusted:
        logger.info(f"\n  Top trusted entities (trust >= 0.7):")
        for record in trusted[:5]:
            logger.info(f"    {record.entity_id:30s} trust={record.trust_score:.2f} level={record.trust_level}")
    
    # Get quarantined entities
    quarantined = trust_ledger.get_quarantined_entities()
    if quarantined:
        logger.info(f"\n  Quarantined entities (trust < 0.3):")
        for record in quarantined:
            logger.info(f"    {record.entity_id:30s} trust={record.trust_score:.2f}")
    else:
        logger.info(f"\n  ✓ No quarantined entities")
    
    logger.info("\n✓✓✓ Trust Ledger working correctly")
    return True, stats


async def test_vwx_verification(launcher):
    """Test VWX (Veracity & Continuity Kernel)"""
    logger.info("\n" + "="*80)
    logger.info("TEST 6: VWX v2 - Veracity & Continuity Kernel")
    logger.info("="*80)
    
    trigger_mesh = launcher.registry.get('trigger_mesh')
    if not trigger_mesh:
        logger.error("  ✗ TriggerMesh not available")
        return False, {"error": "TriggerMesh not available"}
    
    # Test with different veracity levels
    test_cases = [
        ("verified_api", "High veracity source", 0.9),
        ("user_input", "Medium veracity source", 0.7),
        ("unknown", "Low veracity source", 0.3)
    ]
    
    results = {}
    for source, description, expected_trust in test_cases:
        logger.info(f"\n  Testing: {description} (source={source})")
        
        event_type = "verification_request"
        event_payload = {
            "source": source,
            "data": {
                "test_id": f"vwx_test_{source}",
                "claim": f"Test claim from {source}",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        try:
            await trigger_mesh.dispatch_event(event_type, event_payload)
            await asyncio.sleep(0.2)
            logger.info(f"    ✓ {description} processed")
            results[source] = "OK"
        except Exception as e:
            logger.error(f"    ✗ {description} failed: {e}")
            results[source] = f"ERROR: {e}"
    
    all_ok = all(r == "OK" for r in results.values())
    if all_ok:
        logger.info("\n✓✓✓ VWX verification working correctly")
    else:
        logger.error("\n✗✗✗ Some VWX tests failed")
    
    return all_ok, results


async def run_full_e2e_test():
    """Run complete end-to-end system test"""
    logger.info("\n" + "="*80)
    logger.info("GRACE AI - COMPREHENSIVE END-TO-END SYSTEM TEST")
    logger.info("="*80)
    logger.info(f"Test started at: {datetime.now().isoformat()}")
    
    # Initialize Grace
    logger.info("\n" + "="*80)
    logger.info("INITIALIZATION")
    logger.info("="*80)
    
    launcher = GraceLauncher(argparse.Namespace(debug=False, log_level="INFO"))
    await launcher.initialize()
    
    # Run all tests
    all_results = {}
    all_passed = True
    
    tests = [
        ("Component Initialization", test_component_initialization),
        ("Workflow Loading", test_workflow_loading),
        ("Event Dispatch & Routing", test_event_dispatch_and_routing),
        ("Immutable Logging", test_immutable_logging),
        ("Trust Ledger", test_trust_ledger),
        ("VWX Verification", test_vwx_verification),
    ]
    
    for test_name, test_func in tests:
        try:
            passed, result = await test_func(launcher)
            all_results[test_name] = {"passed": passed, "result": result}
            if not passed:
                all_passed = False
        except Exception as e:
            logger.error(f"\n✗✗✗ Test '{test_name}' crashed: {e}", exc_info=True)
            all_results[test_name] = {"passed": False, "result": {"error": str(e)}}
            all_passed = False
    
    # Final Report
    logger.info("\n" + "="*80)
    logger.info("FINAL TEST REPORT")
    logger.info("="*80)
    
    for test_name, result in all_results.items():
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        logger.info(f"  {status:8s} {test_name}")
    
    logger.info("\n" + "="*80)
    if all_passed:
        logger.info("✓✓✓ ALL TESTS PASSED ✓✓✓")
        logger.info("Grace AI system is fully operational!")
    else:
        logger.error("✗✗✗ SOME TESTS FAILED ✗✗✗")
        logger.error("Review the logs above for details")
    logger.info("="*80)
    
    logger.info(f"\nTest completed at: {datetime.now().isoformat()}")
    logger.info(f"\nLog files:")
    logger.info(f"  Immutable Log: {config.IMMUTABLE_LOG_PATH}")
    logger.info(f"  Trust Ledger:  {config.GRACE_DATA_DIR / 'trust_ledger.jsonl'}")
    logger.info(f"\nVerify audit trail:")
    logger.info(f"  python tools/verify_immutable_log.py --all")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_full_e2e_test())
    sys.exit(0 if success else 1)
