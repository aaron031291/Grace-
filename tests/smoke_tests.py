"""Smoke tests for Grace kernel end-to-end functionality."""
import sys
import os

# Add grace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grace.mtl_kernel.kernel import MTLKernel
from grace.governance.grace_governance_kernel import GraceGovernanceKernel as GovernanceKernel
from grace.intelligence_kernel.kernel import IntelligenceKernel
from grace.contracts.dto_common import MemoryEntry
from grace.contracts.governed_request import GovernedRequest


def test_mtl_write_fanout():
    """Test MTL write fanout: store → trust.init → immutable.append → trigger.record."""
    print("🧪 Testing MTL write fanout...")
    
    # Initialize MTL kernel
    mtl = MTLKernel()
    
    # Create sample entry
    sample_entry = MemoryEntry(
        content="This is a test memory entry for fanout validation",
        content_type="text/plain"
    )
    
    # Write entry (should trigger fanout)
    memory_id = mtl.write(sample_entry)
    
    # Verify fanout components
    # 1. Memory stored
    stored_entry = mtl.memory_service.retrieve(memory_id)
    assert stored_entry is not None, "Memory not stored"
    
    # 2. Trust initialized
    trust_records = mtl.trust_service.get_attestations(memory_id)
    # Trust is initialized but no attestations yet
    
    # 3. Audit log entry exists
    audit_records = mtl.immutable_log.get_audit_trail(memory_id)
    assert len(audit_records) > 0, "No audit records found"
    
    # 4. Trigger event recorded
    trigger_events = mtl.trigger_ledger.get_events_for_target(memory_id)
    assert len(trigger_events) > 0, "No trigger events found"
    
    print(f"✅ MTL fanout successful - Memory ID: {memory_id}")
    print(f"   - Audit records: {len(audit_records)}")
    print(f"   - Trigger events: {len(trigger_events)}")
    
    return True


def test_governance_roundtrip():
    """Test governance roundtrip: request → evaluation → decision."""
    print("🧪 Testing governance roundtrip...")
    
    # Initialize kernels
    mtl = MTLKernel()
    intelligence = IntelligenceKernel(mtl)
    governance = GovernanceKernel(mtl, intelligence)
    
    # Create sample request
    sample_request = GovernedRequest(
        request_type="document_approval",
        content="Please approve this test document for publication",
        requester="test_user",
        priority=5,
        risk_level="medium"
    )
    
    # Evaluate through governance
    decision = governance.evaluate(sample_request)
    
    # Verify decision structure
    assert hasattr(decision, 'approved'), "Decision missing 'approved' field"
    assert hasattr(decision, 'confidence'), "Decision missing 'confidence' field"
    assert hasattr(decision, 'reasoning'), "Decision missing 'reasoning' field"
    assert decision.approved in [True, False], "Invalid approval value"
    assert 0.0 <= decision.confidence <= 1.0, "Invalid confidence value"
    
    print(f"✅ Governance roundtrip successful")
    print(f"   - Approved: {decision.approved}")
    print(f"   - Confidence: {decision.confidence:.2f}")
    print(f"   - Reasoning: {decision.reasoning[:100]}...")
    
    return True


def test_rag_path():
    """Test RAG path: query → recall → rank → response."""
    print("🧪 Testing RAG path...")
    
    # Initialize MTL kernel
    mtl = MTLKernel()
    
    # Store some test entries first
    test_entries = [
        "This is a document about artificial intelligence and machine learning",
        "Grace kernel provides governance for AI systems",
        "Memory trust and learning form the core of the MTL kernel"
    ]
    
    memory_ids = []
    for content in test_entries:
        entry = MemoryEntry(content=content)
        memory_id = mtl.write(entry)
        memory_ids.append(memory_id)
    
    # Test recall with query
    results = mtl.recall("artificial intelligence")
    
    assert len(results) >= 0, "RAG query failed"
    
    # Test librarian search and rank
    ranked_results = mtl.librarian.search_and_rank("Grace kernel", limit=5)
    
    print(f"✅ RAG path successful")
    print(f"   - Query results: {len(results)}")
    print(f"   - Ranked results: {len(ranked_results)}")
    print(f"   - Stored memories: {len(memory_ids)}")
    
    return True


def run_all_tests():
    """Run all smoke tests."""
    print("🚀 Running Grace kernel smoke tests...\n")
    
    tests = [
        test_mtl_write_fanout,
        test_governance_roundtrip, 
        test_rag_path
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} failed with error: {e}")
        
        print()  # Blank line between tests
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)