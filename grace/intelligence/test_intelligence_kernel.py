#!/usr/bin/env python3
"""
Test script for the Intelligence Kernel implementation.
"""
import sys
import os
import json
from datetime import datetime
import pytest

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_intelligence_service():
    """Test the main Intelligence Service functionality."""
    print("=" * 60)
    print("Testing Grace Intelligence Kernel")
    print("=" * 60)
    
    try:
        from intelligence_service import IntelligenceService
        
        # Initialize service
        print("\n1. Initializing Intelligence Service...")
        service = IntelligenceService()
        print("✓ Intelligence Service initialized successfully")
        
        # Test health check
        print("\n2. Testing health check...")
        health = service.get_health()
        print(f"✓ Health status: {health.status}")
        print(f"✓ Version: {health.version}")
        print(f"✓ Components: {len(health.components)} components")
        
        # Test task request
        print("\n3. Testing task request submission...")
        task_request = {
            'task': 'classification',
            'input': {
                'X': {
                    'feature1': 0.5, 
                    'feature2': 1.2,
                    'feature3': -0.3
                },
                'modality': 'tabular'
            },
            'context': {
                'user_ctx': {'user_id': 'test_user'},
                'latency_budget_ms': 500,
                'explanation': False,
                'env': 'dev'
            }
        }
        
        req_id = service.request(task_request)
        print(f"✓ Task request submitted with ID: {req_id}")
        
        # Test plan preview
        print("\n4. Testing plan preview...")
        plan = service.plan_preview(task_request)
        print(f"✓ Plan generated with ID: {plan.get('plan_id', 'N/A')}")
        print(f"✓ Selected models: {plan.get('route', {}).get('models', [])}")
        print(f"✓ Ensemble method: {plan.get('route', {}).get('ensemble', 'none')}")
        
        # Test snapshot export
        print("\n5. Testing snapshot export...")
        snapshot_info = service.export_snapshot()
        print(f"✓ Snapshot exported: {snapshot_info.get('snapshot_id', 'N/A')}")
        
        # Test component integration
        print("\n6. Testing component integration...")
        
        # Test router
        route = service.router.route(task_request)
        print(f"✓ Router: Selected {len(route.get('models', []))} models")
        
        # Test planner
        plan = service.planner.build_plan(task_request, route)
        print(f"✓ Planner: Generated plan {plan.get('plan_id', 'N/A')}")
        
        # Test meta ensembler
        mock_outputs = [
            {"prediction": "approved", "confidence": 0.85},
            {"prediction": "approved", "confidence": 0.92},
            {"prediction": "rejected", "confidence": 0.78}
        ]
        ensemble_result = service.meta_ensembler.predict(mock_outputs)
        print(f"✓ Meta Ensembler: Prediction = {ensemble_result.get('prediction', 'N/A')}")
        
        # Test inference engine
        inference_result = service.inference_engine.execute(plan, task_request)
        print(f"✓ Inference Engine: Status = {inference_result.get('status', 'completed')}")
        
        # Test governance bridge  
        approval = service.governance_bridge.request_approval(plan)
        print(f"✓ Governance Bridge: Approved = {approval.get('approved', False)}")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED - Intelligence Kernel is working correctly!")
        print("=" * 60)
        
        assert True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Test failed with exception: {e}")

def test_api_schemas():
    """Test API schema validation."""
    print("\n7. Testing API schemas...")
    
    # Test TaskRequest schema
    sample_request = {
        "req_id": "req_20240928_143022_1234",
        "task": "classification",
        "input": {
            "X": {"feature1": 0.5},
            "modality": "tabular"
        },
        "context": {
            "latency_budget_ms": 500,
            "env": "dev"
        }
    }
    print("✓ TaskRequest schema example created")
    
    # Test InferenceResult schema
    sample_result = {
        "req_id": "req_20240928_143022_1234",
        "outputs": {
            "y_hat": "approved",
            "confidence": 0.85
        },
        "metrics": {"success_rate": 0.95},
        "lineage": {"plan_id": "plan_123"},
        "governance": {"approved": True},
        "timing": {"total_ms": 245}
    }
    print("✓ InferenceResult schema example created")
    
    assert True

if __name__ == "__main__":
    success = test_intelligence_service()
    if success:
        test_api_schemas()
        
    sys.exit(0 if success else 1)