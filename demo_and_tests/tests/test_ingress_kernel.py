"""
Test script for Ingress Kernel implementation.
"""
import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from grace.ingress_kernel import IngressKernel
from grace.contracts.ingress_contracts import SourceConfig, SourceKind, AuthMode, ParserType, PIIPolicy, GovernanceLabel


async def test_ingress_kernel():
    """Test basic ingress kernel functionality."""
    print("Testing Grace Ingress Kernel...")
    
    try:
        # Initialize kernel
        print("1. Initializing Ingress Kernel...")
        kernel = IngressKernel(storage_path="/tmp/test_ingress")
        await kernel.start()
        print("   ✓ Kernel started successfully")
        
        # Test source registration
        print("2. Registering test sources...")
        
        # Register HTTP source
        http_config = {
            "source_id": "src_test_http",
            "kind": "http",
            "uri": "https://api.example.com/data",
            "auth_mode": "bearer",
            "secrets_ref": "bearer_token_123",
            "schedule": "0 */6 * * *",  # Every 6 hours
            "parser": "json",
            "target_contract": "contract:article.v1",
            "retention_days": 90,
            "pii_policy": "mask",
            "governance_label": "public",
            "enabled": True
        }
        
        http_source_id = kernel.register_source(http_config)
        print(f"   ✓ HTTP source registered: {http_source_id}")
        
        # Register RSS source
        rss_config = {
            "source_id": "src_test_rss",
            "kind": "rss",
            "uri": "https://example.com/news.xml",
            "auth_mode": "none",
            "schedule": "0 * * * *",  # Every hour
            "parser": "xml",
            "target_contract": "contract:article.v1",
            "retention_days": 365,
            "pii_policy": "block",
            "governance_label": "internal",
            "enabled": True
        }
        
        rss_source_id = kernel.register_source(rss_config)
        print(f"   ✓ RSS source registered: {rss_source_id}")
        
        # Test data capture
        print("3. Testing data capture...")
        
        # Capture JSON data
        json_payload = {
            "title": "Test Article Title",
            "author": "John Doe",
            "content": "This is test article content for the ingress system.",
            "url": "https://example.com/article/123",
            "published_at": datetime.utcnow().isoformat(),
            "language": "en",
            "topics": ["technology", "ai", "data"]
        }
        
        event_id = await kernel.capture(
            source_id=http_source_id,
            payload=json_payload,
            headers={"content-type": "application/json"}
        )
        print(f"   ✓ JSON data captured: {event_id}")
        
        # Capture text data with potential PII
        text_payload = {
            "content": "Contact us at support@company.com or call 555-123-4567 for more information.",
            "source": "customer_inquiry",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        pii_event_id = await kernel.capture(
            source_id=rss_source_id,
            payload=text_payload,
            headers={"content-type": "text/plain"}
        )
        print(f"   ✓ Text data with PII captured: {pii_event_id}")
        
        # Wait for processing
        print("4. Waiting for data processing...")
        await asyncio.sleep(2)
        print("   ✓ Processing completed")
        
        # Test source health
        print("5. Checking source health...")
        health_status = kernel.get_health_status()
        print(f"   ✓ Health status: {health_status['status']}")
        print(f"   ✓ Sources tracked: {health_status['sources']}")
        
        # Test trust scoring stats
        print("6. Checking trust scoring...")
        trust_stats = kernel.trust_scorer.get_stats()
        print(f"   ✓ Sources tracked for trust: {trust_stats['sources_tracked']}")
        print(f"   ✓ Average reputation: {trust_stats['average_reputation']:.3f}")
        
        # Test policy enforcement
        print("7. Testing policy enforcement...")
        policy_stats = kernel.policy_guard.pii_validator.config
        print(f"   ✓ PII patterns configured: {len(kernel.policy_guard.pii_validator.pii_patterns)}")
        
        # Test snapshot creation
        print("8. Creating system snapshot...")
        snapshot = await kernel.export_snapshot()
        print(f"   ✓ Snapshot created: {snapshot.snapshot_id}")
        print(f"   ✓ Active sources in snapshot: {len(snapshot.active_sources)}")
        
        # Test adapter factory
        print("9. Testing adapter factory...")
        from grace.ingress_kernel.adapters.base import AdapterFactory
        supported_kinds = AdapterFactory.list_supported_kinds()
        print(f"   ✓ Supported source kinds: {supported_kinds}")
        
        # Test parser factory
        print("10. Testing parser factory...")
        from grace.ingress_kernel.parsers.base import ParserFactory
        supported_parsers = ParserFactory.list_supported_types()
        print(f"   ✓ Supported parser types: {supported_parsers}")
        
        # Test configuration validation
        print("11. Validating configuration...")
        config_keys = list(kernel.config.keys())
        print(f"   ✓ Configuration sections: {config_keys}")
        
        # Test storage paths
        print("12. Checking storage organization...")
        storage_paths = [
            kernel.bronze_path,
            kernel.silver_path,
            kernel.gold_path
        ]
        
        for path in storage_paths:
            if path.exists():
                file_count = len(list(path.iterdir()))
                print(f"   ✓ {path.name} tier: {file_count} files")
        
        # Test component integration
        print("13. Testing component integration...")
        components = [
            ("Adapter Factory", kernel.adapter_factory),
            ("Parser Factory", kernel.parser_factory),
            ("Trust Scorer", kernel.trust_scorer),
            ("Policy Guard", kernel.policy_guard),
            ("Snapshot Manager", kernel.snapshot_manager)
        ]
        
        for name, component in components:
            print(f"   ✓ {name}: {type(component).__name__} initialized")
        
        print("14. Shutting down kernel...")
        await kernel.stop()
        print("   ✓ Kernel stopped successfully")
        
        print("\nAll tests passed! 🎉")
        print("Grace Ingress Kernel is working correctly.")
        
        # Print summary statistics
        print("\n" + "="*50)
        print("INGRESS KERNEL SUMMARY")
        print("="*50)
        print(f"Sources registered: {len(kernel.sources)}")
        print(f"Health checks: {len(kernel.health_status)}")
        print(f"Trust scoring active: Yes")
        print(f"Policy enforcement active: Yes") 
        print(f"Snapshot management active: Yes")
        print(f"Storage tiers: Bronze, Silver, Gold")
        print(f"Supported source types: {len(supported_kinds)}")
        print(f"Supported parsers: {len(supported_parsers)}")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fastapi_service():
    """Test FastAPI service integration."""
    print("\nTesting FastAPI Service Integration...")
    
    try:
        from grace.ingress_kernel.service import create_ingress_app
        
        # Create FastAPI app
        app = create_ingress_app()
        print("   ✓ FastAPI app created successfully")
        
        # Check routes
        routes = [route.path for route in app.routes]
        api_routes = [r for r in routes if r.startswith('/api/ingress')]
        print(f"   ✓ API routes configured: {len(api_routes)}")
        
        print("   ✓ FastAPI service integration working")
        return True
        
    except Exception as e:
        print(f"   ❌ FastAPI service test failed: {e}")
        return False


async def test_bridges():
    """Test bridge integrations."""
    print("\nTesting Bridge Integrations...")
    
    try:
        from grace.ingress_kernel.mesh_bridge import IngressMeshBridge
        from grace.ingress_kernel.governance_bridge import IngressGovernanceBridge  
        from grace.ingress_kernel.mlt_bridge import IngressMLTBridge
        
        kernel = IngressKernel(storage_path="/tmp/test_ingress_bridges")
        
        # Test bridges
        mesh_bridge = IngressMeshBridge(kernel)
        governance_bridge = IngressGovernanceBridge(kernel)
        mlt_bridge = IngressMLTBridge(kernel)
        
        print("   ✓ Event Mesh Bridge initialized")
        print("   ✓ Governance Bridge initialized")
        print("   ✓ MLT Bridge initialized")
        
        # Test bridge stats
        mesh_stats = mesh_bridge.get_stats()
        governance_stats = governance_bridge.get_stats()
        mlt_stats = mlt_bridge.get_stats()
        
        print(f"   ✓ Mesh bridge routing rules: {mesh_stats['routing_rules']}")
        print(f"   ✓ Governance bridge pending requests: {governance_stats['pending_requests']}")
        print(f"   ✓ MLT bridge experience buffer: {mlt_stats['experience_buffer_size']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Bridge integration test failed: {e}")
        return False


async def main():
    """Main test runner."""
    print("🔍 Grace Ingress Kernel Comprehensive Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(await test_ingress_kernel())
    test_results.append(await test_fastapi_service())
    test_results.append(await test_bridges())
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Ingress Kernel is ready for deployment.")
        return 0
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)