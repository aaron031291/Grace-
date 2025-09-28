#!/usr/bin/env python3
"""
Simple integration test for Grace components without external dependencies.
Tests the core governance logic and configuration loading.
"""
import sys
import os
from pathlib import Path

# Add Grace to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_config_loading():
    """Test configuration loading."""
    try:
        from grace.config.environment import get_grace_config, validate_environment
        
        # Test environment validation (should work without .env file)
        missing = validate_environment()
        print(f"✓ Environment validation works (missing: {len(missing)} vars)")
        
        # Test config loading with defaults
        config = get_grace_config()
        print(f"✓ Configuration loaded successfully")
        print(f"  - Instance ID: {config['environment_config']['instance_id']}")
        print(f"  - Version: {config['environment_config']['version']}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_governance_imports():
    """Test governance kernel imports."""
    try:
        from grace.governance.grace_governance_kernel import GraceGovernanceKernel
        print("✓ Governance kernel imports successfully")
        
        # Test basic instantiation
        kernel = GraceGovernanceKernel()
        print("✓ Governance kernel can be instantiated")
        
        return True
    except Exception as e:
        print(f"✗ Governance kernel test failed: {e}")
        return False

def test_service_imports():
    """Test service wrapper imports (without FastAPI)."""
    try:
        # Test individual components
        from grace_service.websocket_manager import WebSocketManager
        print("✓ WebSocket manager imports successfully")
        
        from grace_service.schemas.base import BaseResponse, HealthResponse
        print("✓ Schema models import successfully")
        
        # Test instantiation
        ws_manager = WebSocketManager()
        print("✓ WebSocket manager can be instantiated")
        
        response = BaseResponse(status="success", message="test")
        print("✓ Response models work correctly")
        
        return True
    except Exception as e:
        print(f"✗ Service components test failed: {e}")
        return False

def test_database_schema():
    """Test database schema loading."""
    try:
        schema_file = Path(__file__).parent.parent / "init_db" / "01_init_grace_db.sql"
        if schema_file.exists():
            with open(schema_file, 'r') as f:
                schema_content = f.read()
            
            # Basic validation of SQL content
            required_tables = [
                'constitutional_framework',
                'governance_instances', 
                'structured_memory'
            ]
            
            missing_tables = []
            for table in required_tables:
                if table not in schema_content:
                    missing_tables.append(table)
            
            if not missing_tables:
                print("✓ Database schema contains required tables")
                return True
            else:
                print(f"✗ Database schema missing tables: {missing_tables}")
                return False
        else:
            print("✗ Database schema file not found")
            return False
            
    except Exception as e:
        print(f"✗ Database schema test failed: {e}")
        return False

def test_deployment_files():
    """Test deployment configuration files."""
    try:
        required_files = [
            ".env.template",
            "docker-compose.yml", 
            "Dockerfile",
            "requirements.txt",
            "Makefile"
        ]
        
        missing_files = []
        base_path = Path(__file__).parent.parent
        
        for file_name in required_files:
            if not (base_path / file_name).exists():
                missing_files.append(file_name)
        
        if not missing_files:
            print("✓ All deployment files present")
            return True
        else:
            print(f"✗ Missing deployment files: {missing_files}")
            return False
            
    except Exception as e:
        print(f"✗ Deployment files test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🧪 Grace Integration Tests (No External Dependencies)")
    print("=" * 55)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Governance Imports", test_governance_imports),
        ("Service Components", test_service_imports),
        ("Database Schema", test_database_schema),
        ("Deployment Files", test_deployment_files),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 55)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\n✅ Grace is ready for deployment!")
        print("   Next steps:")
        print("   1. Set up .env file with your configuration")
        print("   2. Run 'make up' to start the system") 
        print("   3. Check health at http://localhost:8080/health/status")
        return 0
    else:
        print("⚠️ Some tests failed - please check the errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())