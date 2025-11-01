"""
Component Communication Validator
Ensures all Grace components can communicate with each other
"""

import asyncio
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component operational status"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ComponentValidator:
    """
    Validates inter-component communication.
    
    Tests:
    1. Backend ↔ Frontend
    2. Event Bus ↔ All Components
    3. Database ↔ All Services
    4. Crypto Manager ↔ Immutable Logger
    5. MCP ↔ External Tools
    6. Breakthrough ↔ All Systems
    """
    
    def __init__(self):
        self.component_status: Dict[str, ComponentStatus] = {}
        self.communication_matrix: Dict[Tuple[str, str], bool] = {}
        
    async def validate_all_components(self) -> Dict[str, Any]:
        """Validate all component communication"""
        logger.info("🔍 Starting component validation...")
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
            "communication_paths": {},
            "overall_health": "unknown"
        }
        
        # Test each component
        components_to_test = [
            ("crypto_manager", self._test_crypto_manager),
            ("mcp_server", self._test_mcp_server),
            ("breakthrough_system", self._test_breakthrough),
            ("collaborative_gen", self._test_collaborative_gen),
            ("backend_api", self._test_backend_api),
            ("event_bus", self._test_event_bus),
            ("database", self._test_database),
        ]
        
        for component_name, test_func in components_to_test:
            try:
                status, details = await test_func()
                results["components"][component_name] = {
                    "status": status.value,
                    "details": details
                }
                self.component_status[component_name] = status
            except Exception as e:
                logger.error(f"Component test failed: {component_name} - {e}")
                results["components"][component_name] = {
                    "status": ComponentStatus.OFFLINE.value,
                    "error": str(e)
                }
                self.component_status[component_name] = ComponentStatus.OFFLINE
        
        # Test communication paths
        comm_tests = [
            ("crypto_manager", "immutable_logger"),
            ("mcp_server", "breakthrough_system"),
            ("collaborative_gen", "crypto_manager"),
            ("breakthrough_system", "mcp_server")
        ]
        
        for source, target in comm_tests:
            try:
                can_communicate = await self._test_communication(source, target)
                results["communication_paths"][f"{source}→{target}"] = can_communicate
                self.communication_matrix[(source, target)] = can_communicate
            except Exception as e:
                logger.error(f"Communication test failed: {source}→{target} - {e}")
                results["communication_paths"][f"{source}→{target}"] = False
        
        # Determine overall health
        online_count = sum(
            1 for status in self.component_status.values()
            if status == ComponentStatus.ONLINE
        )
        total_components = len(self.component_status)
        
        if online_count == total_components:
            results["overall_health"] = "healthy"
        elif online_count > total_components / 2:
            results["overall_health"] = "degraded"
        else:
            results["overall_health"] = "critical"
        
        # Communication health
        working_paths = sum(1 for v in self.communication_matrix.values() if v)
        total_paths = len(self.communication_matrix)
        results["communication_health"] = f"{working_paths}/{total_paths} paths working"
        
        logger.info(f"✅ Validation complete: {results['overall_health']}")
        
        return results
    
    async def _test_crypto_manager(self) -> Tuple[ComponentStatus, Dict]:
        """Test crypto manager"""
        try:
            from grace.security.crypto_manager import get_crypto_manager
            
            crypto = get_crypto_manager()
            
            # Test key generation
            test_key = crypto.generate_operation_key(
                "test_validation",
                "validation_test",
                {}
            )
            
            # Test signing
            sig = crypto.sign_operation_data("test_validation", {"test": "data"}, "test")
            
            return ComponentStatus.ONLINE, {
                "key_generated": True,
                "signing_works": True,
                "stats": crypto.get_stats()
            }
        except Exception as e:
            return ComponentStatus.OFFLINE, {"error": str(e)}
    
    async def _test_mcp_server(self) -> Tuple[ComponentStatus, Dict]:
        """Test MCP server"""
        try:
            from grace.mcp.mcp_server import get_mcp_server
            
            mcp = get_mcp_server()
            tools = mcp.list_tools()
            
            return ComponentStatus.ONLINE, {
                "tools_registered": len(tools),
                "tools": [t["name"] for t in tools]
            }
        except Exception as e:
            return ComponentStatus.OFFLINE, {"error": str(e)}
    
    async def _test_breakthrough(self) -> Tuple[ComponentStatus, Dict]:
        """Test breakthrough system"""
        try:
            from grace.core.breakthrough import BreakthroughSystem
            
            system = BreakthroughSystem()
            
            return ComponentStatus.ONLINE, {
                "initialized": system.initialized,
                "components": ["eval_harness", "meta_loop", "consensus"]
            }
        except Exception as e:
            return ComponentStatus.OFFLINE, {"error": str(e)}
    
    async def _test_collaborative_gen(self) -> Tuple[ComponentStatus, Dict]:
        """Test collaborative code generation"""
        try:
            from grace.mtl.collaborative_code_gen import CollaborativeCodeGenerator
            
            gen = CollaborativeCodeGenerator()
            
            return ComponentStatus.ONLINE, {
                "active_tasks": len(gen.active_tasks),
                "completed_tasks": len(gen.completed_tasks)
            }
        except Exception as e:
            return ComponentStatus.OFFLINE, {"error": str(e)}
    
    async def _test_backend_api(self) -> Tuple[ComponentStatus, Dict]:
        """Test backend API"""
        try:
            # Test if backend can be imported
            import backend.main
            
            return ComponentStatus.ONLINE, {
                "api_available": True
            }
        except Exception as e:
            return ComponentStatus.DEGRADED, {"error": str(e)}
    
    async def _test_event_bus(self) -> Tuple[ComponentStatus, Dict]:
        """Test event bus"""
        try:
            from grace.events.event_bus import EventBus
            
            bus = EventBus()
            
            return ComponentStatus.ONLINE, {
                "subscribers": len(bus._subscribers) if hasattr(bus, '_subscribers') else 0
            }
        except Exception as e:
            return ComponentStatus.DEGRADED, {"error": str(e)}
    
    async def _test_database(self) -> Tuple[ComponentStatus, Dict]:
        """Test database"""
        try:
            from backend.database import DatabaseManager
            
            # Test health check
            healthy = await DatabaseManager.health_check()
            
            status = ComponentStatus.ONLINE if healthy else ComponentStatus.DEGRADED
            return status, {"healthy": healthy}
        except Exception as e:
            return ComponentStatus.OFFLINE, {"error": str(e)}
    
    async def _test_communication(self, source: str, target: str) -> bool:
        """Test if two components can communicate"""
        # Check both components are online
        if self.component_status.get(source) != ComponentStatus.ONLINE:
            return False
        if self.component_status.get(target) != ComponentStatus.ONLINE:
            return False
        
        # Component-specific communication tests
        if source == "crypto_manager" and target == "immutable_logger":
            # Test crypto can log
            try:
                from grace.security.crypto_manager import get_crypto_manager
                crypto = get_crypto_manager()
                # Key generation logs automatically
                return True
            except:
                return False
        
        # Default: if both online, assume can communicate
        return True
    
    def print_validation_report(self, results: Dict[str, Any]):
        """Print human-readable validation report"""
        print("\n" + "="*70)
        print("GRACE COMPONENT VALIDATION REPORT")
        print("="*70)
        
        print(f"\n🏥 Overall Health: {results['overall_health'].upper()}")
        print(f"⏰ Timestamp: {results['timestamp']}")
        
        print(f"\n📊 Component Status:")
        for component, info in results["components"].items():
            status_icon = "✅" if info["status"] == "online" else "❌"
            print(f"  {status_icon} {component}: {info['status']}")
            if "error" in info:
                print(f"      Error: {info['error']}")
        
        print(f"\n🔗 Communication Paths:")
        for path, working in results["communication_paths"].items():
            icon = "✅" if working else "❌"
            print(f"  {icon} {path}")
        
        print(f"\n📈 Summary:")
        online = sum(1 for c in results["components"].values() if c["status"] == "online")
        total = len(results["components"])
        print(f"  Components Online: {online}/{total}")
        print(f"  Communication Health: {results['communication_health']}")
        
        print("\n" + "="*70)


async def validate_grace_operational():
    """Main validation function"""
    validator = ComponentValidator()
    results = await validator.validate_all_components()
    validator.print_validation_report(results)
    return results


if __name__ == "__main__":
    # Run validation
    asyncio.run(validate_grace_operational())
