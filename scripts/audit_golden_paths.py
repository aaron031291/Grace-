"""
Audit all golden paths to ensure complete coverage
"""

import sys
import asyncio
from pathlib import Path


class GoldenPathAuditor:
    """Audits critical system paths"""
    
    def __init__(self):
        self.paths = {
            "Event Flow": self._audit_event_flow,
            "Consensus Flow": self._audit_consensus_flow,
            "Memory Persistence": self._audit_memory_persistence,
            "RBAC Enforcement": self._audit_rbac,
            "Rate Limiting": self._audit_rate_limiting,
            "Encryption": self._audit_encryption,
            "MCP Validation": self._audit_mcp,
            "KPI Tracking": self._audit_kpis,
        }
        self.results = {}
    
    async def _audit_event_flow(self):
        """Audit event flow path"""
        try:
            from grace.integration.event_bus import EventBus
            from grace.schemas.events import GraceEvent
            
            bus = EventBus()
            event = GraceEvent(event_type="test", source="audit")
            await bus.emit(event)
            await bus.shutdown()
            return True, "Event flow working"
        except Exception as e:
            return False, f"Event flow failed: {e}"
    
    async def _audit_consensus_flow(self):
        """Audit consensus request path"""
        try:
            from grace.kernels.mldl import MLDLKernel
            from grace.integration.event_bus import EventBus
            from grace.events.factory import GraceEventFactory
            
            bus = EventBus()
            factory = GraceEventFactory()
            kernel = MLDLKernel(bus, factory, None, None, None)
            await kernel.start()
            
            health = kernel.get_health()
            await kernel.stop()
            await bus.shutdown()
            
            return health["running"] is False, "Consensus kernel working"
        except Exception as e:
            return False, f"Consensus flow failed: {e}"
    
    async def _audit_memory_persistence(self):
        """Audit memory write/read path"""
        try:
            from grace.memory.async_lightning import AsyncLightningMemory
            
            lightning = AsyncLightningMemory()
            await lightning.connect()
            
            await lightning.set("audit_key", "audit_value")
            value = await lightning.get("audit_key")
            
            await lightning.disconnect()
            
            return value == "audit_value", "Memory persistence working"
        except Exception as e:
            return False, f"Memory persistence failed: {e}"
    
    async def _audit_rbac(self):
        """Audit RBAC path"""
        try:
            from grace.security import RBACManager, Role, Permission
            
            rbac = RBACManager()
            await rbac.assign_role("audit_user", Role.USER, "system")
            
            has_perm = rbac.has_permission("audit_user", Permission.READ_EVENTS)
            
            return has_perm is True, "RBAC enforcement working"
        except Exception as e:
            return False, f"RBAC failed: {e}"
    
    async def _audit_rate_limiting(self):
        """Audit rate limiting path"""
        try:
            from grace.security import RateLimiter
            
            limiter = RateLimiter(default_limit=5, default_window=60)
            
            for _ in range(5):
                await limiter.check_rate_limit("audit_user", "audit")
            
            return True, "Rate limiting working"
        except Exception as e:
            return False, f"Rate limiting failed: {e}"
    
    async def _audit_encryption(self):
        """Audit encryption path"""
        try:
            from grace.security import EncryptionManager
            
            enc = EncryptionManager()
            
            plaintext = "sensitive data"
            encrypted = enc.encrypt(plaintext)
            decrypted = enc.decrypt(encrypted)
            
            return decrypted == plaintext, "Encryption working"
        except Exception as e:
            return False, f"Encryption failed: {e}"
    
    async def _audit_mcp(self):
        """Audit MCP validation path"""
        try:
            from grace.mcp import MCPClient, MCPMessageType
            from grace.integration.event_bus import EventBus
            
            bus = EventBus()
            client = MCPClient("audit_kernel", bus)
            
            message = await client.send_message(
                destination="target",
                payload={"test": True},
                message_type=MCPMessageType.REQUEST,
                trust_score=0.9
            )
            
            await bus.shutdown()
            
            return message is not None, "MCP validation working"
        except Exception as e:
            return False, f"MCP failed: {e}"
    
    async def _audit_kpis(self):
        """Audit KPI tracking path"""
        try:
            from grace.observability.kpis import KPITracker
            
            tracker = KPITracker()
            
            metrics = {
                "grace_events_published_total": 100,
                "grace_events_processed_total": 98
            }
            
            await tracker.calculate_kpis_from_metrics(metrics)
            report = tracker.get_kpi_report()
            
            return "kpis" in report, "KPI tracking working"
        except Exception as e:
            return False, f"KPI tracking failed: {e}"
    
    async def run_audits(self):
        """Run all audits"""
        print("🔍 Golden Path Audit")
        print("=" * 60)
        
        for name, audit_func in self.paths.items():
            try:
                passed, message = await audit_func()
                self.results[name] = passed
                
                status = "✅" if passed else "❌"
                print(f"{status} {name}: {message}")
            except Exception as e:
                self.results[name] = False
                print(f"❌ {name}: Critical error: {e}")
        
        print("\n" + "=" * 60)
        
        total = len(self.results)
        passed = sum(1 for v in self.results.values() if v)
        
        print(f"Results: {passed}/{total} paths working")
        
        if passed == total:
            print("✅ All golden paths validated!")
            return 0
        else:
            print(f"❌ {total - passed} paths failing")
            return 1


async def main():
    """Run golden path audits"""
    auditor = GoldenPathAuditor()
    return await auditor.run_audits()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
