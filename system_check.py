#!/usr/bin/env python3
"""
Grace System Health Check Script
Automated verification of all layers and vaults initialization.
"""

import asyncio
import sys
import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add grace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ComponentStatus:
    """Status of a system component."""
    name: str
    status: str  # 'healthy', 'degraded', 'unhealthy', 'missing'
    details: Dict[str, Any]
    error: Optional[str] = None
    response_time: Optional[float] = None

@dataclass
class SystemHealthReport:
    """Complete system health report."""
    timestamp: str
    overall_status: str
    components: List[ComponentStatus]
    layers_status: Dict[str, str]
    vaults_status: Dict[str, str]
    telemetry_active: bool
    ooda_loop_functional: bool
    governance_functional: bool
    recommendations: List[str]

class GraceSystemHealthChecker:
    """Main system health checker for Grace governance system."""
    
    def __init__(self):
        self.start_time = time.time()
        self.report = SystemHealthReport(
            timestamp=datetime.now().isoformat(),
            overall_status="unknown",
            components=[],
            layers_status={},
            vaults_status={},
            telemetry_active=False,
            ooda_loop_functional=False,
            governance_functional=False,
            recommendations=[]
        )
    
    async def run_comprehensive_check(self) -> SystemHealthReport:
        """Run comprehensive system health check."""
        logger.info("🔍 Starting Grace System Health Check...")
        
        # Check core layers
        await self._check_core_layers()
        
        # Check governance components
        await self._check_governance_components()
        
        # Check vault systems
        await self._check_vault_systems()
        
        # Check telemetry and monitoring
        await self._check_telemetry_systems()
        
        # Test OODA loop functionality
        await self._test_ooda_loop()
        
        # Test governance self-test
        await self._test_governance_functionality()
        
        # Generate overall status and recommendations
        self._generate_overall_status()
        
        return self.report
    
    async def _check_core_layers(self):
        """Check initialization of core system layers."""
        logger.info("Checking core layers...")
        
        layers_to_check = [
            ("EventBus", "grace.core.event_bus", "EventBus"),
            ("MemoryCore", "grace.core", "MemoryCore"),  
            ("EventMesh", "grace.layer_02_event_mesh", "GraceEventBus"),
            ("AuditLogs", "grace.layer_04_audit_logs.immutable_logs", "ImmutableLogs"),
        ]
        
        for layer_name, module_path, class_name in layers_to_check:
            status = await self._check_component_import(layer_name, module_path, class_name)
            self.report.components.append(status)
            self.report.layers_status[layer_name] = status.status
    
    async def _check_governance_components(self):
        """Check governance system components."""
        logger.info("Checking governance components...")
        
        components_to_check = [
            ("GovernanceKernel", "grace.governance.grace_governance_kernel", "GraceGovernanceKernel"),
            ("VerificationEngine", "grace.governance.verification_engine", "VerificationEngine"),
            ("TrustCore", "grace.governance.trust_core_kernel", "TrustCoreKernel"),
            ("Parliament", "grace.governance.parliament", "Parliament"),
            ("GovernanceAPI", "grace.governance.governance_api", "GovernanceAPIService"),
        ]
        
        for comp_name, module_path, class_name in components_to_check:
            status = await self._check_component_import(comp_name, module_path, class_name)
            self.report.components.append(status)
    
    async def _check_vault_systems(self):
        """Check vault and secure storage systems."""
        logger.info("Checking vault systems...")
        
        vault_systems = [
            ("MTL_Kernel", "grace.mtl_kernel.kernel", "MTLKernel"),
            ("TrustService", "grace.mtl_kernel.trust_service", "TrustService"),
            ("ImmutableLogService", "grace.mtl_kernel.immutable_log_service", "ImmutableLogService"),
            ("SecurityVault", "grace.contracts.message_envelope_simple", "RBACContext"),
        ]
        
        for vault_name, module_path, class_name in vault_systems:
            status = await self._check_component_import(vault_name, module_path, class_name)
            self.report.components.append(status)
            self.report.vaults_status[vault_name] = status.status
    
    async def _check_telemetry_systems(self):
        """Check telemetry and monitoring systems."""
        logger.info("Checking telemetry systems...")
        
        telemetry_systems = [
            ("StatusService", "grace.interface.status_service", "GraceStatusService"),
            ("InterfaceKernel", "grace.interface_kernel.kernel", "InterfaceKernel"),
            ("OrchestrationService", "grace.orchestration.orchestration_service", "OrchestrationService"),
        ]
        
        telemetry_active = True
        for telem_name, module_path, class_name in telemetry_systems:
            status = await self._check_component_import(telem_name, module_path, class_name)
            self.report.components.append(status)
            if status.status != "healthy":
                telemetry_active = False
        
        self.report.telemetry_active = telemetry_active
    
    async def _test_ooda_loop(self):
        """Test OODA loop functionality."""
        logger.info("Testing OODA loop functionality...")
        
        try:
            # Import scheduler
            from grace.orchestration.scheduler.scheduler import Scheduler, LoopDefinition
            
            # Create test scheduler
            scheduler = Scheduler()
            
            # Define test OODA loop
            ooda_loop = LoopDefinition(
                loop_id="test_ooda",
                name="ooda",
                priority=5,
                interval_s=1,
                kernels=["test"],
                policies={},
                enabled=True
            )
            
            # Execute OODA loop
            result = await scheduler._execute_ooda_loop(ooda_loop)
            
            # Check if all stages completed
            if "stages" in result and len(result["stages"]) == 4:
                stages = result["stages"]
                required_stages = ["observe", "orient", "decide", "act"]
                actual_stages = [stage.get("stage") for stage in stages]
                
                if all(stage in actual_stages for stage in required_stages):
                    self.report.ooda_loop_functional = True
                    status = ComponentStatus(
                        name="OODA_Loop",
                        status="healthy",
                        details=result,
                        response_time=time.time() - self.start_time
                    )
                else:
                    status = ComponentStatus(
                        name="OODA_Loop",
                        status="degraded",
                        details=result,
                        error="Missing required stages"
                    )
            else:
                status = ComponentStatus(
                    name="OODA_Loop",
                    status="unhealthy",
                    details=result,
                    error="Invalid OODA loop result structure"
                )
            
            self.report.components.append(status)
            
        except Exception as e:
            status = ComponentStatus(
                name="OODA_Loop",
                status="unhealthy",
                details={},
                error=str(e)
            )
            self.report.components.append(status)
    
    async def _test_governance_functionality(self):
        """Test governance self-test capabilities."""
        logger.info("Testing governance functionality...")
        
        governance_tests = []
        
        # Test 1: Contradiction Detection
        contradiction_status = await self._test_contradiction_detection()
        governance_tests.append(contradiction_status)
        
        # Test 2: Decision Narration
        narration_status = await self._test_decision_narration()
        governance_tests.append(narration_status)
        
        # Test 3: Trust Scoring
        trust_status = await self._test_trust_scoring()
        governance_tests.append(trust_status)
        
        # Check if all governance tests passed
        all_healthy = all(test.status == "healthy" for test in governance_tests)
        self.report.governance_functional = all_healthy
        
        self.report.components.extend(governance_tests)
    
    async def _test_contradiction_detection(self) -> ComponentStatus:
        """Test contradiction detection capability."""
        try:
            from grace.governance.verification_engine import VerificationEngine
            from grace.core import Claim, EventBus, MemoryCore, Source, Evidence, LogicStep
            
            # Create test components
            event_bus = EventBus()
            memory_core = MemoryCore()
            verification_engine = VerificationEngine(event_bus, memory_core)
            
            # Create contradicting claims with proper structure
            claim1 = Claim(
                id="test_claim_1",
                statement="The system is secure",
                sources=[Source(uri="system://health_check", credibility=0.9)],
                evidence=[Evidence(type="observation", pointer="all_checks_passed")],
                confidence=0.8,
                logical_chain=[LogicStep(step="System checks completed successfully")]
            )
            
            claim2 = Claim(
                id="test_claim_2", 
                statement="The system is not secure",
                sources=[Source(uri="system://security_scan", credibility=0.7)],
                evidence=[Evidence(type="scan", pointer="vulnerabilities_found")],
                confidence=0.7,
                logical_chain=[LogicStep(step="Security vulnerabilities detected")]
            )
            
            # Test contradiction detection
            contradictions = await verification_engine._check_contradictions(claim1, [claim2])
            
            if contradictions and len(contradictions) > 0:
                return ComponentStatus(
                    name="Contradiction_Detection",
                    status="healthy",
                    details={"contradictions_found": len(contradictions)},
                    response_time=time.time() - self.start_time
                )
            else:
                return ComponentStatus(
                    name="Contradiction_Detection",
                    status="degraded",
                    details={"contradictions_found": 0},
                    error="Failed to detect obvious contradiction"
                )
                
        except Exception as e:
            return ComponentStatus(
                name="Contradiction_Detection",
                status="unhealthy",
                details={},
                error=str(e)
            )
    
    async def _test_decision_narration(self) -> ComponentStatus:
        """Test decision narration capability."""
        try:
            from grace.governance.unified_logic import UnifiedLogic
            from grace.core import EventBus, MemoryCore
            
            event_bus = EventBus()
            memory_core = MemoryCore()
            unified_logic = UnifiedLogic(event_bus, memory_core)
            
            # Test decision synthesis (narration) with proper parameters
            test_inputs = {
                "claims": ["System is operational", "All services running"],
                "context": "Health check verification",
                "verification_results": {"status": "verified"}
            }
            
            decision = await unified_logic.synthesize_decision(
                topic="system_health_check",
                inputs=test_inputs
            )
            
            if decision and hasattr(decision, 'recommendation') and decision.recommendation:
                return ComponentStatus(
                    name="Decision_Narration",
                    status="healthy",
                    details={"decision": decision.recommendation, "confidence": getattr(decision, 'confidence', 0.0)},
                    response_time=time.time() - self.start_time
                )
            else:
                return ComponentStatus(
                    name="Decision_Narration",
                    status="degraded",
                    details={"decision_type": str(type(decision))},
                    error="Invalid decision structure or empty decision"
                )
                
        except Exception as e:
            return ComponentStatus(
                name="Decision_Narration",
                status="unhealthy",
                details={},
                error=str(e)
            )
    
    async def _test_trust_scoring(self) -> ComponentStatus:
        """Test trust scoring capability."""
        try:
            from grace.mtl_kernel.trust_service import TrustService
            from grace.mtl_kernel.schemas import MemoryStore
            
            # Create trust service
            memory_store = MemoryStore()
            trust_service = TrustService(memory_store)
            
            # Test trust initialization and scoring
            memory_id = "test_memory_entry"
            audit_id = trust_service.init_trust(memory_id)
            
            # Add test attestation
            attest_id = trust_service.attest(memory_id, {"quality": 0.9}, "system")
            
            # Get trust score
            trust_score = trust_service.get_trust_score(memory_id)
            
            if isinstance(trust_score, (int, float)) and 0 <= trust_score <= 1:
                return ComponentStatus(
                    name="Trust_Scoring",
                    status="healthy",
                    details={
                        "trust_score": trust_score,
                        "audit_id": audit_id,
                        "attestation_id": attest_id
                    },
                    response_time=time.time() - self.start_time
                )
            else:
                return ComponentStatus(
                    name="Trust_Scoring",
                    status="degraded",
                    details={"trust_score": trust_score},
                    error="Invalid trust score range"
                )
                
        except Exception as e:
            return ComponentStatus(
                name="Trust_Scoring",
                status="unhealthy",
                details={},
                error=str(e)
            )
    
    async def _check_component_import(self, component_name: str, module_path: str, class_name: str) -> ComponentStatus:
        """Check if a component can be imported and instantiated."""
        start_time = time.time()
        
        try:
            # Import the module
            module = __import__(module_path, fromlist=[class_name])
            component_class = getattr(module, class_name)
            
            # Basic instantiation check (without full initialization)
            # Most components need specific parameters, so we just verify the class exists
            details = {
                "module_path": module_path,
                "class_name": class_name,
                "importable": True
            }
            
            # Try to get class info
            if hasattr(component_class, '__doc__'):
                details["description"] = component_class.__doc__.split('\n')[0] if component_class.__doc__ else "No description"
            
            response_time = time.time() - start_time
            
            return ComponentStatus(
                name=component_name,
                status="healthy",
                details=details,
                response_time=response_time
            )
            
        except ImportError as e:
            return ComponentStatus(
                name=component_name,
                status="missing",
                details={"module_path": module_path, "class_name": class_name},
                error=f"Import error: {str(e)}"
            )
        except AttributeError as e:
            return ComponentStatus(
                name=component_name,
                status="degraded",
                details={"module_path": module_path, "class_name": class_name},
                error=f"Class not found: {str(e)}"
            )
        except Exception as e:
            return ComponentStatus(
                name=component_name,
                status="unhealthy",
                details={"module_path": module_path, "class_name": class_name},
                error=f"Unexpected error: {str(e)}"
            )
    
    def _generate_overall_status(self):
        """Generate overall system status and recommendations."""
        
        # Count component statuses
        status_counts = {"healthy": 0, "degraded": 0, "unhealthy": 0, "missing": 0}
        for component in self.report.components:
            status_counts[component.status] += 1
        
        total_components = len(self.report.components)
        healthy_percentage = (status_counts["healthy"] / total_components) * 100 if total_components > 0 else 0
        
        # Determine overall status
        if healthy_percentage >= 90:
            self.report.overall_status = "healthy"
        elif healthy_percentage >= 70:
            self.report.overall_status = "degraded"
        else:
            self.report.overall_status = "unhealthy"
        
        # Generate recommendations
        if status_counts["missing"] > 0:
            self.report.recommendations.append(
                f"Missing components detected ({status_counts['missing']}). Check installation and imports."
            )
        
        if status_counts["unhealthy"] > 0:
            self.report.recommendations.append(
                f"Unhealthy components detected ({status_counts['unhealthy']}). Review component errors."
            )
        
        if not self.report.telemetry_active:
            self.report.recommendations.append(
                "Telemetry systems not fully operational. Monitor system visibility may be limited."
            )
        
        if not self.report.ooda_loop_functional:
            self.report.recommendations.append(
                "OODA loop not functional. System may not respond properly to events."
            )
        
        if not self.report.governance_functional:
            self.report.recommendations.append(
                "Governance system not fully functional. Security and decision-making may be compromised."
            )

def print_health_report(report: SystemHealthReport):
    """Print formatted health report."""
    print("\n" + "="*80)
    print("🏥 GRACE SYSTEM HEALTH REPORT")
    print("="*80)
    print(f"📅 Timestamp: {report.timestamp}")
    print(f"🎯 Overall Status: {report.overall_status.upper()}")
    print(f"📡 Telemetry Active: {'✅' if report.telemetry_active else '❌'}")
    print(f"🔄 OODA Loop Functional: {'✅' if report.ooda_loop_functional else '❌'}")
    print(f"⚖️  Governance Functional: {'✅' if report.governance_functional else '❌'}")
    
    print("\n📊 COMPONENT STATUS SUMMARY:")
    print("-" * 60)
    
    status_symbols = {
        "healthy": "✅",
        "degraded": "⚠️ ",
        "unhealthy": "❌",
        "missing": "❓"
    }
    
    for component in sorted(report.components, key=lambda x: x.name):
        symbol = status_symbols.get(component.status, "❓")
        response_time = f" ({component.response_time:.3f}s)" if component.response_time else ""
        print(f"{symbol} {component.name:<20} {component.status.upper():<10}{response_time}")
        if component.error:
            print(f"    └─ Error: {component.error}")
    
    print(f"\n🏗️  LAYER STATUS:")
    print("-" * 40)
    for layer, status in report.layers_status.items():
        symbol = status_symbols.get(status, "❓")
        print(f"{symbol} {layer:<20} {status.upper()}")
    
    print(f"\n🏛️  VAULT STATUS:")
    print("-" * 40)
    for vault, status in report.vaults_status.items():
        symbol = status_symbols.get(status, "❓")
        print(f"{symbol} {vault:<20} {status.upper()}")
    
    if report.recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 50)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")
    
    print("\n" + "="*80)

async def main():
    """Main entry point for system health check."""
    try:
        checker = GraceSystemHealthChecker()
        report = await checker.run_comprehensive_check()
        
        # Print report to console
        print_health_report(report)
        
        # Save report to file
        report_file = f"system_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        print(f"\n📄 Full report saved to: {report_file}")
        
        # Exit with appropriate code
        if report.overall_status == "healthy":
            sys.exit(0)
        elif report.overall_status == "degraded":
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        print(f"\n❌ CRITICAL ERROR: {e}")
        sys.exit(3)

if __name__ == "__main__":
    asyncio.run(main())