#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
governance_kernel.py
Grace Governance Kernel — Main integration and initialization (production).

Responsibilities
- Wire core infra (EventBus, MemoryCore)
- Bring up governance components (Verification, UnifiedLogic, GovernanceEngine, Parliament, Trust)
- Register with Trigger Mesh, AVN health, Immutable Logs, MLDL quorum
- Provide async start/shutdown lifecycle
- Provide synchronous-API compatibility (evaluate)
- Offer a robust process_governance_request with correlation-scoped wait

Notes
- UTC-safe timestamps
- Defensive imports (some bridges may be optional in certain builds)
- Uses GovernanceEngine.async_init() to attach subscriptions safely
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Core infrastructure
from ..core import EventBus, MemoryCore

# Governance components (same package)
from .verification_engine import VerificationEngine
from .unified_logic import UnifiedLogic
from .governance_engine import GovernanceEngine, GovernanceEngineConfig
from .parliament import Parliament
from .trust_core_kernel import TrustCoreKernel

# Optional/extended components (these paths are project-specific; keep guarded)
try:
    from .policy_engine import PolicyEngine
except Exception:  # pragma: no cover
    PolicyEngine = None  # type: ignore

try:
    from .verification_bridge import VerificationBridge
except Exception:  # pragma: no cover
    VerificationBridge = None  # type: ignore

try:
    from .quorum_bridge import QuorumBridge
except Exception:  # pragma: no cover
    QuorumBridge = None  # type: ignore

try:
    from .synthesizer import Synthesizer
except Exception:  # pragma: no cover
    Synthesizer = None  # type: ignore

# Supporting infra (guarded imports)
try:
    from grace.mtl_kernel.immutable_log_service import ImmutableLogService
except Exception:  # pragma: no cover
    ImmutableLogService = None  # type: ignore

try:
    from grace.multi_os.bridges.mesh_bridge import MeshBridge
except Exception:  # pragma: no cover
    MeshBridge = None  # type: ignore

try:
    from ..immune import EnhancedAVNCore
except Exception:  # pragma: no cover
    EnhancedAVNCore = None  # type: ignore

try:
    from ..mldl import MLDLQuorum
except Exception:  # pragma: no cover
    MLDLQuorum = None  # type: ignore


logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GraceGovernanceKernel:
    """
    Main Grace Governance Kernel that orchestrates all components.
    Implements the complete governance architecture with specialists,
    event routing, audit trails, and health monitoring.
    """

    def __init__(
        self,
        mtl_kernel: Optional[Any] = None,
        intelligence_kernel: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # New-style single-arg (config dict) compatibility
        if isinstance(mtl_kernel, dict) and intelligence_kernel is None and config is None:
            config = mtl_kernel
            mtl_kernel = None

        self.config = config or {}
        self.components: Dict[str, Any] = {}
        self.is_initialized = False
        self.is_running = False

        # Optional external kernels (for legacy/eval compat)
        self.mtl_kernel = mtl_kernel
        self.intelligence_kernel = intelligence_kernel

        # Legacy compatibility shims (guarded)
        self.policy_engine = PolicyEngine() if PolicyEngine else None
        self.verification_bridge = VerificationBridge() if VerificationBridge else None
        self.quorum_bridge = QuorumBridge(self.intelligence_kernel) if QuorumBridge else None
        self.synthesizer = Synthesizer() if Synthesizer else None

    # -------------------- Public Lifecycle --------------------

    async def initialize(self) -> None:
        """Initialize all governance kernel components."""
        if self.is_initialized:
            logger.warning("Governance kernel already initialized")
            return

        logger.info("Initializing Grace Governance Kernel...")
        try:
            await self._initialize_core_infrastructure()
            await self._initialize_governance_components()
            await self._initialize_support_systems()
            await self._setup_component_integration()
            self.is_initialized = True
            logger.info("Grace Governance Kernel initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize governance kernel: %s", e)
            raise

    async def start(self) -> None:
        """Start the governance kernel."""
        if not self.is_initialized:
            await self.initialize()
        if self.is_running:
            logger.warning("Governance kernel already running")
            return

        logger.info("Starting Grace Governance Kernel...")
        try:
            self.is_running = True

            # Bring GovernanceEngine online (attach subscriptions)
            ge: GovernanceEngine = self.components["governance_engine"]
            await ge.async_init()

            # Initial snapshot
            snapshot = await ge.create_snapshot()
            logger.info("Initial snapshot created: %s", snapshot.snapshot_id)

            # Immutable audit of startup (if available)
            if self.components.get("immutable_logs"):
                await self.components["immutable_logs"].log_governance_action(  # type: ignore[call-arg]
                    "kernel_startup",
                    {
                        "timestamp": snapshot.created_at.isoformat(),
                        "snapshot_id": snapshot.snapshot_id,
                        "components_initialized": list(self.components.keys()),
                    },
                    "democratic_oversight",
                )

            logger.info("Grace Governance Kernel started successfully")
        except Exception as e:
            self.is_running = False
            logger.exception("Failed to start governance kernel: %s", e)
            raise

    async def shutdown(self) -> None:
        """Shutdown the governance kernel gracefully."""
        if not self.is_running:
            logger.warning("Governance kernel not running")
            return

        logger.info("Shutting down Grace Governance Kernel...")
        try:
            # Final snapshot (best-effort)
            ge: Optional[GovernanceEngine] = self.components.get("governance_engine")
            snapshot_id = None
            ts = _iso_now()
            if ge:
                snap = await ge.create_snapshot()
                snapshot_id = snap.snapshot_id
                ts = snap.created_at.isoformat()

            # Immutable log (best-effort)
            if self.components.get("immutable_logs"):
                await self.components["immutable_logs"].log_governance_action(  # type: ignore[call-arg]
                    "kernel_shutdown",
                    {"timestamp": ts, "snapshot_id": snapshot_id, "reason": "graceful_shutdown"},
                    "democratic_oversight",
                )

            # Close MemoryCore (SQLite cleanup is internal; still emit log)
            mc: Optional[MemoryCore] = self.components.get("memory_core")
            if mc:
                mc.close()

            # Close EventBus (drain and stop queue workers if any)
            bus: Optional[EventBus] = self.components.get("event_bus")
            if bus:
                await bus.aclose()

            self.is_running = False
            logger.info("Grace Governance Kernel shutdown complete")
        except Exception as e:
            logger.exception("Error during governance kernel shutdown: %s", e)

    # -------------------- Processing API --------------------

    async def process_governance_request(
        self,
        decision_subject: str,
        inputs: Dict[str, Any],
        thresholds: Optional[Dict[str, float]] = None,
        *,
        timeout_s: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Process a governance validation request and await the correlated verdict.

        Returns:
            Governance decision payload (APPROVED/REJECTED/NEEDS_REVIEW) or a timeout/error structure.
        """
        if not self.is_running:
            raise RuntimeError("Governance kernel not running")

        try:
            trigger_mesh: Optional[Any] = self.components.get("trigger_mesh")
            event_bus: EventBus = self.components["event_bus"]

            # Fallback path: publish directly via EventBus if Trigger Mesh is not available
            if trigger_mesh is None:
                correlation_id = await event_bus.publish(
                    "GOVERNANCE_VALIDATION",
                    {
                        "decision_subject": decision_subject,
                        "inputs": inputs,
                        "thresholds": thresholds or {},
                    },
                )
            else:
                correlation_id = await trigger_mesh.route_event(  # type: ignore[assignment]
                    "GOVERNANCE_VALIDATION",
                    {
                        "decision_subject": decision_subject,
                        "inputs": inputs,
                        "thresholds": thresholds or {},
                    },
                    correlation_id=None,
                    source_component="external_request",
                )

            # Wait for a verdict event with matching correlation_id (robust, no polling)
            verdict = await self._await_verdict(event_bus, correlation_id, timeout_s=timeout_s)
            if verdict is not None:
                return verdict

            return {
                "correlation_id": correlation_id,
                "outcome": "GOVERNANCE_TIMEOUT",
                "rationale": "Request processing timeout",
                "timestamp": _iso_now(),
            }

        except Exception as e:
            logger.exception("Error processing governance request: %s", e)
            return {
                "outcome": "GOVERNANCE_ERROR",
                "rationale": f"Processing error: {e}",
                "error": True,
                "timestamp": _iso_now(),
            }

    # -------------------- Legacy Compatibility --------------------

    def evaluate(self, request: Any) -> Dict[str, Any]:
        """
        Synchronous governance evaluation pipeline compatible with the original GovernanceKernel.

        Pipeline:
          1) policy_engine.check(request)
          2) verification_bridge.verify(request)
          3) feed = mtl.feed_for_quorum(filters_from(request)) [if available]
          4) result = quorum_bridge.consensus(feed)
          5) decision = synthesizer.merge(request, results)
          6) mtl.store_decision(decision) [if available]
        """
        try:
            # Step 1
            policy_results = self.policy_engine.check(request) if self.policy_engine else []
            # Step 2
            verification_result = (
                self.verification_bridge.verify(request) if self.verification_bridge else {"verified": True}
            )
            # Step 3 & 4
            quorum_consensus = None
            requires_quorum = self._check_requires_quorum(request, policy_results)
            if requires_quorum and self.mtl_kernel and self.quorum_bridge:
                filters = self._filters_from_request(request)
                feed_ids = self.mtl_kernel.feed_for_quorum(filters)
                quorum_consensus = self.quorum_bridge.consensus(feed_ids, context={"request": request})

            # Step 5
            if self.synthesizer:
                decision = self.synthesizer.merge(
                    request=request,
                    policy_results=policy_results,
                    verification_result=verification_result,
                    quorum_consensus=quorum_consensus,
                )
            else:
                # Minimal fallback
                decision = {
                    "request_id": self._extract_request_id(request),
                    "approved": bool(verification_result),
                    "confidence": 0.8 if verification_result else 0.0,
                    "reasoning": "Fallback synthesizer path",
                }

            # Step 6
            if self.mtl_kernel:
                try:
                    self.mtl_kernel.store_decision(decision)
                except Exception as e:
                    logger.warning("Failed to store decision in MTL: %s", e)

            return decision

        except Exception as e:
            logger.exception("Error in synchronous governance evaluation: %s", e)
            return {
                "request_id": self._extract_request_id(request),
                "approved": False,
                "confidence": 0.0,
                "reasoning": f"Evaluation error: {e}",
                "error": True,
            }

    # -------------------- Internal Initialization --------------------

    async def _initialize_core_infrastructure(self) -> None:
        logger.info("Initializing core infrastructure...")
        # Event bus for system-wide communication
        self.components["event_bus"] = EventBus()
        # Memory core
        memory_db_path = self.config.get("memory_db_path", "grace_governance.db")
        self.components["memory_core"] = MemoryCore(memory_db_path)
        logger.info("Core infrastructure initialized")

    async def _initialize_governance_components(self) -> None:
        logger.info("Initializing governance components...")
        bus: EventBus = self.components["event_bus"]
        mem: MemoryCore = self.components["memory_core"]

        # Engines
        self.components["verification_engine"] = VerificationEngine(bus, mem)
        self.components["unified_logic"] = UnifiedLogic(bus, mem)
        self.components["trust_core"] = TrustCoreKernel(bus, mem)
        self.components["parliament"] = Parliament(bus, mem)

        # Governance engine (orchestrator)
        # Allow thresholds from config to flow into the engine
        ge_cfg = GovernanceEngineConfig(
            instance_id=self.config.get("instance_id", "governance-primary"),
            version=self.config.get("version", "1.0.0"),
            policies=self.config.get("policies"),
            thresholds=self.config.get("governance_thresholds"),
        )
        self.components["governance_engine"] = GovernanceEngine(
            bus,
            mem,
            self.components["verification_engine"],
            self.components["unified_logic"],
            avn_core=None,
            config=ge_cfg,
        )

        logger.info("Governance components initialized")

    async def _initialize_support_systems(self) -> None:
        logger.info("Initializing support systems...")
        bus: EventBus = self.components["event_bus"]
        mem: MemoryCore = self.components["memory_core"]

        # Immutable audit logs (optional)
        if ImmutableLogService:
            self.components["immutable_logs"] = ImmutableLogService(mem)  # type: ignore[call-arg]

        # Trigger mesh (optional)
        if MeshBridge:
            self.components["trigger_mesh"] = MeshBridge(bus)  # type: ignore[call-arg]

        # Enhanced AVN core (optional)
        if EnhancedAVNCore:
            self.components["avn_core"] = EnhancedAVNCore(bus, mem)  # type: ignore[call-arg]

        # MLDL quorum (optional)
        if MLDLQuorum:
            self.components["mldl_quorum"] = MLDLQuorum(bus, mem)  # type: ignore[call-arg]

        logger.info("Support systems initialized")

    async def _setup_component_integration(self) -> None:
        logger.info("Setting up component integration...")

        trigger_mesh: Optional[Any] = self.components.get("trigger_mesh")
        if trigger_mesh:
            # Governance engine routes
            trigger_mesh.register_component("governance_engine", "governance", ["GOVERNANCE_VALIDATION", "GOVERNANCE_ROLLBACK"])
            trigger_mesh.register_component("verification_engine", "governance", ["CLAIM_VERIFICATION"])
            trigger_mesh.register_component("unified_logic", "governance", ["CONSENSUS_REQUEST"])
            trigger_mesh.register_component("parliament", "governance", ["GOVERNANCE_NEEDS_REVIEW", "PARLIAMENT_VOTE_CAST"])
            trigger_mesh.register_component("trust_core", "trust", ["TRUST_UPDATED"])
            # Audit routes
            trigger_mesh.register_component(
                "immutable_logs", "audit", ["GOVERNANCE_APPROVED", "GOVERNANCE_REJECTED", "GOVERNANCE_SNAPSHOT_CREATED"]
            )
            # Health & ML
            trigger_mesh.register_component("avn_core", "health", ["ANOMALY_DETECTED", "COMPONENT_FAILOVER"])
            trigger_mesh.register_component("mldl_quorum", "ml", ["MLDL_CONSENSUS_REQUEST"])

        # AVN registrations (optional)
        avn = self.components.get("avn_core")
        if avn:
            for comp in ("governance_engine", "verification_engine", "unified_logic", "parliament", "trust_core", "mldl_quorum"):
                if self.components.get(comp):
                    avn.register_component(comp)

            # Give GovernanceEngine a handle to AVN
            self.components["governance_engine"].avn_core = avn

        logger.info("Component integration completed")

    # -------------------- Helpers --------------------

    async def _await_verdict(
        self,
        event_bus: EventBus,
        correlation_id: str,
        *,
        timeout_s: float = 10.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Await the first verdict event (APPROVED/REJECTED/NEEDS_REVIEW) matching correlation_id.
        Uses a one-shot subscription to avoid polling.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        async def _handler(evt: Dict[str, Any]) -> None:
            try:
                if evt.get("correlation_id") == correlation_id:
                    # Match found; resolve once
                    if not fut.done():
                        fut.set_result(evt.get("payload"))
            except Exception as e:
                if not fut.done():
                    fut.set_exception(e)

        # Subscribe once to all three terminal outcomes
        await event_bus.subscribe("GOVERNANCE_APPROVED", _handler)
        await event_bus.subscribe("GOVERNANCE_REJECTED", _handler)
        await event_bus.subscribe("GOVERNANCE_NEEDS_REVIEW", _handler)

        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError:
            return None
        finally:
            # Best-effort cleanup
            try:
                await event_bus.unsubscribe("GOVERNANCE_APPROVED", _handler)
                await event_bus.unsubscribe("GOVERNANCE_REJECTED", _handler)
                await event_bus.unsubscribe("GOVERNANCE_NEEDS_REVIEW", _handler)
            except Exception:
                pass

    def _check_requires_quorum(self, request: Any, policy_results: List[Dict[str, Any]]) -> bool:
        """Check if request requires quorum based on policy results or request flag."""
        for result in policy_results:
            if result.get("requires_quorum", False):
                return True
        if hasattr(request, "requires_quorum"):
            return bool(getattr(request, "requires_quorum"))
        if isinstance(request, dict):
            return bool(request.get("requires_quorum", False))
        return False

    def _filters_from_request(self, request: Any) -> Dict[str, Any]:
        """Generate MTL query filters from governance request."""
        filters: Dict[str, Any] = {}
        # request_type
        if hasattr(request, "request_type"):
            filters["request_type"] = request.request_type
        elif isinstance(request, dict):
            filters["request_type"] = request.get("request_type", "unknown")
        # tags
        if hasattr(request, "tags"):
            filters["tags"] = request.tags
        elif isinstance(request, dict):
            filters["tags"] = request.get("tags", [])
        # domains
        if hasattr(request, "policy_domains"):
            filters["domains"] = request.policy_domains
        elif isinstance(request, dict):
            filters["domains"] = request.get("policy_domains", [])
        return filters

    def _extract_request_id(self, request: Any) -> str:
        """Extract request ID from object/dict forms."""
        if hasattr(request, "id"):
            return str(getattr(request, "id"))
        if hasattr(request, "request_id"):
            return str(getattr(request, "request_id"))
        if isinstance(request, dict):
            return str(request.get("id") or request.get("request_id") or "unknown")
        return "unknown"

    # -------------------- Diagnostics --------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get governance kernel statistics (compat)."""
        return {
            "status": "running" if self.is_running else ("initialized" if self.is_initialized else "created"),
            "components_initialized": len(self.components),
            "policy_engine_policies": len(getattr(self.policy_engine, "policies", [])) if self.policy_engine else 0,
            "verification_methods": len(getattr(self.verification_bridge, "verification_methods", []))
            if self.verification_bridge
            else 0,
            "has_mtl_kernel": self.mtl_kernel is not None,
            "has_intelligence_kernel": self.intelligence_kernel is not None,
        }

    def set_mtl_kernel(self, mtl_kernel: Any) -> None:
        """Set the MTL kernel for decision storage and feed generation."""
        self.mtl_kernel = mtl_kernel

    def set_intelligence_kernel(self, intelligence_kernel: Any) -> None:
        """Set the intelligence kernel for quorum operations."""
        self.intelligence_kernel = intelligence_kernel
        if self.quorum_bridge and hasattr(self.quorum_bridge, "set_intelligence_kernel"):
            self.quorum_bridge.set_intelligence_kernel(intelligence_kernel)

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and component health (if AVN is available)."""
        if not self.is_initialized:
            return {"status": "not_initialized"}

        status: Dict[str, Any] = {
            "status": "running" if self.is_running else "initialized",
            "components": {},
            "system_health": {},
        }

        for name, component in self.components.items():
            if hasattr(component, "get_status") and callable(component.get_status):
                try:
                    status["components"][name] = component.get_status()
                except Exception:
                    status["components"][name] = {"status": "unknown_error"}
            elif hasattr(component, "health_check"):
                status["components"][name] = {"has_health_check": True}
            else:
                status["components"][name] = {"status": "active"}

        if self.components.get("avn_core") and hasattr(self.components["avn_core"], "get_system_health_summary"):
            try:
                status["system_health"] = self.components["avn_core"].get_system_health_summary()
            except Exception:
                status["system_health"] = {"status": "unknown_error"}

        return status

    def get_governance_metrics(self) -> Dict[str, Any]:
        """Get governance performance metrics from sub-systems where available."""
        metrics: Dict[str, Any] = {}

        tm = self.components.get("trigger_mesh")
        if tm and hasattr(tm, "get_routing_metrics"):
            try:
                metrics["routing"] = tm.get_routing_metrics()
            except Exception:
                metrics["routing"] = {"status": "unknown_error"}

        parl = self.components.get("parliament")
        if parl and hasattr(parl, "get_member_stats"):
            try:
                metrics["parliament"] = parl.get_member_stats()
            except Exception:
                metrics["parliament"] = {"status": "unknown_error"}

        quorum = self.components.get("mldl_quorum")
        if quorum and hasattr(quorum, "get_quorum_status"):
            try:
                metrics["mldl_quorum"] = quorum.get_quorum_status()
            except Exception:
                metrics["mldl_quorum"] = {"status": "unknown_error"}

        imm = self.components.get("immutable_logs")
        if imm and hasattr(imm, "get_audit_statistics"):
            try:
                metrics["audit"] = imm.get_audit_statistics()
            except Exception:
                metrics["audit"] = {"status": "unknown_error"}

        trust = self.components.get("trust_core")
        if trust and hasattr(trust, "get_trust_statistics"):
            try:
                metrics["trust"] = trust.get_trust_statistics()
            except Exception:
                metrics["trust"] = {"status": "unknown_error"}

        return metrics


# -------------------- CLI runner (optional) --------------------

async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    kernel = GraceGovernanceKernel()
    try:
        await kernel.start()

        # Example request
        result = await kernel.process_governance_request(
            "policy",
            {
                "claims": [
                    {
                        "id": "claim_001",
                        "statement": "This policy update improves system security",
                        "sources": [{"uri": "https://security-report.example.com", "credibility": 0.8}],
                        "evidence": [{"type": "doc", "pointer": "security_audit_2024.pdf"}],
                        "confidence": 0.85,
                        "logical_chain": [{"step": "Security audit identified vulnerabilities"}],
                    }
                ],
                "context": {"decision_type": "policy", "urgency": "normal"},
            },
        )
        print("Governance Decision Result:", result)

        status = kernel.get_system_status()
        print("System Status:", status.get("status"))

        print("Governance kernel running... (Press Ctrl+C to stop)")
        await asyncio.sleep(60)

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await kernel.shutdown()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
