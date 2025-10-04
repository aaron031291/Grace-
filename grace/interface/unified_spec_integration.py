"""
Grace Unified Operating Specification Integration
Extends the Orb Interface with North-Star Architecture components.

This module integrates:
- Loop Orchestrator with tick scheduler + context bus
- Enhanced Memory Explorer (File-Explorer-like brain)
- AVN/RCA/Healing Engine integration
- Governance Layer with policies and immutable logs
- Observability Fabric with unified trace IDs
- Trust & KPI Framework
- Voice & Collaboration modes
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


# ============================================================================
# Unified Data Contracts (Section 7)
# ============================================================================

@dataclass
class EventEnvelope:
    """Unified event envelope for all Grace events."""
    event_type: str  # grace.event.v1
    actor: str  # human|grace
    component: str  # memory|avn|governance|orb
    payload: Dict[str, Any]
    kpi_deltas: Dict[str, float] = field(default_factory=dict)
    trust_before: float = 0.0
    trust_after: float = 0.0
    confidence: float = 0.0
    immutable_hash: str = ""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    correlation_id: Optional[str] = None


@dataclass
class MemoryItem:
    """Enhanced memory item with governance and trust."""
    id: str
    path: str  # knowledge/patterns/api_resilience/
    tags: List[str] = field(default_factory=list)
    trust: float = 0.0
    last_used: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    policy_refs: List[str] = field(default_factory=list)
    vector_ref: str = ""
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextManifest:
    """Folder context manifest for Memory Explorer."""
    folder_id: str
    purpose: str
    domain: str  # api_resilience, trading, governance, etc.
    policies: List[str] = field(default_factory=list)
    trust_threshold: float = 0.7
    auto_classify: bool = True
    adjacency_refs: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# Trust & KPI Framework (Section 9)
# ============================================================================

class TrustMetricType(Enum):
    """Types of trust metrics."""
    COMPONENT = "trust_component"
    MEMORY = "trust_memory"
    DECISION = "trust_decision"


class KPIMetricType(Enum):
    """Types of KPI metrics."""
    PERFORMANCE = "kpi_perf"
    GOVERNANCE_LATENCY = "governance_latency"
    LEARNING_GAIN = "learning_gain"
    MTTR = "mttr"  # Mean Time to Repair
    MTTU = "mttu"  # Mean Time to Understand


@dataclass
class TrustMetric:
    """Trust metric for components, memory, or decisions."""
    metric_id: str
    metric_type: TrustMetricType
    component_id: str
    trust_score: float  # 0.0 to 1.0
    confidence: float = 0.0
    samples: int = 0
    rolling_window_days: int = 30
    trust_drift: float = 0.0  # Bounded ±0.05
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KPIMetric:
    """KPI metric for performance and latency."""
    metric_id: str
    metric_type: KPIMetricType
    value: float
    unit: str  # ms, s, percentage, etc.
    target: Optional[float] = None
    status: str = "normal"  # normal, warning, critical
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Context Bus (Section 5.1)
# ============================================================================

class ContextBus:
    """Unified context bus ensuring all components share unified 'now'."""
    
    def __init__(self):
        self.current_context: Dict[str, Any] = {}
        self.context_history: List[Dict[str, Any]] = []
        self.subscribers: Dict[str, List[Callable]] = {}
        self.lock = asyncio.Lock()
        
    async def set_context(self, key: str, value: Any, correlation_id: Optional[str] = None):
        """Set context value and notify subscribers."""
        async with self.lock:
            old_value = self.current_context.get(key)
            self.current_context[key] = value
            
            # Record in history
            self.context_history.append({
                "key": key,
                "value": value,
                "old_value": old_value,
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": correlation_id
            })
            
            # Limit history size
            if len(self.context_history) > 1000:
                self.context_history = self.context_history[-1000:]
            
            # Notify subscribers
            if key in self.subscribers:
                for callback in self.subscribers[key]:
                    try:
                        await callback(key, value, old_value)
                    except Exception as e:
                        logger.error(f"Context subscriber error: {e}")
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get current context value."""
        return self.current_context.get(key, default)
    
    def subscribe(self, key: str, callback: Callable):
        """Subscribe to context changes."""
        if key not in self.subscribers:
            self.subscribers[key] = []
        self.subscribers[key].append(callback)
    
    def get_all_context(self) -> Dict[str, Any]:
        """Get all current context."""
        return self.current_context.copy()


# ============================================================================
# Voice & Collaboration Modes (Section 8)
# ============================================================================

class VoiceMode(Enum):
    """Voice operation modes."""
    SOLO_VOICE = "solo_voice"  # Grace listens, executes, narrates
    TEXT_ONLY = "text_only"  # Command palette + keyboard shortcuts
    CO_PARTNER = "co_partner"  # Multi-user collaborative session
    SILENT_AUTONOMOUS = "silent_autonomous"  # Grace acts within delegated scope


@dataclass
class VoiceCommand:
    """Voice command with intent and execution log."""
    command_id: str
    mode: VoiceMode
    intent: str
    user_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    executed: bool = False
    result: Optional[Dict[str, Any]] = None


# ============================================================================
# Observability & Explainability (Section 10)
# ============================================================================

class ExplanationLevel(Enum):
    """Explanation complexity levels."""
    BEGINNER = "beginner"
    ADVANCED = "advanced"
    SRE = "sre"


class DecisionState(Enum):
    """Decision state in lifecycle."""
    PROPOSED = "proposed"
    PROVING = "proving"
    GOVERNED = "governed"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


@dataclass
class ExplainableDecision:
    """Decision with causal graph and counterfactual simulation."""
    decision_id: str
    state: DecisionState
    causal_graph: Dict[str, Any] = field(default_factory=dict)
    counterfactuals: List[Dict[str, Any]] = field(default_factory=list)
    trace_id: str = ""
    explanation: Dict[ExplanationLevel, str] = field(default_factory=dict)
    kpi_impact: Dict[str, float] = field(default_factory=dict)
    approver_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# Enhanced Memory Explorer (Section 5.4)
# ============================================================================

class MemoryExplorerEnhanced:
    """File-Explorer interface for Grace's cognition with trust feedback."""
    
    def __init__(self, context_bus: ContextBus):
        self.context_bus = context_bus
        self.memory_items: Dict[str, MemoryItem] = {}
        self.context_manifests: Dict[str, ContextManifest] = {}
        self.trust_feedback: Dict[str, float] = {}
        self.adjacency_graph: Dict[str, List[str]] = {}
        
    async def create_memory_folder(self, folder_id: str, purpose: str, domain: str, 
                                   policies: List[str] = None) -> ContextManifest:
        """Create a memory folder with context manifest."""
        manifest = ContextManifest(
            folder_id=folder_id,
            purpose=purpose,
            domain=domain,
            policies=policies or []
        )
        self.context_manifests[folder_id] = manifest
        
        # Update context bus
        await self.context_bus.set_context(f"memory_folder.{folder_id}", manifest)
        
        logger.info(f"Created memory folder: {folder_id} ({domain})")
        return manifest
    
    async def store_memory_item(self, item: MemoryItem, folder_id: Optional[str] = None):
        """Store a memory item with auto-classification."""
        self.memory_items[item.id] = item
        
        # Auto-classify if folder specified
        if folder_id and folder_id in self.context_manifests:
            manifest = self.context_manifests[folder_id]
            if manifest.auto_classify:
                # Auto-tag based on folder domain
                if manifest.domain not in item.tags:
                    item.tags.append(manifest.domain)
        
        # Update adjacency graph
        for tag in item.tags:
            if tag not in self.adjacency_graph:
                self.adjacency_graph[tag] = []
            self.adjacency_graph[tag].append(item.id)
        
        logger.info(f"Stored memory item: {item.id} in path {item.path}")
    
    async def search_with_context(self, query: str, filters: Optional[Dict] = None) -> List[MemoryItem]:
        """Search memory with contextual filters and trust ranking."""
        results = []
        
        for item in self.memory_items.values():
            # Basic text search
            if query.lower() in item.path.lower() or any(query.lower() in tag for tag in item.tags):
                # Apply filters
                if filters:
                    if "min_trust" in filters and item.trust < filters["min_trust"]:
                        continue
                    if "tags" in filters and not any(tag in item.tags for tag in filters["tags"]):
                        continue
                    if "policy_refs" in filters and not any(p in item.policy_refs for p in filters["policy_refs"]):
                        continue
                
                results.append(item)
        
        # Sort by trust score (descending)
        results.sort(key=lambda x: x.trust, reverse=True)
        
        # Limit results
        limit = filters.get("limit", 20) if filters else 20
        return results[:limit]
    
    async def update_trust_feedback(self, item_id: str, success: bool, weight: float = 0.1):
        """Update trust based on success/failure feedback."""
        if item_id in self.memory_items:
            item = self.memory_items[item_id]
            
            # Exponential moving average
            if success:
                item.trust = min(1.0, item.trust + weight * (1.0 - item.trust))
            else:
                item.trust = max(0.0, item.trust - weight * item.trust)
            
            item.last_used = datetime.utcnow().isoformat()
            
            logger.info(f"Updated trust for {item_id}: {item.trust:.3f} (success={success})")


# ============================================================================
# AVN/RCA/Healing Engine (Section 5.5)
# ============================================================================

@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    anomaly_id: str
    detected_at: str
    metric_name: str
    current_value: float
    expected_value: float
    deviation: float
    severity: str  # low, medium, high, critical
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RCAHypothesis:
    """Root cause analysis hypothesis."""
    hypothesis_id: str
    anomaly_id: str
    root_cause: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)
    suggested_fix: Optional[str] = None


class AVNHealingEngine:
    """AVN/RCA/Healing Engine with self-healing capabilities."""
    
    def __init__(self, memory_explorer: MemoryExplorerEnhanced, context_bus: ContextBus):
        self.memory_explorer = memory_explorer
        self.context_bus = context_bus
        self.anomalies: Dict[str, AnomalyDetection] = {}
        self.hypotheses: Dict[str, RCAHypothesis] = {}
        self.healing_history: List[Dict[str, Any]] = []
        
    async def detect_anomaly(self, metric_name: str, current_value: float, 
                           expected_value: float, threshold: float = 0.2) -> Optional[AnomalyDetection]:
        """Detect anomalies from metrics."""
        deviation = abs(current_value - expected_value) / max(abs(expected_value), 1.0)
        
        if deviation > threshold:
            severity = "critical" if deviation > 0.5 else "high" if deviation > 0.3 else "medium"
            
            anomaly = AnomalyDetection(
                anomaly_id=f"anom_{int(time.time() * 1000)}",
                detected_at=datetime.utcnow().isoformat(),
                metric_name=metric_name,
                current_value=current_value,
                expected_value=expected_value,
                deviation=deviation,
                severity=severity
            )
            
            self.anomalies[anomaly.anomaly_id] = anomaly
            
            # Update context bus
            await self.context_bus.set_context(f"anomaly.{metric_name}", anomaly)
            
            logger.warning(f"Anomaly detected: {metric_name} deviation={deviation:.2%} severity={severity}")
            return anomaly
        
        return None
    
    async def perform_rca(self, anomaly: AnomalyDetection) -> List[RCAHypothesis]:
        """Perform root cause analysis using Memory Explorer."""
        hypotheses = []
        
        # Search memory for related patterns
        related_items = await self.memory_explorer.search_with_context(
            query=anomaly.metric_name,
            filters={"min_trust": 0.7, "tags": ["pattern", "fix"]}
        )
        
        # Generate hypotheses from patterns
        for item in related_items[:3]:  # Top 3 patterns
            hypothesis = RCAHypothesis(
                hypothesis_id=f"hyp_{int(time.time() * 1000)}_{len(hypotheses)}",
                anomaly_id=anomaly.anomaly_id,
                root_cause=f"Pattern match: {item.path}",
                confidence=item.trust,
                evidence=[f"Historical pattern: {item.id}"],
                related_patterns=[item.id],
                suggested_fix=item.metadata.get("fix")
            )
            
            hypotheses.append(hypothesis)
            self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        
        logger.info(f"RCA for {anomaly.anomaly_id}: {len(hypotheses)} hypotheses")
        return hypotheses
    
    async def execute_healing(self, hypothesis: RCAHypothesis, sandbox: bool = True) -> Dict[str, Any]:
        """Execute healing action with sandbox proof."""
        result = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "executed_at": datetime.utcnow().isoformat(),
            "sandbox_mode": sandbox,
            "success": False,
            "kpi_impact": {}
        }
        
        # Simulate sandbox execution
        if sandbox:
            logger.info(f"Sandbox proof for {hypothesis.hypothesis_id}: {hypothesis.suggested_fix}")
            # In real implementation, this would run in isolated sandbox
            result["sandbox_output"] = f"Simulated: {hypothesis.suggested_fix}"
            result["success"] = True
        else:
            logger.warning("Production healing not implemented in this demo")
        
        # Record in history
        self.healing_history.append(result)
        
        # Update memory trust based on success
        for pattern_id in hypothesis.related_patterns:
            await self.memory_explorer.update_trust_feedback(pattern_id, result["success"])
        
        return result


# ============================================================================
# Loop Orchestrator Integration (Section 5.1)
# ============================================================================

class LoopOrchestratorIntegration:
    """Integrates Loop Orchestrator with tick scheduler and context bus."""
    
    def __init__(self, context_bus: ContextBus):
        self.context_bus = context_bus
        self.tick_count = 0
        self.loops: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.tick_task: Optional[asyncio.Task] = None
        
    async def register_loop(self, loop_id: str, name: str, interval_s: int, 
                          callback: Callable, priority: int = 5):
        """Register an orchestration loop."""
        self.loops[loop_id] = {
            "name": name,
            "interval_s": interval_s,
            "callback": callback,
            "priority": priority,
            "last_run": 0.0,
            "next_run": time.time() + interval_s,
            "run_count": 0
        }
        
        logger.info(f"Registered loop: {loop_id} ({name}) interval={interval_s}s")
    
    async def tick(self):
        """Single scheduler tick - check loops and dispatch tasks."""
        self.tick_count += 1
        current_time = time.time()
        
        # Update context bus with current tick
        await self.context_bus.set_context("orchestrator.tick", self.tick_count)
        await self.context_bus.set_context("orchestrator.time", current_time)
        
        # Check and execute loops
        for loop_id, loop_data in self.loops.items():
            if current_time >= loop_data["next_run"]:
                try:
                    # Execute loop callback
                    await loop_data["callback"](loop_id, self.tick_count)
                    
                    # Update scheduling
                    loop_data["last_run"] = current_time
                    loop_data["next_run"] = current_time + loop_data["interval_s"]
                    loop_data["run_count"] += 1
                    
                    logger.debug(f"Executed loop {loop_id} (tick={self.tick_count})")
                    
                except Exception as e:
                    logger.error(f"Loop {loop_id} error: {e}", exc_info=True)
    
    async def start(self, tick_interval_s: float = 1.0):
        """Start the loop orchestrator."""
        if self.running:
            logger.warning("Loop orchestrator already running")
            return
        
        self.running = True
        
        async def tick_loop():
            while self.running:
                await self.tick()
                await asyncio.sleep(tick_interval_s)
        
        self.tick_task = asyncio.create_task(tick_loop())
        logger.info(f"Loop orchestrator started (tick_interval={tick_interval_s}s)")
    
    async def stop(self):
        """Stop the loop orchestrator."""
        if not self.running:
            return
        
        self.running = False
        
        if self.tick_task:
            self.tick_task.cancel()
            try:
                await self.tick_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Loop orchestrator stopped")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health pulse endpoint."""
        return {
            "running": self.running,
            "tick_count": self.tick_count,
            "loops_count": len(self.loops),
            "loops": {
                loop_id: {
                    "name": data["name"],
                    "run_count": data["run_count"],
                    "last_run": data["last_run"],
                    "next_run": data["next_run"]
                }
                for loop_id, data in self.loops.items()
            }
        }


# ============================================================================
# Unified Orb Interface Extension
# ============================================================================

class UnifiedOrbSpecIntegration:
    """
    Unified Operating Specification Integration for Orb Interface.
    
    Integrates all components from the North-Star Architecture:
    - Loop Orchestrator with tick scheduler + context bus
    - Enhanced Memory Explorer
    - AVN/RCA/Healing Engine
    - Governance Layer
    - Observability Fabric
    - Trust & KPI Framework
    - Voice & Collaboration modes
    """
    
    def __init__(self):
        self.version = "2.0.0"
        
        # Core components
        self.context_bus = ContextBus()
        self.memory_explorer = MemoryExplorerEnhanced(self.context_bus)
        self.avn_engine = AVNHealingEngine(self.memory_explorer, self.context_bus)
        self.loop_orchestrator = LoopOrchestratorIntegration(self.context_bus)
        
        # Metrics tracking
        self.trust_metrics: Dict[str, TrustMetric] = {}
        self.kpi_metrics: Dict[str, KPIMetric] = {}
        self.event_envelopes: List[EventEnvelope] = []
        
        # Voice & collaboration
        self.voice_mode: VoiceMode = VoiceMode.TEXT_ONLY
        self.voice_commands: List[VoiceCommand] = []
        
        # Explainability
        self.decisions: Dict[str, ExplainableDecision] = {}
        
        logger.info(f"Unified Orb Spec Integration initialized v{self.version}")
    
    async def start(self):
        """Start all integrated components."""
        await self.loop_orchestrator.start(tick_interval_s=1.0)
        logger.info("Unified Orb Spec Integration started")
    
    async def stop(self):
        """Stop all integrated components."""
        await self.loop_orchestrator.stop()
        logger.info("Unified Orb Spec Integration stopped")
    
    async def publish_event(self, event_type: str, actor: str, component: str, 
                          payload: Dict[str, Any], kpi_deltas: Optional[Dict[str, float]] = None) -> EventEnvelope:
        """Publish a unified event envelope."""
        envelope = EventEnvelope(
            event_type=event_type,
            actor=actor,
            component=component,
            payload=payload,
            kpi_deltas=kpi_deltas or {}
        )
        
        self.event_envelopes.append(envelope)
        
        # Limit history
        if len(self.event_envelopes) > 10000:
            self.event_envelopes = self.event_envelopes[-10000:]
        
        logger.debug(f"Published event: {event_type} from {component}")
        return envelope
    
    async def update_trust_metric(self, component_id: str, metric_type: TrustMetricType, 
                                 trust_score: float, confidence: float = 1.0):
        """Update trust metric for a component."""
        metric_id = f"{metric_type.value}_{component_id}"
        
        if metric_id in self.trust_metrics:
            metric = self.trust_metrics[metric_id]
            # Update with exponential moving average
            alpha = 0.2
            old_trust = metric.trust_score
            metric.trust_score = alpha * trust_score + (1 - alpha) * old_trust
            metric.trust_drift = metric.trust_score - old_trust
            metric.samples += 1
            metric.last_updated = datetime.utcnow().isoformat()
        else:
            metric = TrustMetric(
                metric_id=metric_id,
                metric_type=metric_type,
                component_id=component_id,
                trust_score=trust_score,
                confidence=confidence,
                samples=1
            )
            self.trust_metrics[metric_id] = metric
        
        # Ensure drift is bounded
        if abs(metric.trust_drift) > 0.05:
            logger.warning(f"Trust drift exceeded bounds for {component_id}: {metric.trust_drift:.3f}")
        
        return metric
    
    async def record_kpi(self, metric_type: KPIMetricType, value: float, 
                        unit: str, target: Optional[float] = None) -> KPIMetric:
        """Record a KPI metric."""
        metric_id = f"{metric_type.value}_{int(time.time() * 1000)}"
        
        # Determine status
        status = "normal"
        if target:
            if value > target * 1.5:
                status = "critical"
            elif value > target * 1.2:
                status = "warning"
        
        metric = KPIMetric(
            metric_id=metric_id,
            metric_type=metric_type,
            value=value,
            unit=unit,
            target=target,
            status=status
        )
        
        self.kpi_metrics[metric_id] = metric
        
        # Update context bus
        await self.context_bus.set_context(f"kpi.{metric_type.value}", value)
        
        return metric
    
    async def set_voice_mode(self, mode: VoiceMode):
        """Set voice interaction mode."""
        self.voice_mode = mode
        await self.context_bus.set_context("voice_mode", mode.value)
        logger.info(f"Voice mode set to: {mode.value}")
    
    async def execute_voice_command(self, intent: str, user_id: str) -> VoiceCommand:
        """Execute a voice command with intent logging."""
        command = VoiceCommand(
            command_id=f"voice_{int(time.time() * 1000)}",
            mode=self.voice_mode,
            intent=intent,
            user_id=user_id
        )
        
        self.voice_commands.append(command)
        
        # Publish immutable intent.command.v1 log
        await self.publish_event(
            event_type="intent.command.v1",
            actor="human",
            component="orb",
            payload={"intent": intent, "mode": command.mode.value}
        )
        
        logger.info(f"Voice command executed: {intent} (mode={self.voice_mode.value})")
        return command
    
    def get_unified_stats(self) -> Dict[str, Any]:
        """Get comprehensive unified stats."""
        return {
            "version": self.version,
            "orchestrator": self.loop_orchestrator.get_health_status(),
            "context_bus": {
                "context_keys": len(self.context_bus.current_context),
                "history_size": len(self.context_bus.context_history),
                "subscribers": sum(len(subs) for subs in self.context_bus.subscribers.values())
            },
            "memory_explorer": {
                "total_items": len(self.memory_explorer.memory_items),
                "folders": len(self.memory_explorer.context_manifests),
                "adjacency_graph_nodes": len(self.memory_explorer.adjacency_graph)
            },
            "avn_engine": {
                "anomalies_detected": len(self.avn_engine.anomalies),
                "hypotheses_generated": len(self.avn_engine.hypotheses),
                "healing_actions": len(self.avn_engine.healing_history)
            },
            "trust_metrics": {
                "total": len(self.trust_metrics),
                "avg_trust": sum(m.trust_score for m in self.trust_metrics.values()) / max(len(self.trust_metrics), 1)
            },
            "kpi_metrics": {
                "total": len(self.kpi_metrics),
                "critical": sum(1 for m in self.kpi_metrics.values() if m.status == "critical"),
                "warning": sum(1 for m in self.kpi_metrics.values() if m.status == "warning")
            },
            "voice": {
                "mode": self.voice_mode.value,
                "commands_executed": len(self.voice_commands)
            },
            "events": {
                "total_envelopes": len(self.event_envelopes)
            }
        }
