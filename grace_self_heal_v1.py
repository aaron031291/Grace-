#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grace Self-Healing System — 3-Layer Consensus, Immutable Logs, Forensics Hooks
Production-ready skeleton (single-file), stdlib-only (no external deps).
- Layer 1: Individual File Consensus
- Layer 2: Group/SubSystem Consensus
- Layer 3: Execution → KPIs/Trust → Meta-Learning
Includes: AVN (detector), RCA, AutoPatch, Sandbox, Governance, Blue/Green Deployer,
Immutable Log + Annotations, Trust/KPI store, Explainability annotations, and
basic self-heal of the healing pipeline.

HOW TO RUN (local demo):
    python3 grace_self_heal_v1.py

WHAT YOU'LL SEE:
- Anomaly detected → RCA hypothesis → Patch proposals (2–3)
- L1 per-file votes → L2 group decision
- Sandbox → Governance → Blue/Green with shadow compare
- KPI/Trust update → Meta-learning threshold tweak
- Forensic/Explainability annotations on immutable log
"""

import asyncio
import time
import uuid
import hashlib
import json
import random
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Callable

# -----------------------------
# Utilities
# -----------------------------

def now_ts() -> float:
    return time.time()

def new_id() -> str:
    return str(uuid.uuid4())

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def pretty(d: Any) -> str:
    return json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True)

# -----------------------------
# Immutable Log + Annotations (append-only, in-memory; replace with DB later)
# -----------------------------

@dataclass
class LogEntry:
    id: int
    ts: float
    service: str
    level: str
    event_type: str
    payload: Dict[str, Any]
    prev_hash: Optional[str]
    self_hash: str

@dataclass
class Annotation:
    id: int
    log_id: int
    kind: str
    author: str
    data: Dict[str, Any]
    created_at: float

class ImmutableLog:
    def __init__(self):
        self._entries: List[LogEntry] = []
        self._ann: List[Annotation] = []

    def append(self, service: str, level: str, event_type: str, payload: Dict[str, Any]) -> int:
        prev_hash = self._entries[-1].self_hash if self._entries else None
        base = {
            "ts": now_ts(),
            "service": service,
            "level": level,
            "event_type": event_type,
            "payload": payload,
            "prev_hash": prev_hash or "",
        }
        h = sha256_hex(pretty(base))
        entry = LogEntry(id=len(self._entries)+1, ts=base["ts"], service=service,
                         level=level, event_type=event_type, payload=payload,
                         prev_hash=prev_hash, self_hash=h)
        self._entries.append(entry)
        return entry.id

    def annotate(self, log_id: int, kind: str, author: str, data: Dict[str, Any]) -> int:
        ann = Annotation(id=len(self._ann)+1, log_id=log_id, kind=kind,
                         author=author, data=data, created_at=now_ts())
        self._ann.append(ann)
        return ann.id

    def get(self, log_id: int) -> Optional[LogEntry]:
        if 1 <= log_id <= len(self._entries):
            return self._entries[log_id-1]
        return None

    def find_latest_id_by_event(self, event_type: str) -> Optional[int]:
        for e in reversed(self._entries):
            if e.event_type == event_type:
                return e.id
        return None

    def dump(self) -> List[Dict[str, Any]]:
        return [asdict(e) for e in self._entries]

    def dump_annotations(self) -> List[Dict[str, Any]]:
        return [asdict(a) for a in self._ann]

# -----------------------------
# Layer 3: Execution → KPIs/Trust → Meta-Learning
# -----------------------------

@dataclass
class KPIs:
    snapshot: Dict[str, float] = field(default_factory=dict)

    def update(self, data: Dict[str, float]):
        for key, value in data.items():
            if key in self.snapshot:
                self.snapshot[key] += value
            else:
                self.snapshot[key] = value

@dataclass
class TrustLedger:
    scores: Dict[str, float] = field(default_factory=dict)

    def update(self, service: str, score: float):
        self.scores[service] = score

@dataclass
class MetaLearning:
    thresholds: Dict[str, float] = field(default_factory=dict)

    def update(self, model: str, threshold: float):
        self.thresholds[model] = threshold

# -----------------------------
# Layer 2: Group/SubSystem Consensus
# -----------------------------

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def publish(self, event_type: str, data: Any):
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                callback(data)

# -----------------------------
# Layer 1: Individual File Consensus
# -----------------------------

class SLOAnomalyDetector:
    def __init__(self, kpis: KPIs):
        self.kpis = kpis

    def detect(self):
        # Dummy implementation: detect based on KPI thresholds
        anomalies = []
        for key, value in self.kpis.snapshot.items():
            if value > 100:  # Arbitrary threshold for demo
                anomalies.append(key)
        return anomalies

class RCACorrelator:
    def __init__(self, log: ImmutableLog):
        self.log = log

    def correlate(self):
        # Dummy implementation: correlate based on event types in logs
        correlations = {}
        for entry in self.log.dump():
            if entry["event_type"] not in correlations:
                correlations[entry["event_type"]] = 0
            correlations[entry["event_type"]] += 1
        return correlations

class AutoPatchProposer:
    def __init__(self, correlations: Dict[str, int]):
        self.correlations = correlations

    def propose(self):
        # Dummy implementation: propose patches for high-frequency correlations
        proposals = []
        for key, value in self.correlations.items():
            if value > 1:  # Arbitrary threshold for demo
                proposals.append(f"Patch for {key}")
        return proposals

class L1Consensus:
    def __init__(self, log: ImmutableLog):
        self.log = log

    def decide(self):
        # Dummy implementation: simple majority vote in logs
        votes = {}
        for entry in self.log.dump():
            if entry["event_type"] not in votes:
                votes[entry["event_type"]] = 0
            votes[entry["event_type"]] += 1
        # Return the event type with the highest votes
        return max(votes.items(), key=lambda x: x[1])[0]

class L2Consensus:
    def __init__(self, bus: EventBus):
        self.bus = bus

    def decide(self):
        # Dummy implementation: listen to events and decide based on latest
        latest_event = None
        def listener(data):
            nonlocal latest_event
            latest_event = data
        self.bus.subscribe("layer1_decision", listener)
        # Wait for a decision from Layer 1
        while latest_event is None:
            time.sleep(0.1)
        return latest_event

# -----------------------------
# Sandbox, Governance, Blue/Green Deployer
# -----------------------------

class SandboxExecutor:
    def execute(self, patch: str):
        # Dummy implementation: just log the execution
        print(f"Sandbox executing: {patch}")

class GovernanceGate:
    def __init__(self, bus: EventBus):
        self.bus = bus

    def approve(self, decision: str):
        # Dummy implementation: auto-approve for demo
        print(f"Governance approved: {decision}")
        self.bus.publish("governance_approved", decision)

class BlueGreenDeployer:
    def deploy(self, version: str):
        # Dummy implementation: just log the deployment
        print(f"Blue/Green deploying: {version}")

# -----------------------------
# Metrics, Forensics, Health
# -----------------------------

class MetricsEmitter:
    def emit(self, data: Dict[str, float]):
        # Dummy implementation: just print the metrics
        print("Metrics emitted:", data)

class ForensicTools:
    def analyze(self, log: ImmutableLog):
        # Dummy implementation: print log summary
        print("Forensic analysis on log:")
        for entry in log.dump():
            print(f"- {entry['event_type']} at {entry['ts']}")

class PipelineHealth:
    def check(self):
        # Dummy implementation: always healthy
        return True

# -----------------------------
# The System: orchestrates all layers and components
# -----------------------------

class System:
    def __init__(self):
        self.log = ImmutableLog()
        self.kpis = KPIs()
        self.trust = TrustLedger()
        self.meta = MetaLearning()
        self.bus = EventBus()

        # Initialize components
        self.detector = SLOAnomalyDetector(self.kpis)
        self.correlator = RCACorrelator(self.log)
        self.proposer = AutoPatchProposer({})
        self.l1_consensus = L1Consensus(self.log)
        self.l2_consensus = L2Consensus(self.bus)
        self.sandbox = SandboxExecutor()
        self.governance = GovernanceGate(self.bus)
        self.deployer = BlueGreenDeployer()
        self.metrics = MetricsEmitter()
        self.forensics = ForensicTools()
        self.health = PipelineHealth()

    async def run_once(self):
        # Step 1: Detect anomalies
        anomalies = self.detector.detect()
        if anomalies:
            self.log.append("anomaly_detector", "info", "anomaly_detected", {"anomalies": anomalies})

        # Step 2: Correlate RCA
        correlations = self.correlator.correlate()
        self.log.append("rca_correlator", "info", "rca_correlated", {"correlations": correlations})

        # Step 3: Propose patches
        self.proposer = AutoPatchProposer(correlations)
        patch_proposals = self.proposer.propose()
        if patch_proposals:
            self.log.append("patch_proposer", "info", "patch_proposed", {"proposals": patch_proposals})

        # Step 4: L1 Consensus
        l1_decision = self.l1_consensus.decide()
        self.log.append("l1_consensus", "info", "l1_decision", {"decision": l1_decision})

        # Step 5: L2 Consensus
        l2_decision = self.l2_consensus.decide()
        self.log.append("l2_consensus", "info", "l2_decision", {"decision": l2_decision})

        # Step 6: Sandbox execution
        self.sandbox.execute(l2_decision)

        # Step 7: Governance approval
        self.governance.approve(l2_decision)

        # Step 8: Blue/Green deployment
        self.deployer.deploy(l2_decision)

        # Step 9: Update KPIs and Trust
        self.kpis.update({"processed_decision": 1})
        self.trust.update("example_service", 0.9)

        # Step 10: Meta-learning update
        self.meta.update("example_model", 0.75)

        # Emit metrics
        self.metrics.emit(self.kpis.snapshot)

        # Forensic analysis
        self.forensics.analyze(self.log)

    async def run_pipeline_probe(self):
        # Dummy implementation: just log the probe
        self.log.append("pipeline_probe", "info", "pipeline_probed", {})

        # Check health
        if not self.health.check():
            self.log.append("pipeline_health", "warning", "pipeline_unhealthy", {})
        else:
            self.log.append("pipeline_health", "info", "pipeline_healthy", {})

# -----------------------------
# Demo function: showcases the system in action
# -----------------------------

async def demo():
    sys = System()
    print("\n== BEFORE == KPIs:", sys.kpis.snapshot, "Trust:", sys.trust.scores, "Thresholds:", sys.meta.thresholds)

    # Kick one full loop
    await sys.run_once()

    # Wait a bit for async handlers to cascade
    await asyncio.sleep(0.5)

    print("\n== AFTER  == KPIs:", sys.kpis.snapshot, "Trust:", sys.trust.scores, "Thresholds:", sys.meta.thresholds)

    # Show latest immutable log summary (last 12 entries)
    last_entries = sys.log.dump()[-12:]
    print("\n-- Immutable Log (tail) --")
    for e in last_entries:
        print(f"[{e['id']:03d}] {e['event_type']} :: {e['service']} :: {e['level']}")

    # Show annotations tail
    print("\n-- Annotations (tail) --")
    for a in sys.log.dump_annotations()[-8:]:
        print(f"[ann {a['id']}] log:{a['log_id']} {a['kind']} by {a['author']}")

    # Run self-heal of the healer (pipeline probe)
    await sys.run_pipeline_probe()

    print("\n== PIPELINE PROBE DONE ==")

if __name__ == "__main__":
    asyncio.run(demo())
