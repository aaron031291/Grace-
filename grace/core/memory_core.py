"""
memory_core.py
MemoryCore — Persistent storage & retrieval for governance precedents, decisions, snapshots, and experiences.
"""

from __future__ import annotations

import json
import hashlib
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .contracts import UnifiedDecision, GovernanceSnapshot, Experience  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------
# Helpers
# ---------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _iso(dt: Optional[datetime] = None) -> str:
    d = dt or _utcnow()
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.isoformat()

def _dump_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def _to_dict(obj: Any) -> Dict[str, Any]:
    # Support dataclass (with .to_dict() optional) and Pydantic v2 models.
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # Pydantic v2
    if hasattr(obj, "to_dict"):
        return obj.to_dict()     # your dataclasses provided this
    if is_dataclass(obj):
        from dataclasses import asdict
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported object type for serialization: {type(obj)}")


# ---------------------------
# MemoryCore
# ---------------------------

class MemoryCore:
    """
    Central memory system for storing and retrieving governance decisions,
    precedents, snapshots, and learning experiences.
    """

    def __init__(self, db_path: str = "grace_governance.db") -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    # ---- Connection / Txn ----

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None)  # autocommit
        conn.row_factory = sqlite3.Row
        with conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    @contextmanager
    def _txn(self):
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ---- Schema init ----

    def _init_database(self) -> None:
        with self._txn() as conn:
            # Governance decisions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS governance_decisions (
                    decision_id     TEXT PRIMARY KEY,
                    subject         TEXT NOT NULL,
                    inputs_hash     TEXT NOT NULL,
                    recommendation  TEXT NOT NULL,
                    rationale       TEXT NOT NULL,
                    confidence      REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    trust_score     REAL NOT NULL CHECK (trust_score >= 0 AND trust_score <= 1),
                    outcome         TEXT,
                    instance_id     TEXT NOT NULL,
                    version         TEXT NOT NULL,
                    timestamp       TEXT NOT NULL, -- ISO8601 UTC
                    raw_data        TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_govdec_instance_ts ON governance_decisions(instance_id, timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_govdec_inputs_hash ON governance_decisions(inputs_hash)")

            # Snapshots
            conn.execute("""
                CREATE TABLE IF NOT EXISTS governance_snapshots (
                    snapshot_id         TEXT PRIMARY KEY,
                    instance_id         TEXT NOT NULL,
                    version             TEXT NOT NULL,
                    thresholds_json     TEXT NOT NULL,
                    policies_json       TEXT NOT NULL,
                    model_weights_json  TEXT NOT NULL,
                    state_hash          TEXT NOT NULL,
                    timestamp           TEXT NOT NULL -- ISO8601 UTC
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_instance_ts ON governance_snapshots(instance_id, timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_statehash ON governance_snapshots(state_hash)")

            # Shadow deltas
            conn.execute("""
                CREATE TABLE IF NOT EXISTS governance_deltas (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id     TEXT NOT NULL,
                    instance_a      TEXT NOT NULL,
                    instance_b      TEXT NOT NULL,
                    a_vs_b_diff     TEXT NOT NULL,
                    latency_diff    REAL NOT NULL,
                    compliance_diff REAL NOT NULL,
                    timestamp       TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_deltas_pair_ts ON governance_deltas(instance_a, instance_b, timestamp DESC)")

            # Experiences
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    type          TEXT NOT NULL,
                    component_id  TEXT NOT NULL,
                    context_json  TEXT NOT NULL,
                    outcome_json  TEXT NOT NULL,
                    success_score REAL NOT NULL CHECK (success_score >= 0 AND success_score <= 1),
                    timestamp     TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_type_ts ON experiences(type, timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_component_ts ON experiences(component_id, timestamp DESC)")

            # Precedents
            conn.execute("""
                CREATE TABLE IF NOT EXISTS precedents (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_hash           TEXT NOT NULL,
                    decision_id          TEXT NOT NULL,
                    similarity_keywords  TEXT NOT NULL,
                    outcome              TEXT NOT NULL,
                    confidence           REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    timestamp            TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_precedents_topic_ts ON precedents(topic_hash, timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_precedents_decision ON precedents(decision_id)")
        logger.info("MemoryCore: database schema initialized")

    # ---- Public API ----

    def store_decision(
        self,
        decision: UnifiedDecision,
        outcome: Optional[str] = None,
        *,
        instance_id: str = "default",
        version: str = "1.0.0",
    ) -> None:
        """Store a governance decision and create a precedent entry."""
        d = _to_dict(decision)
        inputs_hash = self._hash_dict(d.get("inputs", {}))
        ts = d.get("timestamp")
        ts_iso = _iso(ts if isinstance(ts, datetime) else None)

        raw_json = _dump_json(d)
        with self._txn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO governance_decisions
                (decision_id, subject, inputs_hash, recommendation, rationale, confidence, trust_score,
                 outcome, instance_id, version, timestamp, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["decision_id"],
                    d["topic"],
                    inputs_hash,
                    d["recommendation"],
                    d["rationale"],
                    float(d["confidence"]),
                    float(d["trust_score"]),
                    outcome,
                    instance_id,
                    version,
                    ts_iso,
                    raw_json,
                ),
            )

        # Precedent: use topic keywords + outcome/confidence
        self._create_precedent(d["topic"], d["decision_id"], d["recommendation"], float(d["confidence"]))
        logger.info("MemoryCore: stored decision %s", d["decision_id"])

    def get_decision_by_id(self, decision_id: str) -> Optional[Dict[str, Any]]:
        with self._txn() as conn:
            row = conn.execute(
                "SELECT * FROM governance_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
            return dict(row) if row else None

    def store_snapshot(self, snapshot: GovernanceSnapshot) -> None:
        s = _to_dict(snapshot)
        ts = s.get("created_at")
        ts_iso = _iso(ts if isinstance(ts, datetime) else None)

        with self._txn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO governance_snapshots
                (snapshot_id, instance_id, version, thresholds_json, policies_json,
                 model_weights_json, state_hash, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    s["snapshot_id"],
                    s["instance_id"],
                    s["version"],
                    _dump_json(s["thresholds"]),
                    _dump_json(s["policies"]),
                    _dump_json(s["model_weights"]),
                    s["state_hash"],
                    ts_iso,
                ),
            )
        logger.info("MemoryCore: stored snapshot %s", s["snapshot_id"])

    def get_latest_snapshot(self, instance_id: str) -> Optional[Dict[str, Any]]:
        with self._txn() as conn:
            row = conn.execute(
                """
                SELECT * FROM governance_snapshots
                WHERE instance_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (instance_id,),
            ).fetchone()
            return dict(row) if row else None

    def store_experience(self, experience: Experience) -> None:
        e = _to_dict(experience)
        ts = e.get("timestamp")
        ts_iso = _iso(ts if isinstance(ts, datetime) else None)

        with self._txn() as conn:
            conn.execute(
                """
                INSERT INTO experiences
                (type, component_id, context_json, outcome_json, success_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    e["type"],
                    e["component_id"],
                    _dump_json(e["context"]),
                    _dump_json(e["outcome"]),
                    float(e["success_score"]),
                    ts_iso,
                ),
            )
        logger.info("MemoryCore: stored experience type=%s component_id=%s", e["type"], e["component_id"])

    def get_similar_decisions(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve similar decisions for precedent-based reasoning (keyword match)."""
        topic_hash = hashlib.md5(topic.encode()).hexdigest()
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM precedents
                WHERE topic_hash = ?
                ORDER BY confidence DESC, timestamp DESC
                LIMIT ?
                """,
                (topic_hash, int(limit)),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_decision_history(self, instance_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent decision history for an instance."""
        cutoff = _utcnow() - timedelta(days=days)
        cutoff_iso = _iso(cutoff)
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM governance_decisions
                WHERE instance_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                """,
                (instance_id, cutoff_iso),
            ).fetchall()
            return [dict(r) for r in rows]

    def store_shadow_delta(
        self,
        decision_id: str,
        instance_a: str,
        instance_b: str,
        diff_data: Dict[str, Any],
        latency_diff: float,
        compliance_diff: float,
    ) -> None:
        with self._txn() as conn:
            conn.execute(
                """
                INSERT INTO governance_deltas
                (decision_id, instance_a, instance_b, a_vs_b_diff, latency_diff, compliance_diff, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    instance_a,
                    instance_b,
                    _dump_json(diff_data),
                    float(latency_diff),
                    float(compliance_diff),
                    _iso(),
                ),
            )

    def get_shadow_performance_metrics(self, instance_a: str, instance_b: str, limit: int = 1000) -> Dict[str, Any]:
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT latency_diff, compliance_diff
                FROM governance_deltas
                WHERE instance_a = ? AND instance_b = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (instance_a, instance_b, int(limit)),
            ).fetchall()

        if not rows:
            return {"average_latency_diff": 0.0, "average_compliance_diff": 0.0, "count": 0}

        lat = [float(r["latency_diff"]) for r in rows]
        comp = [float(r["compliance_diff"]) for r in rows]
        return {
            "average_latency_diff": sum(lat) / len(lat),
            "average_compliance_diff": sum(comp) / len(comp),
            "count": len(rows),
            "latest_latency_diff": lat[0],
            "latest_compliance_diff": comp[0],
        }

    # ---- Internals ----

    def _create_precedent(self, topic: str, decision_id: str, outcome: str, confidence: float) -> None:
        """Create a precedent entry for case-based reasoning."""
        topic_hash = hashlib.md5(topic.encode()).hexdigest()
        keywords = self._extract_keywords(topic)
        with self._txn() as conn:
            conn.execute(
                """
                INSERT INTO precedents
                (topic_hash, decision_id, similarity_keywords, outcome, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (topic_hash, decision_id, keywords, outcome, float(confidence), _iso()),
            )

    @staticmethod
    def _extract_keywords(text: str) -> str:
        """Very simple keyword extraction (stopword filter)."""
        import re
        words = re.findall(r"\b\w+\b", (text or "").lower())
        stop = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [w for w in words if w not in stop and len(w) > 2]
        return " ".join(keywords[:10])

    @staticmethod
    def _hash_dict(data: Dict[str, Any]) -> str:
        return hashlib.sha256(_dump_json(data).encode("utf-8")).hexdigest()

    # ---- Lifecycle ----

    def close(self) -> None:
        # Connections are scoped to context managers; nothing persistent to close.
        logger.info("MemoryCore: connections closed (SQLite autoclosed)")
