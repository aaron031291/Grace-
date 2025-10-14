"""
Lightweight FusionDB compatibility shim for MCP development.

This module provides a small async-friendly wrapper around SQLite3 to satisfy
MCP and PushbackHandler expectations (insert, execute, query_one, query_many,
query_scalar, fetch_many). It's intentionally minimal — for production use the
real DB client should be wired in.

This shim opens `grace_system.db` in the workspace root by default.
"""
import sqlite3
import asyncio
import json
import time
from typing import Any, Dict, List, Optional

DB_PATH = "grace_system.db"


class FusionDB:
    _instance = None

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # Use check_same_thread=False because we call from threads
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_placeholder_tables()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_placeholder_tables(self):
        # Ensure tables used by MCP exist (idempotent)
        # Note: audit_logs already exists in grace_system.db with full schema
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                observation_id TEXT PRIMARY KEY,
                observation_type TEXT,
                source_module TEXT,
                observation_data TEXT,
                context TEXT,
                credibility_score REAL,
                novelty_score REAL,
                observed_at REAL,
                processed INTEGER DEFAULT 0
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_id TEXT,
                intended_outcome TEXT,
                actual_outcome TEXT,
                success INTEGER,
                performance_metrics TEXT,
                side_effects_identified TEXT,
                error_analysis TEXT,
                lessons_learned TEXT,
                confidence_adjustment REAL,
                evaluated_at REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trust_scores (
                component TEXT PRIMARY KEY,
                trust_score REAL,
                previous_score REAL,
                change_reason TEXT,
                updated_at REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS outcome_patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                action_type TEXT,
                conditions TEXT,
                outcome TEXT,
                frequency INTEGER,
                confidence REAL,
                actionable_insight TEXT,
                first_occurrence REAL,
                last_occurrence REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta_loop_escalations (
                escalation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                loop_type TEXT,
                trigger_observation_id TEXT,
                escalation_reason TEXT,
                escalation_target TEXT,
                escalation_data TEXT,
                escalated_at REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS forensic_cases (
                case_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_type TEXT,
                error_code TEXT,
                domain TEXT,
                audit_id TEXT,
                observation_id TEXT,
                evaluation_id INTEGER,
                severity TEXT,
                description TEXT,
                evidence TEXT,
                status TEXT,
                created_at REAL
            )
            """
        )
        self.conn.commit()

    # --- Async wrappers using threads ---
    async def insert(self, table: str, payload: Dict[str, Any]) -> Any:
        return await asyncio.to_thread(self._insert_sync, table, payload)

    def _insert_sync(self, table: str, payload: Dict[str, Any]) -> Any:
        cur = self.conn.cursor()
        if table == 'audit_logs':
            # Use the actual audit_logs schema from grace_system.db
            # Schema: entry_id, category, data_json, transparency_level, timestamp, hash, previous_hash, chain_hash, chain_position, verified
            entry_id = payload.get('entry_id') or f"audit_{int(time.time()*1000)}"
            cur.execute(
                """INSERT INTO audit_logs (entry_id, category, data_json, transparency_level, timestamp, hash, previous_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry_id,
                    payload.get('category', payload.get('action', 'mcp_action')),
                    json.dumps(payload.get('data', payload.get('payload', {}))),
                    payload.get('transparency_level', 'internal'),
                    payload.get('timestamp', time.time()),
                    payload.get('hash', ''),
                    payload.get('prev_hash', payload.get('previous_hash', ''))
                )
            )
            self.conn.commit()
            return entry_id
        elif table == 'observations':
            # generate observation_id if missing
            obs_id = payload.get('observation_id') or f"obs_{int(time.time()*1000)}"
            cur.execute(
                "INSERT OR REPLACE INTO observations (observation_id, observation_type, source_module, observation_data, context, credibility_score, novelty_score, observed_at, processed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    obs_id,
                    payload.get('observation_type'),
                    payload.get('source_module'),
                    json.dumps(payload.get('observation_data', {})),
                    json.dumps(payload.get('context', {})),
                    payload.get('credibility_score', 0.5),
                    payload.get('novelty_score', 0.0),
                    payload.get('observed_at', time.time()),
                    1 if payload.get('processed') else 0
                )
            )
            self.conn.commit()
            return obs_id
        elif table == 'evaluations':
            # Handle evaluations table schema
            cur.execute(
                """INSERT INTO evaluations 
                   (evaluation_id, action_id, intended_outcome, actual_outcome, success, 
                    performance_metrics, side_effects_identified, error_analysis, 
                    lessons_learned, confidence_adjustment, evaluated_at, sent_to_reflection)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload.get('evaluation_id', f"eval_{int(time.time()*1000)}"),
                    payload.get('action_id', ''),
                    payload.get('intended_outcome', ''),
                    payload.get('actual_outcome', ''),
                    payload.get('success', 0),
                    payload.get('performance_metrics', '{}'),
                    payload.get('side_effects_identified', ''),
                    payload.get('error_analysis', ''),
                    payload.get('lessons_learned', ''),
                    payload.get('confidence_adjustment', 0.0),
                    payload.get('evaluated_at', time.time()),
                    payload.get('sent_to_reflection', False)
                )
            )
            self.conn.commit()
            return cur.lastrowid
        elif table == 'outcome_patterns':
            # Handle outcome_patterns table schema
            pattern_id = payload.get('pattern_id', f"pattern_{int(time.time()*1000)}")
            cur.execute(
                """INSERT INTO outcome_patterns 
                   (pattern_id, pattern_type, action_type, conditions, outcome, frequency, 
                    confidence, first_observed, last_observed, actionable_insight)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pattern_id,
                    payload.get('pattern_type', ''),
                    payload.get('action_type', ''),
                    json.dumps(payload.get('conditions', {})),  # JSON serialize
                    json.dumps(payload.get('outcome', {})),  # JSON serialize
                    payload.get('frequency', 1),
                    payload.get('confidence', 0.0),
                    payload.get('first_observed', time.time()),
                    payload.get('last_observed', time.time()),
                    json.dumps(payload.get('actionable_insight', ''))  # JSON serialize (might be list)
                )
            )
            self.conn.commit()
            return pattern_id
        elif table == 'meta_loop_escalations':
            # Handle meta_loop_escalations table schema
            cur.execute(
                """INSERT INTO meta_loop_escalations 
                   (loop_type, trigger_observation_id, escalation_reason, escalation_target, 
                    escalation_data, escalated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    payload.get('loop_type', ''),
                    payload.get('trigger_observation_id', ''),
                    payload.get('escalation_reason', ''),
                    payload.get('escalation_target', ''),
                    json.dumps(payload.get('escalation_data', {})),
                    payload.get('escalated_at', time.time())
                )
            )
            self.conn.commit()
            return cur.lastrowid
        else:
            # Generic insert: store payload as JSON into a text column 'payload' if table has it
            try:
                payload_text = json.dumps(payload)
                cur.execute(f"INSERT INTO {table} (payload) VALUES (?)", (payload_text,))
                self.conn.commit()
                return cur.lastrowid
            except Exception:
                # Fallback: try inserting fields if columns exist
                raise

    async def execute(self, sql: str, params: Optional[tuple] = None) -> None:
        return await asyncio.to_thread(self._execute_sync, sql, params)

    def _execute_sync(self, sql: str, params: Optional[tuple]) -> None:
        cur = self.conn.cursor()
        cur.execute(sql, params or ())
        self.conn.commit()

    async def query_one(self, sql: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self._query_one_sync, sql, params)

    def _query_one_sync(self, sql: str, params: Optional[tuple]) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(sql, params or ())
        row = cur.fetchone()
        return dict(row) if row else None

    async def query_many(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._query_many_sync, sql, params)

    def _query_many_sync(self, sql: str, params: Optional[tuple]) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(sql, params or ())
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    async def query_scalar(self, sql: str, params: Optional[tuple] = None) -> Any:
        return await asyncio.to_thread(self._query_scalar_sync, sql, params)

    def _query_scalar_sync(self, sql: str, params: Optional[tuple]) -> Any:
        cur = self.conn.cursor()
        cur.execute(sql, params or ())
        row = cur.fetchone()
        if row:
            # return first column
            return list(row)[0]
        return None

    async def fetch_many(self, table: str, ids: List[str]) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._fetch_many_sync, table, ids)

    def _fetch_many_sync(self, table: str, ids: List[str]) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        placeholders = ','.join('?' for _ in ids)
        cur.execute(f"SELECT * FROM {table} WHERE observation_id IN ({placeholders})", ids)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
