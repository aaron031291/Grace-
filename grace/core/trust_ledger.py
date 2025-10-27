"""
Grace AI - Trust Ledger
Dynamic trust scoring and tracking for entities, sources, and memory fragments

The Trust Ledger maintains a running trust score for all entities Grace interacts with:
- Data sources (APIs, sensors, users, external systems)
- Memory fragments (claims, facts, reasoning chains)
- LLM models and agents
- External services

Trust scores are adjusted based on:
- Verification outcomes from VWX
- Consistency with known truths
- Historical accuracy
- Temporal decay
- Contradiction detection
"""
import json
import logging
import os
from datetime import datetime, timezone
from threading import RLock

logger = logging.getLogger(__name__)

class TrustLedger:
    """
    Simple append-only JSONL trust ledger with in-memory rollups per entity.
    """
    def __init__(self, path: str):
        self.path = path
        self.entities: dict[str, dict] = {}
        self.total_interactions: int = 0
        self._lock = RLock()
        # ensure directory
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._load()
        logger.info("Trust Ledger initialized with %d entities from %s", len(self.entities), self.path)

    def _load(self):
        lines = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            pass
        
        interactions = 0
        for line in lines:
            try:
                rec = json.loads(line)
                interactions += 1
                ent = rec.get("entity")
                if not ent:
                    continue
                # roll-up
                slot = self.entities.setdefault(ent, {"score": 0.0, "interactions": 0, "last_update": None})
                slot["score"] = rec.get("score_after", slot["score"])
                slot["interactions"] = rec.get("seq", slot["interactions"])
                slot["last_update"] = rec.get("ts")
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in trust ledger: %s", line.strip())
                continue
        self.total_interactions = interactions
        logger.info("Loaded %d trust records, tracking %d entities.", interactions, len(self.entities))

    def _current_score(self, entity_id: str) -> float:
        return self.entities.get(entity_id, {}).get("score", 0.0)

    def update_score(
        self,
        entity_id: str,
        delta: float,
        reason: str,
        context: dict | None = None,
    ) -> dict:
        """
        Append an interaction to the JSONL ledger and update in-memory rollups.
        Returns the record written.
        """
        if not entity_id:
            raise ValueError("entity_id cannot be empty")

        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            prev_score = self._current_score(entity_id)
            new_score = round(prev_score + float(delta), 6)
            
            self.total_interactions += 1
            seq = self.entities.get(entity_id, {}).get("interactions", 0) + 1
            
            record = {
                "ts": ts,
                "entity": entity_id,
                "delta": delta,
                "score_before": prev_score,
                "score_after": new_score,
                "seq": seq,
                "reason": reason,
                "context": context or {},
            }
            
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except IOError as e:
                logger.error("Failed to write to trust ledger at %s: %s", self.path, e)
                # If write fails, we should not update the in-memory state
                self.total_interactions -= 1 # Decrement because the write failed
                raise
            
            # update rollup
            self.entities[entity_id] = {"score": new_score, "interactions": seq, "last_update": ts}
        return record

    def get_entity_stats(self, entity_id: str) -> dict | None:
        with self._lock:
            return self.entities.get(entity_id)

    def total_entities(self) -> int:
        with self._lock:
            return len(self.entities)
            
    def stats(self) -> dict:
        with self._lock:
            return {
                "entities_tracked": len(self.entities),
                "total_interactions": self.total_interactions,
            }
