# -*- coding: utf-8 -*-
import json, os, time, threading
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class TrustLedger:
    """
    A persistent, append-only trust ledger that provides a stable API for tests.
    """
    def __init__(self, data_path: Path):
        self._path = Path(data_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self._entities: Dict[str, Dict[str, Any]] = {}
        self._interactions_total: int = 0
        self._by_type: Dict[str, int] = defaultdict(int)
        self._by_level: Dict[str, int] = defaultdict(int)
        
        self._load()

        logger.info("Trust Ledger initialized with %d entities from %s",
                    len(self._entities), str(self._path))

    def record_interaction(self, entity_id: str, itype: str, level: str = "info", delta: float = 0.0, meta: Optional[Dict] = None):
        """Records an interaction and optionally adjusts trust score."""
        with self._lock:
            ent = self._ensure_entity(entity_id)
            if delta != 0.0:
                ent["score"] = max(min(ent.get("score", 0.0) + delta, 1.0), -1.0)
            
            self._interactions_total += 1
            self._by_type[itype] += 1
            self._by_level[level] += 1

            rec = {
                "ts": time.time(),
                "type": "interaction",
                "entity_id": entity_id,
                "level": level,
                "itype": itype,
                "delta": delta,
                "meta": meta or {},
            }
            self._append(rec)
            return ent.get("score", 0.0)

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics in the shape expected by E2E tests."""
        with self._lock:
            return {
                "total_entities": len(self._entities),
                "total_interactions": self._interactions_total,
                "by_type": dict(self._by_type),
                "by_level": dict(self._by_level),
            }

    def get_quarantined_entities(self, max_trust: float = 0.1) -> List[str]:
        """Returns entities with a trust score at or below the threshold."""
        with self._lock:
            return [
                eid for eid, e in self._entities.items() 
                if e.get("score", 0.0) <= max_trust
            ]

    def adjust_score(
        self,
        entity_id: str,
        delta: float = 0.0,
        *,
        context: Optional[str] = None,
        event_id: Optional[str] = None,
        min_score: float = 0.0,
        max_score: float = 1.0,
    ) -> float:
        """
        Adjusts the trust score for `source` by `delta` and persists an entry.

        Returns the score AFTER the adjustment.
        """
        now = time.time()
        with self._lock:
            before = float(self.scores.get(entity_id, self.default))
            after = _clamp(before + float(delta), float(min_score), float(max_score))
            self.scores[entity_id] = after
            self.seq += 1
            self.interactions += 1
            rec = {
                "ts": now,
                "seq": int(self.seq),
                "interactions": int(self.interactions),
                "source": entity_id,
                "score_before": before,
                "score_delta": float(delta),
                "score_after": after,
                "context": context,
                "event_id": event_id,
            }
            self._append(rec)
            return after

    def get(self, entity_id: str, default: Optional[float] = None) -> float:
        if default is None:
            default = self.default
        with self._lock:
            return self._entities.get(entity_id)

    def get_trusted_entities(
        self, min_trust: float = 0.7, limit: Optional[int] = None, include_scores: bool = False
    ) -> list:
        with self._lock:
            items = [(eid, e.get("score", 0.0)) for eid, e in self._entities.items()]
            items.sort(key=lambda x: x[1], reverse=True)
            filtered = [item for item in items if item[1] >= min_trust]
            if limit is not None:
                filtered = filtered[: int(limit)]
            if include_scores:
                return filtered
            return [eid for eid, _ in filtered]

    # ---------- internals ----------

    def _ensure_entity(self, entity_id: str) -> Dict[str, Any]:
        if entity_id not in self._entities:
            self._entities[entity_id] = {"score": 0.0, "seq": 0, "meta": {}}
        return self._entities[entity_id]

    def _append(self, rec: Dict[str, Any]) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _load(self) -> None:
        if not self._path.exists():
            return

        loaded = 0
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    loaded += 1
                    entity_id = rec.get("entity_id")
                    if not entity_id:
                        continue

                    ent = self._ensure_entity(entity_id)
                    
                    if rec.get("type") == "score_adjust":
                        ent["score"] = rec.get("score_after", ent.get("score", 0.0))
                        ent["seq"] = rec.get("seq", ent.get("seq", 0))
                    
                    elif rec.get("type") == "interaction":
                        self._interactions_total += 1
                        self._by_level[rec.get("level", "info")] += 1
                        self._by_type[rec.get("itype", "generic")] += 1
                        if "delta" in rec and rec["delta"] != 0.0:
                             ent["score"] = max(min(ent.get("score", 0.0) + rec["delta"], 1.0), -1.0)

            logger.info("Loaded %d trust records, tracking %d entities.", loaded, len(self._entities))
        except Exception as e:
            logger.warning("Failed to load trust ledger from %s: %s", str(self._path), e)
