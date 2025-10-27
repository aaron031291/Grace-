from __future__ import annotations
import json, time, threading
from pathlib import Path
from typing import Dict, Any, Optional, DefaultDict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class TrustLedger:
    """
    Persistent, append-only-ish trust ledger with:
      - adjust_score / update_score (alias)  -> returns (score_after, seq)
      - record_interaction                  -> tracks interaction counts by type/level
      - get_stats                           -> returns dict with by_type/by_level keys (tests expect these)
    Storage format: JSONL at <grace_data>/trust_ledger.jsonl
    """

    def __init__(self, data_path: Path):
        self._path = Path(data_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self._entities: Dict[str, Dict[str, Any]] = {}   # entity_id -> {score, seq, meta}
        self._by_type: DefaultDict[str, int]  = defaultdict(int)
        self._by_level: DefaultDict[str, int] = defaultdict(int)
        self._interactions_total = 0
        self._seq_global = 0

        self._load()

        logger.info("Trust Ledger initialized with %d entities from %s",
                    len(self._entities), str(self._path))

    # ---------- public API ----------

    def adjust_score(
        self,
        entity_id: str,
        delta: float,
        *,
        reason: str = "",
        event_id: str = "",
        source_type: str = "",
        context: dict | None = None,   # <— NEW: accept context explicitly
        **_ignore,                     # <— NEW: ignore any other legacy kwargs safely
    ) -> Dict[str, Any]:
        """Primary mutator. Returns dict with new score/seq."""
        with self._lock:
            ent = self._ensure_entity(entity_id)
            ent["score"] = float(ent.get("score", 0.0)) + float(delta)
            ent["score"] = max(min(ent["score"], 1.0), -1.0)  # clamp for sanity
            ent["seq"] = ent.get("seq", 0) + 1
            self._seq_global += 1

            rec = {
                "ts": time.time(),
                "type": "score_adjust",
                "entity_id": entity_id,
                "delta": delta,
                "score_after": ent["score"],
                "seq": ent["seq"],
                "reason": reason,
                "event_id": event_id,
                "source_type": source_type,
                "context": context or {},    # <— persist if provided
            }
            self._append(rec)
            return {"score_after": ent["score"], "seq": ent["seq"]}

    # Backward-compat alias (handlers/tests call update_score)
    def update_score(self, entity_id: str, delta: float, **kwargs) -> Dict[str, Any]:
        return self.adjust_score(entity_id, delta, **kwargs)

    def record_interaction(self, *, entity_id: str, level: str = "info", itype: str = "generic", meta: Optional[Dict[str, Any]] = None) -> None:
        """Counts interactions and indexes them by type/level for stats."""
        with self._lock:
            self._ensure_entity(entity_id)
            self._by_level[level] += 1
            self._by_type[itype]  += 1
            self._interactions_total += 1

            rec = {
                "ts": time.time(),
                "type": "interaction",
                "entity_id": entity_id,
                "level": level,
                "itype": itype,
                "meta": meta or {},
            }
            self._append(rec)

    def get_stats(self) -> Dict[str, Any]:
        """Tests expect keys: by_level, by_type. Always present, even if empty."""
        with self._lock:
            stats = {
                "entities": len(self._entities),
                "total_entities": len(self._entities),      # <— NEW for your test
                "interactions": self._interactions_total,
                "total_interactions": self._interactions_total,  # <— friendly alias
                "by_level": dict(self._by_level),
                "by_type": dict(self._by_type),
            }
            return stats

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._entities.get(entity_id)

    def get_trusted_entities(
        self,
        min_trust: float = 0.7,
        limit: int | None = None,
        include_scores: bool = False,
    ) -> list:
        """Return entities with score >= min_trust, sorted desc by score."""
        with self._lock:
            items = [(eid, e.get("score", 0.0)) for eid, e in self._entities.items()]
            items.sort(key=lambda x: x[1], reverse=True)
            filtered = [it for it in items if it[1] >= float(min_trust)]
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
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _load(self) -> None:
        if not self._path.exists():
            logger.info("Loaded 0 trust records, tracking 0 entities.")
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
                    if rec.get("type") == "score_adjust":
                        ent = self._ensure_entity(rec["entity_id"])
                        ent["score"] = rec.get("score_after", ent.get("score", 0.0))
                        ent["seq"] = rec.get("seq", ent.get("seq", 0))
                    elif rec.get("type") == "interaction":
                        self._by_level[rec.get("level", "info")] += 1
                        self._by_type[rec.get("itype", "generic")] += 1
                        self._interactions_total += 1
            logger.info("Loaded %d trust records, tracking %d entities.", loaded, len(self._entities))
        except Exception as e:
            logger.warning("Failed to load trust ledger from %s: %s", str(self._path), e)
