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
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TrustLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        # entity_id -> {score: float, interactions: int, level: str, meta: dict}
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._seq: int = 0
        # rollups (kept in-memory; rebuilt on load)
        self._total_interactions: int = 0
        self._by_type = defaultdict(int)   # e.g. {"ingestion": 10, "verification": 3}
        self._by_level = defaultdict(int)  # e.g. {"low": 2, "medium": 5, "high": 1}
        self._load()

    def _load(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
            logger.info("Loaded 0 trust records, tracking 0 entities.")
            return

        # naive line-by-line load (jsonl)
        cnt = 0
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        cnt += 1
                        # minimal rebuild: entity + last score/level + counters
                        ent = rec.get("entity_id") or rec.get("entity") or "unknown"
                        score = rec.get("score_after", 0.0)
                        level = rec.get("level", rec.get("trust_level", "unknown"))
                        etype = rec.get("type", rec.get("event_type", "unknown"))
                        ecount = self._entities.get(ent, {}).get("interactions", 0) + 1
                        self._entities[ent] = {
                            "score": score,
                            "interactions": ecount,
                            "level": level,
                            "meta": rec.get("meta", {}),
                        }
                        self._total_interactions += 1
                        self._by_type[etype] += 1
                        self._by_level[level] += 1
                        self._seq = max(self._seq, int(rec.get("seq", 0)))
                    except Exception:
                        # keep loading even if a line is borked
                        continue
            logger.info(f"Trust Ledger initialized with {len(self._entities)} entities from {self.path}")
        except Exception as e:
            logger.exception("Failed loading trust ledger: %s", e)
            logger.info("Trust Ledger initialized with 0 entities from %s", self.path)

    def record(self, entity_id: str, delta: float, *, level: str = "unknown", etype: str = "generic", meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Atomically append a JSONL record and update in-memory aggregates.
        """
        meta = meta or {}
        ent = self._entities.get(entity_id, {"score": 0.0, "interactions": 0, "level": "unknown", "meta": {}})
        score_before = float(ent["score"])
        score_after = score_before + float(delta)
        ent["score"] = score_after
        ent["interactions"] = int(ent["interactions"]) + 1
        ent["level"] = level
        ent["meta"] = {**ent.get("meta", {}), **meta}
        self._entities[entity_id] = ent
        self._seq += 1
        # rollups
        self._total_interactions += 1
        self._by_type[etype] += 1
        self._by_level[level] += 1

        rec = {
            "seq": self._seq,
            "entity_id": entity_id,
            "delta": float(delta),
            "score_before": score_before,
            "score_after": score_after,
            "level": level,
            "type": etype,
            "meta": meta,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def get(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self._entities.get(entity_id)

    def stats(self) -> Dict[str, Any]:
        """
        Return a stable schema expected by tests/consumers.
        Keys are always present.
        """
        # compute top_entities (sorted by score desc)
        top = sorted(
            ({"entity_id": k, "score": v["score"], "interactions": v["interactions"], "level": v.get("level", "unknown")}
             for k, v in self._entities.items()),
            key=lambda x: (x["score"], x["interactions"]),
            reverse=True,
        )[:10]
        return {
            "total_entities": len(self._entities),
            "total_interactions": int(self._total_interactions),
            "by_type": dict(self._by_type),     # may be {}
            "by_level": dict(self._by_level),   # may be {}
            "top_entities": top,                # [] when none
            "last_seq": int(self._seq),
        }
