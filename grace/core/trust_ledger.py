# -*- coding: utf-8 -*-
import json, os, time, threading
from typing import Dict, List, Optional, Tuple

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    x = float(x)
    if x < lo: return lo
    if x > hi: return hi
    return x

class TrustLedger:
    """
    JSONL-backed, append-only trust ledger with an in-memory index.

    File path example: grace_data/trust_ledger.jsonl

    Exposed API (compat with your tests/workflows):
      - get(source, default=None) -> float
      - adjust_score(source, delta=0.0, context=None, event_id=None, min_score=0.0, max_score=1.0) -> float
      - update_score(...) -> alias of adjust_score
      - get_stats() -> dict(total_entities, total_interactions, mean_score)
      - get_trusted_entities(min_trust=0.7) -> List[Tuple[source, score]]
      - get_quarantined_entities(max_trust=0.3) -> List[Tuple[source, score]]
    """
    def __init__(self, path: str, default_score: float = 0.5):
        self.path = path
        self.default = float(default_score)
        self.scores: Dict[str, float] = {}
        self.interactions: int = 0
        self.seq: int = 0
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Ensure file exists
        if not os.path.exists(self.path):
            open(self.path, "a", encoding="utf-8").close()
        self._load()

    # -------- persistence --------
    def _load(self):
        with self._lock:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        src = rec.get("source")
                        after = rec.get("score_after")
                        if src is not None and after is not None:
                            self.scores[src] = float(after)
                        self.seq = max(self.seq, int(rec.get("seq", 0)))
                        self.interactions = max(self.interactions, int(rec.get("interactions", 0)))
            except FileNotFoundError:
                pass

    def _append(self, record: dict):
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -------- read API --------
    def get(self, source: str, default: Optional[float] = None) -> float:
        if default is None:
            default = self.default
        with self._lock:
            return float(self.scores.get(source, default))

    def get_stats(self) -> dict:
        with self._lock:
            n = len(self.scores)
            mean = sum(self.scores.values()) / n if n else 0.0
            return {
                "total_entities": n,
                "total_interactions": int(self.interactions),
                "mean_score": mean,
            }

    def get_trusted_entities(self, min_trust: float = 0.7) -> list:
        with self._lock:
            return sorted(
                [(k, v) for k, v in self.scores.items() if v >= float(min_trust)],
                key=lambda kv: kv[1],
                reverse=True,
            )

    def get_quarantined_entities(self, max_trust: float = 0.3) -> list:
        with self._lock:
            return sorted(
                [(k, v) for k, v in self.scores.items() if v <= float(max_trust)],
                key=lambda kv: kv[1],
            )

    # -------- write API --------
    def adjust_score(
        self,
        source: str,
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
            before = float(self.scores.get(source, self.default))
            after = _clamp(before + float(delta), float(min_score), float(max_score))
            self.scores[source] = after
            self.seq += 1
            self.interactions += 1
            rec = {
                "ts": now,
                "seq": int(self.seq),
                "interactions": int(self.interactions),
                "source": source,
                "score_before": before,
                "score_delta": float(delta),
                "score_after": after,
                "context": context,
                "event_id": event_id,
            }
            self._append(rec)
            return after

    # Back-compat alias used in earlier logs
    def update_score(self, *args, **kwargs) -> float:
        return self.adjust_score(*args, **kwargs)
