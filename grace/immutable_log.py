"""Immutable log service: append, fetch, and hybrid search.

Conservative, production-minded implementation but lightweight for local dev.
Uses SQLite for local dev, Postgres migration provided separately.
Vector search is stubbed and can be wired to Chroma/Qdrant later.
"""

import json
import sqlite3
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException

from .utils.time import iso_now_utc, now_utc

logger = logging.getLogger(__name__)

router = APIRouter()

# Simple local SQLite DB path (for dev); in production use Postgres connection
DB_PATH = ":memory:"


def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS immutable_entries (
            entry_id TEXT PRIMARY KEY,
            entry_cid TEXT UNIQUE,
            timestamp TEXT,
            who_actor_id TEXT,
            who_actor_type TEXT,
            who_actor_display TEXT,
            what TEXT,
            where_host TEXT,
            where_region TEXT,
            where_service_path TEXT,
            when_ts TEXT,
            why TEXT,
            how TEXT,
            error_code TEXT,
            severity TEXT,
            tags TEXT,
            text_summary TEXT,
            related_cids TEXT,
            signature TEXT,
            payload_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


_init_db()


def _row_to_entry(row: sqlite3.Row) -> Dict[str, Any]:
    payload = None
    try:
        payload = json.loads(row["payload_json"]) if row["payload_json"] else None
    except Exception:
        payload = row["payload_json"]

    return {
        "entry_id": row["entry_id"],
        "entry_cid": row["entry_cid"],
        "timestamp": row["timestamp"],
        "who": {
            "actor_id": row["who_actor_id"],
            "actor_type": row["who_actor_type"],
            "actor_display": row["who_actor_display"],
        },
        "what": row["what"],
        "where": {"host": row["where_host"], "region": row["where_region"], "service_path": row["where_service_path"]},
        "when": row["when_ts"],
        "why": row["why"],
        "how": row["how"],
        "error_code": row["error_code"],
        "severity": row["severity"],
        "tags": json.loads(row["tags"]) if row["tags"] else [],
        "text_summary": row["text_summary"],
        "related_cids": json.loads(row["related_cids"]) if row["related_cids"] else [],
        "signature": row["signature"],
        "payload": payload,
    }


@router.post("/api/v1/logs")
async def append_entry(entry: Dict[str, Any]):
    """Append an immutable log entry. Returns entry_id and entry_cid."""
    try:
        entry_id = str(uuid.uuid4())
        entry_cid = entry.get("entry_cid") or f"cid:sha256:{uuid.uuid4().hex}"
        timestamp = entry.get("timestamp") or iso_now_utc()

        conn = _get_conn()
        conn.execute(
            """
            INSERT INTO immutable_entries (
                entry_id, entry_cid, timestamp, who_actor_id, who_actor_type, who_actor_display,
                what, where_host, where_region, where_service_path, when_ts, why, how, error_code, severity,
                tags, text_summary, related_cids, signature, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                entry_cid,
                timestamp,
                entry.get("who", {}).get("actor_id"),
                entry.get("who", {}).get("actor_type"),
                entry.get("who", {}).get("actor_display"),
                entry.get("what"),
                entry.get("where", {}).get("host"),
                entry.get("where", {}).get("region"),
                entry.get("where", {}).get("service_path"),
                entry.get("when"),
                entry.get("why"),
                entry.get("how"),
                entry.get("error_code"),
                entry.get("severity"),
                json.dumps(entry.get("tags", [])),
                entry.get("text_summary"),
                json.dumps(entry.get("related_cids", [])),
                entry.get("signature"),
                json.dumps(entry.get("payload", {})),
            ),
        )
        conn.commit()
        conn.close()

        # TODO: compute embeddings and push to vector DB

        return {"entry_id": entry_id, "entry_cid": entry_cid}

    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception("Failed to append entry")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/logs/{entry_cid}")
async def get_entry(entry_cid: str):
    conn = _get_conn()
    cursor = conn.execute("SELECT * FROM immutable_entries WHERE entry_cid = ?", (entry_cid,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Entry not found")

    return _row_to_entry(row)


@router.post("/api/v1/logs/search")
async def search_logs(body: Dict[str, Any]):
    """Hybrid search: conservative implementation.

    Body fields: query, filters (from/to/tags/error_code), mode (exact|semantic|hybrid), limit
    """
    query = body.get("query", "")
    filters = body.get("filters", {})
    mode = body.get("mode", "hybrid")
    limit = int(body.get("limit", 20))

    # Conservative approach: use SQL full-text like LIKE on key fields, and optionally call vector DB stub
    sql = "SELECT * FROM immutable_entries WHERE 1=1"
    params: List[Any] = []

    if filters.get("from"):
        sql += " AND timestamp >= ?"
        params.append(filters.get("from"))
    if filters.get("to"):
        sql += " AND timestamp <= ?"
        params.append(filters.get("to"))
    if filters.get("error_code"):
        sql += " AND error_code = ?"
        params.append(filters.get("error_code"))
    if filters.get("tags"):
        # naive tags matching
        for tag in filters.get("tags"):
            sql += " AND tags LIKE ?"
            params.append(f"%{tag}%")

    if mode in ("exact", "hybrid") and query:
        sql += " AND (what LIKE ? OR why LIKE ? OR how LIKE ? OR text_summary LIKE ?)"
        qparam = f"%{query}%"
        params.extend([qparam, qparam, qparam, qparam])

    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    conn = _get_conn()
    cursor = conn.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    results = [_row_to_entry(r) for r in rows]

    # If semantic mode / hybrid, we could rerank here using vector DB distances; currently stubbed
    return {"results": results, "count": len(results)}
