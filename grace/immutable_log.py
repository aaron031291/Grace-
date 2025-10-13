"""Immutable log service: append, fetch, and hybrid search.

Conservative, production-minded implementation but lightweight for local dev.
Uses SQLite for local dev, Postgres migration provided separately.
Vector search is stubbed and can be wired to Chroma/Qdrant later.
"""


import os
import json
import uuid
import logging
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from fastapi import APIRouter, HTTPException

from .utils.time import iso_now_utc, now_utc

logger = logging.getLogger(__name__)

router = APIRouter()

# DB adapter: use Postgres if DATABASE_URL is set and psycopg2 is available; otherwise use SQLite
DATABASE_URL = os.environ.get("DATABASE_URL")

_use_postgres = False
_pg = None
_sqlite_path = ":memory:"

if DATABASE_URL:
    try:
        import psycopg2
        import psycopg2.extras

        _pg = psycopg2
        _pg_extras = psycopg2.extras
        _use_postgres = True
    except Exception:
        logger.warning("psycopg2 not available, falling back to SQLite for immutable_log")


def _get_sqlite_conn():
    conn = sqlite3.connect(_sqlite_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _get_pg_conn():
    # psycopg2 connection; caller must manage commit/close
    conn = _pg.connect(DATABASE_URL)
    return conn


def _get_conn() -> Tuple[str, Any]:
    """Return tuple (db_type, conn). db_type is 'pg' or 'sqlite'."""
    if _use_postgres:
        return ("pg", _get_pg_conn())
    return ("sqlite", _get_sqlite_conn())


def _init_db():
    db_type, conn = _get_conn()
    if db_type == "pg":
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS immutable_entries (
                entry_id UUID PRIMARY KEY,
                entry_cid TEXT UNIQUE,
                timestamp TIMESTAMP WITH TIME ZONE,
                who_actor_id TEXT,
                who_actor_type TEXT,
                who_actor_display TEXT,
                what TEXT,
                where_host TEXT,
                where_region TEXT,
                where_service_path TEXT,
                when_ts TIMESTAMP WITH TIME ZONE,
                why TEXT,
                how TEXT,
                error_code TEXT,
                severity TEXT,
                tags TEXT[],
                text_summary TEXT,
                related_cids TEXT[],
                signature TEXT,
                payload_json JSONB
            )
            """
        )
        conn.commit()
        cur.close()
        conn.close()
    else:
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

    # Accept either a mapping (sqlite3.Row or dict) or dict-like
    def _get(k):
        try:
            return row[k]
        except Exception:
            return row.get(k) if isinstance(row, dict) else None

    return {
        "entry_id": row["entry_id"],
        "entry_cid": row["entry_cid"],
        "timestamp": _get("timestamp"),
        "who": {
            "actor_id": _get("who_actor_id"),
            "actor_type": _get("who_actor_type"),
            "actor_display": _get("who_actor_display"),
        },
        "what": _get("what"),
        "where": {"host": _get("where_host"), "region": _get("where_region"), "service_path": _get("where_service_path")},
        "when": _get("when_ts"),
        "why": _get("why"),
        "how": _get("how"),
        "error_code": _get("error_code"),
        "severity": _get("severity"),
        "tags": json.loads(_get("tags")) if _get("tags") else [],
        "text_summary": _get("text_summary"),
        "related_cids": json.loads(_get("related_cids")) if _get("related_cids") else [],
        "signature": _get("signature"),
        "payload": payload,
    }


@router.post("/api/v1/logs")
async def append_entry(entry: Dict[str, Any]):
    """Append an immutable log entry. Returns entry_id and entry_cid."""
    try:
        entry_id = str(uuid.uuid4())
        entry_cid = entry.get("entry_cid") or f"cid:sha256:{uuid.uuid4().hex}"
        timestamp = entry.get("timestamp") or iso_now_utc()

        db_type, conn = _get_conn()

        if db_type == "pg":
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO immutable_entries (
                    entry_id, entry_cid, timestamp, who_actor_id, who_actor_type, who_actor_display,
                    what, where_host, where_region, where_service_path, when_ts, why, how, error_code, severity,
                    tags, text_summary, related_cids, signature, payload_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    entry.get("tags", []),
                    entry.get("text_summary"),
                    entry.get("related_cids", []),
                    entry.get("signature"),
                    json.dumps(entry.get("payload", {})),
                ),
            )
            conn.commit()
            cur.close()
            conn.close()
        else:
            cur = conn.cursor()
            cur.execute(
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
    db_type, conn = _get_conn()
    if db_type == "pg":
        cur = conn.cursor()
        cur.execute("SELECT * FROM immutable_entries WHERE entry_cid = %s", (entry_cid,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        # psycopg2 returns tuples by default; map using column names if available
        if row and hasattr(row, "_asdict"):
            row = row._asdict()
    else:
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

    db_type, conn = _get_conn()
    if db_type == "pg":
        cur = conn.cursor()
        # translate ? params into %s for Postgres is already handled because we built SQL with %s earlier only in append; here params are compatible
        cur.execute(sql.replace("?", "%s"), tuple(params))
        rows = cur.fetchall()
        cur.close()
        conn.close()
    else:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

    results = [_row_to_entry(r) for r in rows]

    # If semantic mode / hybrid, we could rerank here using vector DB distances; currently stubbed
    return {"results": results, "count": len(results)}
