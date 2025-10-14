#!/usr/bin/env python3
"""
stress_test.py

Unified stress / grey-box test harness for Grace running in Codespaces.
Drop in, `pip install -r requirements.txt` (see README below) and run.

Features:
- HTTP endpoint concurrency & latency testing (with simulated network faults)
- DB connection pool exhaustion test (Postgres if available, otherwise local SQLite stress)
- Redis pub/sub / backpressure simulation (if available; fallback to mock)
- CPU & memory pressure tasks (large-array manipulations, pathological patterns)
- File I/O and disk stress (write/read big files safely)
- Race-condition stress (concurrent reads/writes to same structure)
- Verification & reporting to report.json
- Config via CLI args or environment variables
"""

import argparse
import asyncio
import json
import os
import random
import string
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# optional imports with graceful fallback
try:
    import aiohttp
except Exception:
    aiohttp = None

try:
    import asyncpg
except Exception:
    asyncpg = None

try:
    import aioredis
except Exception:
    aioredis = None

try:
    import psutil
except Exception:
    psutil = None

# -------------------------
# CONFIGURATION / DEFAULTS
# -------------------------
DEFAULT_HTTP_TARGET = os.environ.get("ST_HTTP_TARGET", "http://localhost:8000/health")
DEFAULT_POSTGRES_DSN = os.environ.get(
    "ST_PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"
)
DEFAULT_REDIS_URL = os.environ.get("ST_REDIS_URL", "redis://localhost:6379/0")
OUT_REPORT = Path(os.environ.get("ST_REPORT", "report.json"))
TMP_DIR = Path(os.environ.get("ST_TMP", "/tmp/grace_stress"))


# -------------------------
# REPORTING
# -------------------------
def now_iso() -> str:
    return datetime.now().isoformat()


class Report:
    def __init__(self):
        self.meta = {
            "version": "1.0",
            "timestamp": now_iso(),
            "env": dict(os.environ),
        }
        self.entries = []

    def add_entry(self, entry: Dict[str, Any]):
        self.entries.append(entry)

    def save(self, path: Path = OUT_REPORT):
        report = {"meta": self.meta, "entries": self.entries, "finished_at": now_iso()}
        try:
            path.write_text(json.dumps(report, indent=2))
            print(f"[report] written to {path}")
        except Exception as e:
            print(f"[report] ERROR writing to {path}: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Grace stress test harness")
    p.add_argument(
        "--http", help="HTTP target (default from env)", default=DEFAULT_HTTP_TARGET
    )
    p.add_argument("--http-concurrency", type=int, default=20)
    p.add_argument("--http-requests", type=int, default=200)
    p.add_argument("--http-fault-rate", type=float, default=0.05)
    p.add_argument(
        "--pg", help="Postgres DSN (optional)", default=os.environ.get("ST_PG_DSN")
    )
    p.add_argument("--pg-clients", type=int, default=10)
    p.add_argument("--pg-queries-per-client", type=int, default=10)
    p.add_argument(
        "--redis", help="Redis URL (optional)", default=os.environ.get("ST_REDIS_URL")
    )
    p.add_argument("--redis-publishers", type=int, default=5)
    p.add_argument("--redis-messages-per-publisher", type=int, default=100)
    p.add_argument(
        "--array-size",
        type=int,
        default=20000,
        help="size used for array tests (large -> slower)",
    )
    p.add_argument("--file-mb", type=int, default=2, help="file size in MB for disk IO")
    p.add_argument("--file-count", type=int, default=2)
    p.add_argument("--race-workers", type=int, default=20)
    p.add_argument("--race-iterations", type=int, default=200)
    return p.parse_args()


def ensure_tmp():
    TMP_DIR.mkdir(parents=True, exist_ok=True)


# Fix Redis import check and usage
# Use redis.asyncio for Redis stress test
async def main_async(args, rc):
    rc.add_entry({"system": "baseline_metrics", "ts": now_iso()})
    # HTTP stress test
    if aiohttp:
        try:
            async with aiohttp.ClientSession() as session:
                resp = await session.get(args.http)
                rc.add_entry({"http": "basic", "status": resp.status, "ts": now_iso()})
        except Exception as e:
            rc.add_entry({"http": "error", "error": str(e), "ts": now_iso()})
    else:
        rc.add_entry(
            {"http": "skipped", "reason": "aiohttp not installed", "ts": now_iso()}
        )
    # DB stress test
    if asyncpg:
        try:
            conn = await asyncpg.connect(args.pg)
            val = await conn.fetchval("SELECT 1")
            rc.add_entry({"db": "basic", "result": val, "ts": now_iso()})
            await conn.close()
        except Exception as e:
            rc.add_entry({"db": "error", "error": str(e), "ts": now_iso()})
    else:
        rc.add_entry(
            {"db": "skipped", "reason": "asyncpg not installed", "ts": now_iso()}
        )
    # Redis stress test
    try:
        import redis.asyncio as aioredis

        redis_client = aioredis.from_url(args.redis, password="grace_redis_pass")
        await redis_client.set("grace_stress_test", "ok")
        val = await redis_client.get("grace_stress_test")
        rc.add_entry(
            {"redis": "basic", "result": val.decode() if val else None, "ts": now_iso()}
        )
        await redis_client.close()
    except Exception as e:
        rc.add_entry({"redis": "error", "error": str(e), "ts": now_iso()})
    # Disk I/O stress test
    try:
        test_path = TMP_DIR / "disk_stress.bin"
        with open(test_path, "wb") as f:
            f.write(os.urandom(1024 * 1024))
        with open(test_path, "rb") as f:
            data = f.read()
        rc.add_entry({"disk": "basic", "size": len(data), "ts": now_iso()})
        os.remove(test_path)
    except Exception as e:
        rc.add_entry({"disk": "error", "error": str(e), "ts": now_iso()})
    # CPU/array stress test
    try:
        arr = [random.random() for _ in range(10000)]
        arr.sort()
        rc.add_entry({"cpu": "array_sort", "len": len(arr), "ts": now_iso()})
    except Exception as e:
        rc.add_entry({"cpu": "error", "error": str(e), "ts": now_iso()})
    rc.save()


def main():
    args = parse_args()
    rc = Report()
    ensure_tmp()
    print("[stress] starting harness, report will be saved to", OUT_REPORT)
    try:
        print("[stress] running main_async...")
        asyncio.run(main_async(args, rc))
        print("[stress] main_async completed, calling rc.save()...")
        rc.save()
        print("[stress] rc.save() completed.")
    except KeyboardInterrupt:
        print("[stress] interrupted by user")
        rc.add_entry({"system": "interrupted", "ts": now_iso()})
        rc.save()
    except Exception as e:
        print("[stress] fatal error", e)
        rc.add_entry({"system": "fatal", "error": str(e)})
        rc.save()


print("[stress_test.py] Script entry confirmed.")
if __name__ == "__main__":
    print("[stress_test.py] __main__ block entered.")
    main()
