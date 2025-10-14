import os
import subprocess
import time
import importlib
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pytest


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="Postgres DATABASE_URL not set")
def test_postgres_event_logging_integration():
    # Apply migrations to the provided DATABASE_URL
    cmd = ["python3", "scripts/apply_migrations.py"]
    env = os.environ.copy()
    # ensure the script uses DATABASE_URL from env
    subprocess.check_call(cmd, env=env)

    # Import and reload the immutable_log module after migrations
    mod = importlib.import_module("grace.immutable_log")
    importlib.reload(mod)

    app = FastAPI()
    app.include_router(mod.router)
    client = TestClient(app)

    # Create test users via ORM session
    from grace.db.session import get_session
    from grace.db.models_phase_a import UserAccount
    sess = get_session()
    try:
        reader = UserAccount(user_id="pg_test_reader", username="reader", display_name="Reader", role="auditor")
        writer = UserAccount(user_id="pg_test_writer", username="writer", display_name="Writer", role="logger")
        sess.add_all([reader, writer])
        sess.commit()
    finally:
        sess.close()

    # Append an entry with writer header
    entry = {"what": "pg integration test", "text_summary": "pg test"}
    r = client.post("/api/v1/logs", json=entry, headers={"X-Actor-Id": "pg_test_writer"})
    assert r.status_code == 200
    cid = r.json()["entry_cid"]

    # Read with reader header
    r2 = client.get(f"/api/v1/logs/{cid}", headers={"X-Actor-Id": "pg_test_reader"})
    assert r2.status_code == 200

    # Search
    r3 = client.post("/api/v1/logs/search", json={"query": "pg test"}, headers={"X-Actor-Id": "pg_test_reader"})
    assert r3.status_code == 200

    # Give DB a moment for commits
    time.sleep(0.5)

    # Verify event_logs contains at least one immutable_* event
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM event_logs WHERE event_type LIKE 'immutable_%'")
    cnt = cur.fetchone()[0]
    cur.close()
    conn.close()

    assert cnt >= 1
