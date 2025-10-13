from fastapi import FastAPI
from fastapi.testclient import TestClient
import os
import importlib


def make_app():
    # Ensure the env uses sqlite for the test
    os.environ.pop("DATABASE_URL", None)
    os.environ["GRACE_SQLITE_PATH"] = "./test_immutable_http.sqlite"

    # Import module after env set
    mod = importlib.import_module("grace.immutable_log")
    importlib.reload(mod)

    app = FastAPI()
    app.include_router(mod.router)
    return app


def test_http_append_get_search():
    app = make_app()
    client = TestClient(app)

    entry = {
        "who": {"actor_id": "http_tester", "actor_type": "user", "actor_display": "HTTP Tester"},
        "what": "http test append",
        "why": "integration",
        "how": "testclient",
        "text_summary": "http test entry",
    }

    r = client.post("/api/v1/logs", json=entry)
    assert r.status_code == 200
    body = r.json()
    assert "entry_cid" in body

    cid = body["entry_cid"]
    r2 = client.get(f"/api/v1/logs/{cid}")
    assert r2.status_code == 200
    assert r2.json()["what"] == "http test append"

    # search
    r3 = client.post("/api/v1/logs/search", json={"query": "http test"})
    assert r3.status_code == 200
    results = r3.json()["results"]
    assert any(r["entry_cid"] == cid for r in results)
