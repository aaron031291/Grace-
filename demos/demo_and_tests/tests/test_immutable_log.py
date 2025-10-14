import os
import json
import importlib
import asyncio


def test_append_and_get_sqlite(tmp_path):
    # Ensure sqlite file path is used. Set env BEFORE importing the module.
    dbfile = tmp_path / "test_immutable.sqlite"
    os.environ.pop("DATABASE_URL", None)
    os.environ["GRACE_SQLITE_PATH"] = str(dbfile)

    # Now import (fresh) the module so it picks up env var
    mod = importlib.import_module("grace.immutable_log")
    importlib.reload(mod)

    entry = {
        "who": {"actor_id": "tester", "actor_type": "user", "actor_display": "Tester"},
        "what": "unit test append",
        "why": "testing",
        "how": "pytest",
        "text_summary": "a test entry",
    }

    res = asyncio.run(mod.append_entry(entry))

    assert isinstance(res, dict)
    assert "entry_id" in res
    assert "entry_cid" in res

    fetched = asyncio.run(mod.get_entry(res["entry_cid"]))
    assert fetched["what"] == "unit test append"


def test_postgres_optional():
    # This test is skipped unless DATABASE_URL is present in the environment
    if not os.environ.get("DATABASE_URL"):
        return

    mod = importlib.import_module("grace.immutable_log")
    importlib.reload(mod)

    entry = {
        "who": {"actor_id": "tester_pg", "actor_type": "user", "actor_display": "TesterPG"},
        "what": "pg test append",
        "why": "testing",
        "how": "pytest",
        "text_summary": "a pg test entry",
    }

    res = asyncio.run(mod.append_entry(entry))
    assert isinstance(res, dict)
    fetched = asyncio.run(mod.get_entry(res["entry_cid"]))
    assert fetched["what"] == "pg test append"
