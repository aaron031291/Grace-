"""Simple migration runner: apply all SQL files in migrations/ in order to the target DB.

Usage:
    python scripts/apply_migrations.py --dry-run
    DATABASE_URL=postgresql://... python scripts/apply_migrations.py

If DATABASE_URL is not set, it will apply SQL files to a local SQLite DB at GRACE_SQLITE_PATH.
"""
from __future__ import annotations

import os
import glob
import argparse
import sqlite3
import re

MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "migrations")


def _apply_sql_sqlite(db_path: str, sql_text: str):
    # SQLite can't execute many Postgres-specific statements. Extract and run
    # only CREATE TABLE statements so the schema tables are created for dev.
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Find CREATE TABLE ... ) blocks (non-greedy)
    pattern = re.compile(r"(?is)(CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\w\.\"]+\s*\(.*?\))\s*;?")
    matches = pattern.findall(sql_text)
    if matches:
        for stmt in matches:
            try:
                cur.executescript(stmt + ";")
            except sqlite3.OperationalError:
                # If there's some syntax we can't handle, skip that statement
                # to allow other tables to be created.
                continue
    else:
        # Fallback: try to run the whole script and let SQLite report errors
        cur.executescript(sql_text)

    conn.commit()
    conn.close()


def _apply_sql_postgres(database_url: str, sql_text: str):
    try:
        import psycopg2
    except Exception as e:
        raise RuntimeError(
            "psycopg2 is required to apply migrations to Postgres but it's not installed"
        ) from e

    conn = psycopg2.connect(database_url)
    cur = conn.cursor()
    cur.execute(sql_text)
    conn.commit()
    cur.close()
    conn.close()


def main(dry_run: bool = False):
    database_url = os.environ.get("DATABASE_URL")
    sqlite_path = os.environ.get("GRACE_SQLITE_PATH", "./grace_dev.sqlite3")

    files = sorted(glob.glob(os.path.join(MIGRATIONS_DIR, "*.sql")))
    print(f"Found {len(files)} migration files")

    for path in files:
        print("--- applying:", path)
        with open(path, "r") as f:
            sql_text = f.read()
        if dry_run:
            print(sql_text[:400])
            continue

        if database_url:
            print("Applying to Postgres")
            _apply_sql_postgres(database_url, sql_text)
        else:
            print("Applying to SQLite at", sqlite_path)
            _apply_sql_sqlite(sqlite_path, sql_text)

    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print SQL but do not execute")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
