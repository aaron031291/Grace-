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
import psycopg2

MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "migrations")


def _apply_sql_sqlite(db_path: str, sql_text: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(sql_text)
    conn.commit()
    conn.close()


def _apply_sql_postgres(database_url: str, sql_text: str):
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
