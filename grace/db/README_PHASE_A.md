Phase A database setup (Postgres)

This file explains how to apply the Phase A core tables migration and how to enable Postgres-backed immutable_log.

1) Apply Postgres migration

- The SQL migration is available at `migrations/002_phase_a_core_tables.sql`.
- If you use Alembic, create a new Alembic revision that includes the SQL or `op.execute(open(...).read())`.
- Alternatively, run directly against Postgres:

```bash
psql "$DATABASE_URL" -f migrations/002_phase_a_core_tables.sql
```

2) Enable Postgres-backed immutable log

- Set the `DATABASE_URL` environment variable to point to your Postgres DB (e.g. `postgresql://user:pass@localhost:5432/grace`)
- Install psycopg2 in your environment:

```bash
pip install psycopg2-binary
```

- The `grace/immutable_log.py` module will attempt to use psycopg2 when `DATABASE_URL` is present.

3) Notes and next steps

- The SQL includes a `tsv` column and trigger for full-text search; it assumes `pg_trgm`/`fuzzystrmatch` extensions are enabled if you want similarity operators. You can add them with:

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

- For production, consider using a persistent schema migration tool (Alembic) and moving indexes into dedicated migrations.
- Next recommended work: implement Phase B tables and add ORM + unit tests that exercise immutable log in Postgres mode.
