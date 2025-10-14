"""Phase A core tables wrapper

Revision ID: 20251013_phase_a
Revises: 6046a445fabc
Create Date: 2025-10-13
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20251013_phase_a'
down_revision = '6046a445fabc'
branch_labels = None
depends_on = None


def upgrade():
    sql = open('migrations/002_phase_a_core_tables.sql', 'r').read()
    op.execute(sa.text(sql))


def downgrade():
    # Downgrade not implemented; drop tables explicitly if needed
    op.execute("DROP TABLE IF EXISTS immutable_entries CASCADE;")
    op.execute("DROP TABLE IF EXISTS evidence_blobs CASCADE;")
    op.execute("DROP TABLE IF EXISTS event_logs CASCADE;")
    op.execute("DROP TABLE IF EXISTS user_accounts CASCADE;")
    op.execute("DROP TABLE IF EXISTS kpi_snapshots CASCADE;")
    op.execute("DROP TABLE IF EXISTS search_index_meta CASCADE;")
