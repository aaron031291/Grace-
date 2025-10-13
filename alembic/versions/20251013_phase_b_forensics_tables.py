"""Phase B forensics tables wrapper

Revision ID: 20251013_phase_b
Revises: 20251013_phase_a
Create Date: 2025-10-13
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20251013_phase_b'
down_revision = '20251013_phase_a'
branch_labels = None
depends_on = None


def upgrade():
    sql = open('migrations/003_phase_b_forensics_tables.sql', 'r').read()
    op.execute(sa.text(sql))


def downgrade():
    op.execute("DROP TABLE IF EXISTS specialist_analyses CASCADE;")
    op.execute("DROP TABLE IF EXISTS forensic_reports CASCADE;")
    op.execute("DROP TABLE IF EXISTS incidents CASCADE;")
    op.execute("DROP TABLE IF EXISTS remediation_actions CASCADE;")
    op.execute("DROP TABLE IF EXISTS sandbox_runs CASCADE;")
    op.execute("DROP TABLE IF EXISTS patches CASCADE;")
