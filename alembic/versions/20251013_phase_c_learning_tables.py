"""Phase C learning & governance tables wrapper

Revision ID: 20251013_phase_c
Revises: 20251013_phase_b
Create Date: 2025-10-13
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20251013_phase_c'
down_revision = '20251013_phase_b'
branch_labels = None
depends_on = None


def upgrade():
    sql = open('migrations/004_phase_c_learning_tables.sql', 'r').read()
    op.execute(sa.text(sql))


def downgrade():
    # Drops for Phase C tables
    op.execute("DROP TABLE IF EXISTS training_bundles CASCADE;")
    op.execute("DROP TABLE IF EXISTS model_artifacts CASCADE;")
    op.execute("DROP TABLE IF EXISTS trust_ledger CASCADE;")
    op.execute("DROP TABLE IF EXISTS governance_proposals CASCADE;")
    op.execute("DROP TABLE IF EXISTS approvals CASCADE;")
    op.execute("DROP TABLE IF EXISTS canary_rollouts CASCADE;")
    op.execute("DROP TABLE IF EXISTS dependency_snapshots CASCADE;")
    op.execute("DROP TABLE IF EXISTS model_validation_results CASCADE;")
    op.execute("DROP TABLE IF EXISTS federation_references CASCADE;")
