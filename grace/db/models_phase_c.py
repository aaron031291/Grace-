"""SQLAlchemy ORM models for Phase C (learning, governance & federation).

Tables: training_bundles, model_artifacts, trust_ledger, governance_proposals, approvals,
canary_rollouts, dependency_snapshots, model_validation_results, federation_references
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Text, DateTime, Boolean, JSON, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid

Base = declarative_base()


class TrainingBundle(Base):
    __tablename__ = "training_bundles"

    bundle_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bundle_cid = Column(String, unique=True, nullable=False)
    selection_logic = Column(Text, nullable=True)
    example_cids = Column(JSON, nullable=True)
    label_source_cids = Column(JSON, nullable=True)
    created_by = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    artifact_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    artifact_cid = Column(String, unique=True, nullable=False)
    type = Column(String, nullable=True)
    version = Column(String, nullable=True)
    training_bundle_cid = Column(String, nullable=True)
    validation_metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    promoted_at = Column(DateTime(timezone=True), nullable=True)


class TrustLedger(Base):
    __tablename__ = "trust_ledger"

    entity_id = Column(String, primary_key=True)
    entity_type = Column(String, nullable=True)
    alpha = Column(String, nullable=True)
    beta = Column(String, nullable=True)
    trust_score = Column(String, nullable=True)
    last_updated = Column(DateTime(timezone=True), default=datetime.utcnow)
    history_cid = Column(String, nullable=True)


class GovernanceProposal(Base):
    __tablename__ = "governance_proposals"

    proposal_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    proposal_cid = Column(String, unique=True, nullable=False)
    report_cid = Column(String, nullable=True)
    action_json = Column(JSON, nullable=True)
    risk_score = Column(String, nullable=True)
    required_approvals = Column(Integer, nullable=True)
    status = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class Approval(Base):
    __tablename__ = "approvals"

    approval_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    proposal_id = Column(UUID(as_uuid=True), nullable=False)
    approver_id = Column(String, nullable=True)
    role = Column(String, nullable=True)
    signature = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class CanaryRollout(Base):
    __tablename__ = "canary_rollouts"

    canary_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    artifact_cid = Column(String, nullable=True)
    target_nodes = Column(JSON, nullable=True)
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    kpi_before = Column(JSON, nullable=True)
    kpi_after = Column(JSON, nullable=True)
    result_cid = Column(String, nullable=True)


class DependencySnapshot(Base):
    __tablename__ = "dependency_snapshots"

    dep_snapshot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dep_snapshot_cid = Column(String, unique=True, nullable=False)
    manifest_json = Column(JSON, nullable=True)
    os_fingerprint = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    image_cid = Column(String, nullable=True)


class ModelValidationResult(Base):
    __tablename__ = "model_validation_results"

    validation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    artifact_cid = Column(String, nullable=True)
    validation_bundle_cid = Column(String, nullable=True)
    metrics_json = Column(JSON, nullable=True)
    passed_bool = Column(Boolean, nullable=True)
    run_cid = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class FederationReference(Base):
    __tablename__ = "federation_references"

    federation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    remote_entry_cid = Column(String, nullable=True)
    local_reference_cid = Column(String, nullable=True)
    peer_url = Column(String, nullable=True)
    signature = Column(String, nullable=True)
    fetched_at = Column(DateTime(timezone=True), default=datetime.utcnow)
