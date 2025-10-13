"""SQLAlchemy ORM models for Phase B (forensics & self-heal).

Tables: specialist_analyses, forensic_reports, incidents, remediation_actions, sandbox_runs, patches
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Text, DateTime, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
import uuid

Base = declarative_base()


class SpecialistAnalysis(Base):
    __tablename__ = "specialist_analyses"

    analysis_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_cid = Column(String, unique=True, nullable=False)
    entry_cid = Column(String, nullable=False)
    specialist_id = Column(String, nullable=False)
    domain = Column(String, nullable=True)
    confidence = Column(String, nullable=True)
    analysis_summary = Column(Text, nullable=True)
    analysis_blob_path = Column(String, nullable=True)
    signature = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class ForensicReport(Base):
    __tablename__ = "forensic_reports"

    report_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_cid = Column(String, unique=True, nullable=False)
    entry_cid = Column(String, nullable=False)
    canonical_rca_cid = Column(String, nullable=True)
    recommended_actions_json = Column(JSON, nullable=True)
    consensus_score = Column(String, nullable=True)
    dissent_json = Column(JSON, nullable=True)
    signed_by = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class Incident(Base):
    __tablename__ = "incidents"

    incident_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    primary_error_code = Column(String, nullable=True)
    entry_cids = Column(JSON, nullable=True)
    first_seen = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_seen = Column(DateTime(timezone=True), default=datetime.utcnow)
    severity = Column(String, nullable=True)
    status = Column(String, nullable=True)
    incident_summary = Column(Text, nullable=True)


class RemediationAction(Base):
    __tablename__ = "remediation_actions"

    remediation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    remediation_cid = Column(String, unique=True, nullable=False)
    proposal_id = Column(UUID(as_uuid=True), nullable=True)
    action_type = Column(String, nullable=True)
    params_json = Column(JSON, nullable=True)
    executor = Column(String, nullable=True)
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    result_cid = Column(String, nullable=True)
    success_bool = Column(Boolean, nullable=True)


class SandboxRun(Base):
    __tablename__ = "sandbox_runs"

    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_cid = Column(String, unique=True, nullable=False)
    remediation_id = Column(UUID(as_uuid=True), nullable=True)
    env_matrix = Column(JSON, nullable=True)
    test_results_cid = Column(String, nullable=True)
    kpi_snapshot_cid = Column(String, nullable=True)
    started_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String, nullable=True)


class Patch(Base):
    __tablename__ = "patches"

    patch_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patch_cid = Column(String, unique=True, nullable=False)
    source_bundle_cid = Column(String, nullable=True)
    author = Column(String, nullable=True)
    diff_summary = Column(Text, nullable=True)
    applied_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String, nullable=True)
    validation_cid = Column(String, nullable=True)
