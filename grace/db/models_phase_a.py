"""SQLAlchemy ORM models for Phase A core tables.

This file mirrors `migrations/002_phase_a_core_tables.sql` and is intended for SQLAlchemy/Alchemist-style wiring.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Boolean,
    BigInteger,
    JSON,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, TSVECTOR
from sqlalchemy.ext.declarative import declarative_base
import uuid

Base = declarative_base()


class ImmutableEntry(Base):
    __tablename__ = "immutable_entries"

    entry_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entry_cid = Column(String, unique=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    actor = Column(JSON, nullable=True)
    operation = Column(String, nullable=True)
    error_code = Column(String, nullable=True)
    severity = Column(String, nullable=True)
    what = Column(Text, nullable=True)
    why = Column(Text, nullable=True)
    how = Column(Text, nullable=True)
    where = Column(JSON, nullable=True)
    who = Column(JSON, nullable=True)
    text_summary = Column(Text, nullable=True)
    payload_path = Column(String, nullable=True)
    signature = Column(String, nullable=True)
    tags = Column(ARRAY(String), nullable=True)
    payload_json = Column(JSON, nullable=True)
    # tsvector column is DB-specific; add via migrations


class EvidenceBlob(Base):
    __tablename__ = "evidence_blobs"

    blob_cid = Column(String, primary_key=True)
    storage_path = Column(String, nullable=False)
    content_type = Column(String, nullable=True)
    size = Column(BigInteger, nullable=True)
    checksum = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    access_policy = Column(JSON, nullable=True)


class EventLog(Base):
    __tablename__ = "event_logs"

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String, nullable=False)
    payload_cid = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)
    severity = Column(String, nullable=True)
    payload_json = Column(JSON, nullable=True)


class UserAccount(Base):
    __tablename__ = "user_accounts"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, nullable=False)
    display_name = Column(String, nullable=True)
    role = Column(String, nullable=True)
    public_key = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    active_bool = Column(Boolean, default=True)


class KpiSnapshot(Base):
    __tablename__ = "kpi_snapshots"

    kpi_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_cid = Column(String, nullable=True)
    metrics_json = Column(JSON, nullable=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)


class SearchIndexMeta(Base):
    __tablename__ = "search_index_meta"

    embedding_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entry_cid = Column(String, nullable=False, index=True)
    embed_model = Column(String, nullable=True)
    vector_store_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


# Optional indexes (create via migration for portability)
Index("ix_immutable_entries_tags", ImmutableEntry.tags, postgresql_using="gin")
Index("ix_search_index_meta_entry", SearchIndexMeta.entry_cid)
