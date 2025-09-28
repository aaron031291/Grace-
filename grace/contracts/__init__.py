"""
Grace contracts - Pydantic v2 models and JSON Schema contracts for inter-kernel communication.

This package contains both Pydantic models for runtime validation and JSON Schema
definitions for ML/MLT (Memory, Learning, Trust) kernel operations.

JSON Schema files:
- ml_schemas.json: Core ML data structures (SpecialistReport, Experience, Insight, AdaptationPlan, etc.)
- ml_events.json: Event schemas for system messaging
- ml_api.json: REST API endpoint specifications 
- ml_database_schema.sql: Database schema definitions

Examples are provided in the examples/ directory demonstrating schema usage.
"""