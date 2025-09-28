# Grace ML Contracts

This directory contains the JSON Schema definitions and contract specifications for Grace's Machine Learning components, particularly the MLT (Memory, Learning, Trust) kernel.

## Files

### Schema Definitions
- **`ml_schemas.json`** - Core ML data structure schemas (SpecialistReport, Experience, Insight, AdaptationPlan, etc.)
- **`ml_events.json`** - Event schemas for system events and message passing
- **`ml_api.json`** - REST API endpoint specifications
- **`ml_database_schema.sql`** - Database schema for PostgreSQL/SQLite storage

### Examples
The `examples/` directory contains sample JSON documents demonstrating the schema usage:
- **`adaptation_plan.json`** - Sample adaptation plan with HPO, policy changes, and canary deployment
- **`experience.json`** - Sample experience data from training pipeline
- **`insight.json`** - Sample insight generated from experience analysis
- **`specialist_report.json`** - Sample ML specialist evaluation report
- **`governance_snapshot.json`** - Sample governance state snapshot

## Schema Overview

### Core Definitions ($defs)
- **SemVer** - Semantic version pattern (e.g., "1.2.3")
- **UID** - Unique identifier pattern (lowercase alphanumeric with dashes/underscores, 5-65 chars)
- **ISO8601** - ISO 8601 date-time format
- **Sha256** - SHA-256 hash with "sha256:" prefix

### Task Types
- `classification` - Classification tasks
- `regression` - Regression tasks  
- `clustering` - Clustering tasks
- `dimred` - Dimensionality reduction tasks
- `rl` - Reinforcement learning tasks

### Metrics by Task
- **Classification**: accuracy, f1, auroc, logloss, calibration
- **Regression**: rmse, mae, r2, mape
- **Clustering**: silhouette, davies_bouldin
- **Dimensionality Reduction**: explained_variance
- **Reinforcement Learning**: episode_return, stability

## Key Contract Types

### SpecialistReport
ML specialist evaluation results containing:
- Specialist identity and task type
- Candidate models with metrics, artifacts, and validation hashes
- Risk assessments and explanations (SHAP, feature importance)
- Dataset information and evaluation notes

### Experience  
Training/inference experience data containing:
- Source (training, inference, governance, ops)
- Context (dataset, model, environment)
- Signals (metrics, drift, fairness, latency, compliance)
- Ground truth lag and timestamp

### Insight
Generated insights from experience analysis containing:
- Insight type (performance, drift, fairness, calibration, stability, governance_alignment)
- Scope (model, specialist, policy, dataset, segment)
- Evidence and confidence level
- Recommended actions

### AdaptationPlan
Concrete adaptation plans for system improvements containing:
- Multiple action types:
  - **HPO** - Hyperparameter optimization with budget and success metrics
  - **Reweight Specialists** - Adjust specialist weights
  - **Policy Delta** - Change governance policies
  - **Canary** - Gradual model rollout
- Expected effects and risk controls
- Versioning and timestamps

## API Endpoints

- `GET /health` - Health check with version
- `POST /experience` - Submit experience data
- `GET /insights` - Retrieve insights (optionally filtered by timestamp)
- `POST /plan/propose` - Propose adaptation plan
- `GET /plans/{id}/status` - Check plan approval status
- `POST /snapshot/export` - Export system snapshot
- `POST /rollback` - Request system rollback

## Database Schema

The database schema supports:
- **mlt_experiences** - Experience data with indexed querying by source/task
- **mlt_insights** - Insights with confidence-based indexing
- **mlt_plans** - Adaptation plans with status tracking
- **mlt_snapshots** - System state snapshots with hash verification
- **mlt_specialist_reports** - ML model evaluation reports

All tables include proper constraints, indexes, and PostgreSQL-specific features like JSONB and timestamp triggers.

## Events

System events for inter-component communication:
- Experience ingestion and insight generation
- Adaptation plan proposal and governance validation
- Governance approval/rejection with rationale
- Plan application and model drift alerts
- System rollback requests and completion

## Validation

All schemas include proper validation constraints:
- Pattern matching for IDs and hashes
- Enum validation for categorical fields
- Numeric ranges for confidence scores and metrics
- Required field validation
- Type-specific structure validation

## Usage

These contracts can be used for:
1. **Validation** - Validate JSON documents against schemas
2. **Code Generation** - Generate Pydantic models or other language bindings
3. **API Documentation** - OpenAPI/Swagger documentation generation  
4. **Database Design** - Reference for table structure and constraints
5. **Testing** - Schema-based test data generation

## Integration with Grace

These contracts integrate with existing Grace components:
- Compatible with existing Pydantic v2 models in `grace/contracts/`
- Extends MLT kernel functionality in `grace/mlt_kernel_ml/`
- Supports governance validation workflows
- Provides schema-based validation for API endpoints