Here’s a tightened, production-ready README you can drop in as `README.md` for the **Grace ML Contracts** package. It’s structured for engineers (validation, codegen, CI hooks), adds explicit versioning/compatibility guidance, and keeps all the details you provided—just clearer and more actionable.

---

# Grace ML Contracts

Contracts and JSON Schemas for Grace’s Machine Learning components—especially the **MLT (Memory · Learning · Trust)** kernel. These specs are the single source of truth for events, APIs, storage, and inter-service types.

## What’s here

```
grace-ml-contracts/
├─ ml_schemas.json         # Core data structures (SpecialistReport, Experience, Insight, AdaptationPlan, …)
├─ ml_events.json          # Event payload schemas used on the bus
├─ ml_api.json             # REST API endpoint specification
├─ ml_database_schema.sql  # Postgres DDL for MLT persistence
└─ examples/
   ├─ adaptation_plan.json
   ├─ experience.json
   ├─ insight.json
   ├─ specialist_report.json
   └─ governance_snapshot.json
```

## Schema overview

### Core `$defs`

* **SemVer**: `^[0-9]+\.[0-9]+\.[0-9]+$`
* **UID**: `^[a-z][a-z0-9_-]{4,64}$`
* **ISO8601**: RFC 3339/ISO 8601 timestamp
* **Sha256**: `sha256:<64 hex>`

### Tasks & metrics

* **Tasks**: `classification | regression | clustering | dimred | rl`
* **Metrics**:

  * Classification: `accuracy, f1, auroc, logloss, calibration`
  * Regression: `rmse, mae, r2, mape`
  * Clustering: `silhouette, davies_bouldin`
  * Dimensionality reduction: `explained_variance`
  * RL: `episode_return, stability`

## Key contract types

* **SpecialistReport** — model selection outputs (candidates, artifacts, metrics, risks, SHAP/feature importance, dataset, notes).
* **Experience** — training/inference telemetry: context (dataset/model/env), signals (metrics, drift, fairness, latency, compliance), GT lag, timestamp.
* **Insight** — derived findings: type/scope, evidence, confidence, recommendation.
* **AdaptationPlan** — concrete actions:

  * `hpo` (target, budget, success_metric)
  * `reweight_specialists` (weights)
  * `policy_delta` (path/from/to)
  * `canary` (target_model, steps)
* **GovernanceSnapshot / MLTSnapshot** — state capture with sha256 hashes.

## API endpoints (ML kernel)

* `GET /health` — returns `{status:"ok", version:<SemVer>}`
* `POST /experience` — submit **Experience**
* `GET /insights?since=<ISO8601>` — fetch **Insight[]**
* `POST /plan/propose` — submit **AdaptationPlan**
* `GET /plans/{id}/status` — status: `pending|approved|rejected|applied`
* `POST /snapshot/export` — snapshot id + URI
* `POST /rollback` — request rollback (`governance|mlt|model`)

## Events

* `EXPERIENCE_INGESTED`, `MLT_INSIGHT_READY`
* `ADAPTATION_PLAN_PROPOSED`
* `GOVERNANCE_VALIDATION` → `GOVERNANCE_APPROVED|REJECTED`
* `MLT_PLAN_APPLIED`, `MODEL_DRIFT_ALERT`
* `ROLLBACK_REQUESTED` → `ROLLBACK_COMPLETED`

All event payloads validate against **`ml_events.json`** and (where referenced) definitions in **`ml_schemas.json`**.

## Database schema (PostgreSQL)

Tables in `ml_database_schema.sql`:

* `mlt_experiences` — indexed by `(source, task)` and `ts`
* `mlt_insights` — indexed by `(type, scope)`, `ts`, `confidence DESC`
* `mlt_plans` — status index + `updated_at` trigger
* `mlt_snapshots` — hash-verified sha256 state
* `mlt_specialist_reports` — specialist/task/time indexes

> Constraints enforce UID/SemVer patterns, enums via CHECKs; JSONB used for flexible payloads.

---

## Quickstart: validate & lint

### Validate a JSON document (CLI)

```bash
# Using ajv
npx ajv -s ml_schemas.json -d examples/experience.json --spec=draft2020
npx ajv -s ml_events.json  -d path/to/event.json --spec=draft2020
```

### Validate in Python (Pydantic v2)

```python
from ml_contracts import Experience, AdaptationPlan
import json, pathlib

data = json.loads(pathlib.Path("examples/experience.json").read_text())
exp = Experience.model_validate(data)  # raises on violation
print(exp.model_dump_json())
```

> We ship mirrored **Pydantic v2** models (`ml_contracts.py`, `ml_events.py`, `ml_api` handlers) that align 1:1 with these schemas.

---

## Code generation

* **Python**: (already provided) Pydantic v2 models (`ml_contracts.py`, `ml_events.py`).
* **Alt languages**: use `quicktype`, `openapi-generator`, or `jsonschema2pojo` against the JSON Schema files.
* **OpenAPI**: `ml_api.json` defines endpoints; can be imported into Swagger tools.

---

## CI recommendations

Add a pre-merge job that:

1. **Schema lint**: `$ ajv compile -s ml_schemas.json ml_events.json ml_api.json`
2. **Examples validation**: validate every file in `examples/` against its schema.
3. **Round-trip check**: parse with Pydantic models → dump → re-validate with AJV.
4. **DB drift**: run `ml_database_schema.sql` in ephemeral Postgres and `pg_dump -s` to compare.

Example GitHub Actions step:

```yaml
- name: Validate schemas
  run: |
    npx ajv compile -s ml_schemas.json
    npx ajv compile -s ml_events.json
    npx ajv compile -s ml_api.json
    for f in examples/*.json; do
      npx ajv -s ml_schemas.json -d "$f" --spec=draft2020 || exit 1
    done
```

---

## Versioning & compatibility

* **Schema SemVer**: bump **PATCH** for backward-compatible additions (new optional fields), **MINOR** for additive but breaking validation (new required fields), **MAJOR** for structural changes.
* **Wire-compat** (events): never remove/rename fields without a **MAJOR** bump; prefer adding optional fields.
* **DB schema**: provide migrations for non-compatible changes; keep JSONB field names consistent with contracts.

Compatibility matrix (guidance):

| Component              | Depends on                 | Notes                                |
| ---------------------- | -------------------------- | ------------------------------------ |
| ml_api.json            | ml_schemas.json            | Request/response shapes by `$ref`    |
| ml_events.json         | ml_schemas.json (partial)  | Opaque where external `$ref` is used |
| ml_database_schema.sql | ml_schemas.json            | IDs, enums, timestamps, JSONB shape  |
| Pydantic models        | ml_schemas/events/api.json | 1:1 field parity & validation        |

---

## Conventions (must follow)

* **UID**: `^[a-z][a-z0-9_-]{4,64}$` — generate lowercase, prefix by domain when helpful (`e_plan_…`, `i_insight_…`).
* **Timestamps**: always UTC **with** offset (`Z` or `+00:00`).
* **Hashes**: content digests formatted `sha256:<hex64>`.
* **Enums**: use provided enum values exactly; avoid free-text categories.

---

## Usage patterns

1. **Validation** — enforce contracts at API boundaries and message bus ingress/egress.
2. **Codegen** — generate typed models for clients/services.
3. **API docs** — import `ml_api.json` into Swagger UI / Redoc.
4. **DB design** — `ml_database_schema.sql` as baseline; extend with migrations.
5. **Testing** — seed fixtures from `examples/` and add property-based tests against schemas.

---

## Integration with Grace

* Fully compatible with existing Pydantic v2 models under `grace/contracts/`.
* Extends the MLT kernel (`grace/mlt_kernel_ml/`) for: experience ingestion, insight generation, adaptation planning, and governance validation.
* Plays cleanly with Grace governance thresholds and snapshots (`GovernanceSnapshot`) for approval workflows.

---

## Security & data handling

* Treat **Experience.signals** and **SpecialistReport.candidates** as potentially sensitive (model keys, dataset references).
* Avoid including raw PII in contracts; if needed, reference de-identified artifacts by UID.
* Snapshots include `sha256` state hashes—verify on restore and audit.

---

## Contributing

* Keep changes **schema-first** (update JSON schemas, regenerate models, then services).
* Add/adjust examples alongside any schema change.
* Include a CHANGELOG entry describing compatibility impact and migration notes.

---

