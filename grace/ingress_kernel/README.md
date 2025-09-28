# Grace Ingress Kernel

The Ingress Kernel is a comprehensive data ingestion pipeline that provides Hunter-style ingestion capabilities while staying policy-safe under Governance. It reliably ingests, validates, normalizes, enriches, and publishes data from multiple sources with built-in trust scoring and quality metrics.

## 🚀 Features

### Core Pipeline
- **Multi-source ingestion**: HTTP, RSS, S3, GCS, Azure Blob, GitHub, YouTube, Podcasts, Social media, Kafka, MQTT, SQL, CSV
- **Content parsing**: JSON, CSV, HTML, PDF, Audio (ASR), Video (VTT), XML
- **Schema validation**: Contract compliance and data quality checks
- **PII enforcement**: Detection, masking, hashing, or blocking based on policy
- **Trust scoring**: Source reputation and content quality assessment
- **Data tiers**: Bronze (raw), Silver (normalized), Gold (curated features)

### Security & Governance  
- **Policy enforcement**: PII, schema, format, and governance compliance
- **Trust & reputation**: Dynamic source trust scoring with historical tracking
- **Content validation**: Schema compliance and quality thresholds
- **Governance integration**: Approval workflows for high-risk sources
- **Audit trails**: Complete lineage tracking from ingestion to publication

### Operations
- **Health monitoring**: Source health, metrics, and alerting
- **Snapshot & rollback**: Point-in-time recovery capabilities
- **Configuration management**: Dynamic source registration and updates
- **REST API**: Complete management interface with OpenAPI docs
- **Event integration**: Event Mesh, Governance, and MLT bridges

### Meta-Learning
- **Experience emission**: Performance metrics and quality feedback to MLT
- **Adaptation plans**: Consume MLT recommendations for auto-optimization
- **Trust evolution**: Learn source reliability patterns over time

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Ingress Kernel  │    │   Downstream    │
│                 │    │                  │    │    Systems      │
│ • HTTP APIs     │───▶│ ┌──────────────┐ │───▶│ • Specialists   │
│ • RSS Feeds     │    │ │   Adapters   │ │    │ • Feature Store │
│ • S3 Buckets    │    │ └──────────────┘ │    │ • Event Mesh    │
│ • GitHub Repos  │    │ ┌──────────────┐ │    │ • Data Catalog  │
│ • Media Files   │    │ │   Parsers    │ │    │                 │
│ • Social Media  │    │ └──────────────┘ │    └─────────────────┘
└─────────────────┘    │ ┌──────────────┐ │
                       │ │  Validators  │ │    ┌─────────────────┐
┌─────────────────┐    │ └──────────────┘ │    │   Governance    │
│   Governance    │◀───│ ┌──────────────┐ │───▶│    System       │
│    System       │    │ │ Trust Scorer │ │    │                 │
└─────────────────┘    │ └──────────────┘ │    └─────────────────┘
                       │ ┌──────────────┐ │
┌─────────────────┐    │ │  Snapshots   │ │    ┌─────────────────┐
│   MLT Kernel    │◀───│ └──────────────┘ │───▶│  Storage Tiers  │
│                 │    │ ┌──────────────┐ │    │                 │
└─────────────────┘    │ │ REST API     │ │    │ • Bronze (Raw)  │
                       │ └──────────────┘ │    │ • Silver (Norm) │
                       └──────────────────┘    │ • Gold (Curated)│
                                               └─────────────────┘
```

## 📦 Installation & Usage

### Basic Usage

```python
import asyncio
from grace.ingress_kernel import IngressKernel

async def main():
    # Initialize kernel
    kernel = IngressKernel(storage_path="/path/to/storage")
    await kernel.start()
    
    # Register a source
    source_config = {
        "source_id": "src_news_api",
        "kind": "http",
        "uri": "https://api.news.com/articles",
        "auth_mode": "bearer",
        "parser": "json",
        "target_contract": "contract:article.v1",
        "pii_policy": "mask",
        "governance_label": "public"
    }
    
    source_id = kernel.register_source(source_config)
    
    # Capture data
    article_data = {
        "title": "Breaking News",
        "content": "Article content...",
        "author": "Reporter Name"
    }
    
    event_id = await kernel.capture(source_id, article_data)
    print(f"Captured event: {event_id}")
    
    await kernel.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### FastAPI Service

```python
from grace.ingress_kernel import create_ingress_app

# Create FastAPI application
app = create_ingress_app()

# Run with uvicorn
# uvicorn main:app --host 0.0.0.0 --port 8000
```

### REST API Endpoints

- `GET /api/ingress/v1/health` - Health check
- `POST /api/ingress/v1/sources` - Register source
- `GET /api/ingress/v1/sources/{source_id}` - Get source config
- `POST /api/ingress/v1/capture` - Capture data
- `GET /api/ingress/v1/records/{record_id}` - Get normalized record
- `POST /api/ingress/v1/snapshot/export` - Create snapshot
- `POST /api/ingress/v1/rollback` - Rollback to snapshot
- `GET /api/ingress/v1/metrics` - Get ingestion metrics

## 🔧 Configuration

See [config_example.yaml](config_example.yaml) for complete configuration options.

Key configuration sections:
- **Validation**: Trust and quality thresholds
- **Trust scoring**: Weighting factors for trust calculation
- **PII policies**: Detection patterns and handling rules
- **Storage**: Data tier paths and retention
- **Monitoring**: Health checks and metrics
- **Bridges**: Integration with other systems

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_ingress_kernel.py
```

Run the interactive demo:

```bash
python demo_ingress_kernel.py
```

## 🔒 Security & Compliance

### PII Handling
- **Detection**: Automatic PII pattern recognition (SSN, credit cards, emails, phones)
- **Policies**: Block, mask, hash, or allow with consent
- **Validation**: Governance integration for policy compliance

### Trust Scoring
Formula: `trust_score = w1*source_reputation + w2*schema_compliance + w3*parser_confidence + w4*cross_source_agreement + w5*freshness_factor - w6*pii_risk`

### Data Governance
- Source approval workflows for high-risk sources
- Retention policy enforcement
- Classification labels (public, internal, restricted)
- Audit trail maintenance

## 📊 Data Tiers

### Bronze (Raw Events)
- Immutable storage of original content
- Complete payload preservation
- Deduplication by content hash
- Watermark and offset tracking

### Silver (Normalized Records)
- Contract-compliant structured data
- Quality metrics and trust scores
- PII handling applied
- Lineage tracking

### Gold (Curated Features)
- Topic extraction and entity linking
- Feature store integration
- ML-ready datasets
- Performance optimized views

## 🔗 Integration

### Event Mesh Bridge
Routes ingress events to appropriate systems based on event type and content.

### Governance Bridge
- Source approval workflows
- Policy validation requests
- Risk assessment integration

### MLT Bridge
- Experience emission for meta-learning
- Adaptation plan consumption
- Performance metric feedback

## 📈 Monitoring & Metrics

### Health Metrics
- Source connectivity and latency
- Processing success rates
- Error rates by stage
- Trust score trends

### Quality Metrics
- Schema validation rates
- PII incident tracking
- Deduplication effectiveness
- Completeness scores

### Performance Metrics
- Throughput (events/second)
- Latency (end-to-end processing time)
- Resource utilization
- Storage growth

## 🔄 Snapshot & Rollback

### Snapshot Contents
- Active source configurations
- Parser versions and settings
- Policy configurations
- Stream offsets and watermarks
- Trust score baselines

### Rollback Process
1. Create pre-rollback safety snapshot
2. Stop active ingestion jobs
3. Reset configurations to snapshot state
4. Resume operations with rolled-back state
5. Emit rollback completion events

## 🚦 Operational Patterns

### Source Management
1. **Registration**: Define source with contract and policies
2. **Validation**: Governance approval for high-risk sources  
3. **Monitoring**: Health checks and performance tracking
4. **Adaptation**: MLT-driven optimization of parsers and thresholds

### Data Flow
1. **Capture**: Raw events stored in Bronze tier
2. **Parse**: Content extraction based on type
3. **Normalize**: Schema mapping and validation
4. **Validate**: PII and governance policy enforcement
5. **Enrich**: Entity extraction and topic classification
6. **Persist**: Storage in Silver tier with quality metrics
7. **Publish**: Route to downstream systems via event topics

### Quality Assurance
1. **Contract validation**: Schema compliance checking
2. **Trust scoring**: Dynamic source reputation tracking  
3. **PII detection**: Pattern-based sensitive data identification
4. **Governance enforcement**: Policy compliance validation
5. **Lineage tracking**: Complete audit trail maintenance

## 🤝 Contributing

The Ingress Kernel follows Grace system architecture patterns:
- Event-driven communication via Event Mesh
- Policy enforcement through Governance integration
- Meta-learning feedback via MLT bridges
- Comprehensive audit trails for all operations

For development, see the test suite and demo scripts for usage examples.

## 📄 License

Part of the Grace system - see main repository for license information.