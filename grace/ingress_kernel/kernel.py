"""
Ingress Kernel - Main service for reliable data ingestion pipeline.
Provides Hunter-style ingestion while staying policy-safe under Governance.
"""
import asyncio
import hashlib
import json
import logging
from datetime import datetime
from ..utils.datetime_utils import utc_now, iso_format, format_for_filename
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from grace.contracts.ingress_contracts import (
    SourceConfig, RawEvent, NormRecord, IngressSnapshot, IngressExperience,
    generate_event_id, generate_record_id
)
from grace.contracts.ingress_events import (
    IngressEvent, IngressEventType,
    SourceRegisteredPayload, CapturedRawPayload, NormalizedPayload
)


logger = logging.getLogger(__name__)


class IngressKernel:
    """
    Main Ingress Kernel providing reliable data ingestion pipeline.
    
    Features:
    - Multi-source ingestion (HTTP, RSS, S3, GitHub, etc.)
    - Content parsing and normalization
    - Policy enforcement and PII handling
    - Trust scoring and quality metrics
    - Bronze/Silver/Gold data tiers
    - Snapshot/rollback capabilities
    - MLT integration for meta-learning
    """
    
    def __init__(self, 
                 event_bus=None,
                 governance_bridge=None,
                 mlt_bridge=None,
                 storage_path: str = "/tmp/ingress_storage",
                 snapshot_manager=None):
        """
        Initialize Ingress Kernel.
        
        Args:
            event_bus: Event bus for system communication
            governance_bridge: Bridge to governance system
            mlt_bridge: Bridge to MLT system
            storage_path: Base path for data storage
            snapshot_manager: Optional unified snapshot manager
        """
        self.event_bus = event_bus
        self.governance_bridge = governance_bridge
        self.mlt_bridge = mlt_bridge
        self.snapshot_manager = snapshot_manager
        
        # Storage setup
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)
        self.bronze_path = self.storage_path / "bronze"
        self.silver_path = self.storage_path / "silver"
        self.gold_path = self.storage_path / "gold"
        
        for path in [self.bronze_path, self.silver_path, self.gold_path]:
            path.mkdir(exist_ok=True)
        
        # Internal state
        self.sources: Dict[str, SourceConfig] = {}
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            "validation": {
                "min_validity": 0.80,
                "min_trust": 0.70,
                "block_on_pii": True
            },
            "dedupe": {
                "threshold": 0.87,
                "strategy": "hash+fuzzy"
            },
            "enrichment": {
                "ner_model": "ner.en@2.4.1",
                "embeddings": "text-encoder@1.1.0"
            },
            "governance": {
                "retention_days_default": 365,
                "label_default": "internal"
            }
        }
        
        # Initialize core components
        from .adapters.base import AdapterFactory
        from .parsers.base import ParserFactory
        from .scoring.trust import TrustScorer
        from .validators.policy import PolicyGuard
        from .snapshots.manager import SnapshotManager
        
        self.adapter_factory = AdapterFactory()
        self.parser_factory = ParserFactory()
        self.trust_scorer = TrustScorer(self.config)
        self.policy_guard = PolicyGuard(self.config)
        self.snapshot_manager = SnapshotManager(str(self.storage_path / "snapshots"))
        
        self.running = False
        logger.info("Ingress Kernel initialized")
    
    async def start(self):
        """Start the ingress kernel."""
        if self.running:
            return
        
        logger.info("Starting Ingress Kernel...")
        self.running = True
        
        # Start background tasks
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._process_scheduled_sources())
        
        logger.info("Ingress Kernel started successfully")
    
    async def stop(self):
        """Stop the ingress kernel."""
        logger.info("Stopping Ingress Kernel...")
        self.running = False
        
        # Stop active jobs
        for job_id in list(self.active_jobs.keys()):
            await self._stop_job(job_id)
        
        logger.info("Ingress Kernel stopped")
    
    def register_source(self, cfg: Dict[str, Any]) -> str:
        """
        Register a new ingestion source.
        
        Args:
            cfg: Source configuration dictionary
            
        Returns:
            Source ID
        """
        try:
            source_config = SourceConfig(**cfg)
            self.sources[source_config.source_id] = source_config
            
            # Initialize health tracking
            self.health_status[source_config.source_id] = {
                "status": "registered",
                "last_check": utc_now(),
                "error_count": 0
            }
            
            # Emit registration event
            if self.event_bus:
                event = IngressEvent(
                    event_type=IngressEventType.SOURCE_REGISTERED,
                    correlation_id=f"src_reg_{source_config.source_id}",
                    payload=SourceRegisteredPayload(source=source_config.dict())
                )
                asyncio.create_task(self.event_bus.publish(event.dict()))
            
            logger.info(f"Source registered: {source_config.source_id}")
            return source_config.source_id
            
        except Exception as e:
            logger.error(f"Failed to register source: {e}")
            raise
    
    async def capture(self, source_id: str, payload: Any, headers: Optional[Dict[str, Any]] = None) -> str:
        """
        Capture raw data from a source (webhook/push endpoint).
        
        Args:
            source_id: Source identifier
            payload: Raw payload (bytes, dict, or string)
            headers: Optional HTTP headers
            
        Returns:
            Raw event ID
        """
        if source_id not in self.sources:
            raise ValueError(f"Unknown source: {source_id}")
        
        source_config = self.sources[source_id]
        
        # Generate raw event
        raw_event = RawEvent(
            event_id=generate_event_id(),
            source_id=source_id,
            kind=source_config.parser.value,
            payload=payload,
            headers=headers,
            offset=f"manual_{iso_format()}",
            hash=self._compute_content_hash(payload)
        )
        
        # Store in bronze tier
        await self._store_bronze(raw_event)
        
        # Emit capture event
        if self.event_bus:
            event = IngressEvent(
                event_type=IngressEventType.CAPTURED_RAW,
                correlation_id=raw_event.event_id,
                payload=CapturedRawPayload(event=raw_event.dict())
            )
            asyncio.create_task(self.event_bus.publish(event.dict()))
        
        # Process asynchronously
        asyncio.create_task(self._process_raw_event(raw_event))
        
        logger.info(f"Captured raw event: {raw_event.event_id}")
        return raw_event.event_id
    
    def get_source(self, source_id: str) -> Optional[SourceConfig]:
        """Get source configuration."""
        return self.sources.get(source_id)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        return {
            "status": "running" if self.running else "stopped",
            "sources": len(self.sources),
            "active_jobs": len(self.active_jobs),
            "health_checks": self.health_status
        }
    
    async def export_snapshot(self) -> Dict[str, Any]:
        """Export system snapshot for rollback using unified snapshot manager."""
        # Create snapshot payload with current system state
        snapshot_payload = {
            "snapshot_id": f"ing_{utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "active_sources": list(self.sources.keys()),
            "registry_hash": self._compute_registry_hash(),
            "parser_versions": {"html": "1.3.0", "pdf": "2.1.4", "asr.en": "2.4.1"},
            "dedupe_threshold": self.config["dedupe"]["threshold"],
            "pii_policy_defaults": "mask",
            "offsets": {src_id: f"current_{iso_format()}" 
                       for src_id in self.sources.keys()},
            "watermarks": {src_id: iso_format()
                          for src_id in self.sources.keys()},
            "gold_views_version": "1.2.0",
            "version": "1.0.0",
            "config": self.config,
            "runtime_info": {
                "uptime_seconds": (utc_now() - self.start_time).total_seconds() if hasattr(self, 'start_time') else 0,
                "active_sources_count": len(self.sources),
                "events_processed": getattr(self, 'events_processed', 0)
            }
        }
        
        # Calculate hash
        snapshot_dict = snapshot_payload.copy()
        snapshot_dict.pop('snapshot_id', None)
        hash_content = json.dumps(snapshot_dict, sort_keys=True, default=str)
        snapshot_payload["hash"] = hashlib.sha256(hash_content.encode()).hexdigest()
        
        # If we have a snapshot manager, use it
        if hasattr(self, 'snapshot_manager') and self.snapshot_manager:
            return await self.snapshot_manager.export_snapshot(
                component_type="ingress",
                payload=snapshot_payload,
                description=f"Ingress kernel snapshot at {iso_format()}",
                created_by="ingress_kernel"
            )
        else:
            # Fallback to original behavior
            from ..ingress_kernel.snapshots import IngressSnapshot
            snapshot = IngressSnapshot(
                snapshot_id=snapshot_payload["snapshot_id"],
                active_sources=snapshot_payload["active_sources"],
                registry_hash=snapshot_payload["registry_hash"],
                parser_versions=snapshot_payload["parser_versions"],
                dedupe_threshold=snapshot_payload["dedupe_threshold"],
                pii_policy_defaults=snapshot_payload["pii_policy_defaults"],
                offsets=snapshot_payload["offsets"],
                watermarks={k: datetime.fromisoformat(v) if isinstance(v, str) else v 
                           for k, v in snapshot_payload["watermarks"].items()},
                gold_views_version=snapshot_payload["gold_views_version"],
                hash=snapshot_payload["hash"]
            )
            return snapshot.dict()
    
    def _compute_content_hash(self, payload: Any) -> str:
        """Compute hash of content for deduplication."""
        if isinstance(payload, bytes):
            return hashlib.sha256(payload).hexdigest()
        elif isinstance(payload, str):
            return hashlib.sha256(payload.encode()).hexdigest()
        else:
            content_str = json.dumps(payload, sort_keys=True)
            return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _compute_registry_hash(self) -> str:
        """Compute hash of source registry."""
        registry_data = {src_id: src.dict() for src_id, src in self.sources.items()}
        registry_json = json.dumps(registry_data, sort_keys=True)
        return hashlib.sha256(registry_json.encode()).hexdigest()
    
    async def _store_bronze(self, raw_event: RawEvent):
        """Store raw event in bronze tier."""
        bronze_file = self.bronze_path / f"{raw_event.event_id}.json"
        with open(bronze_file, 'w') as f:
            json.dump(raw_event.dict(), f, default=str)
    
    async def _process_raw_event(self, raw_event: RawEvent):
        """Process raw event through the pipeline."""
        try:
            # Parse
            parsed_content = await self._parse_content(raw_event)
            if not parsed_content:
                return
            
            # Normalize
            norm_record = await self._normalize_content(raw_event, parsed_content)
            if not norm_record:
                return
            
            # Validate
            if not await self._validate_record(norm_record):
                return
            
            # Enrich
            await self._enrich_record(norm_record)
            
            # Store in silver tier
            await self._store_silver(norm_record)
            
            # Publish
            await self._publish_record(norm_record)
            
        except Exception as e:
            logger.error(f"Failed to process raw event {raw_event.event_id}: {e}")
    
    async def _parse_content(self, raw_event: RawEvent) -> Optional[Dict[str, Any]]:
        """Parse raw content based on type."""
        try:
            source_config = self.sources[raw_event.source_id]
            parser = self.parser_factory.create_parser(
                source_config.parser, 
                source_config.parser_opts
            )
            
            result = await parser.parse(raw_event)
            
            if not result.success:
                logger.error(f"Parsing failed for {raw_event.event_id}: {result.errors}")
                return None
            
            return result.data
            
        except Exception as e:
            logger.error(f"Parser creation/execution failed: {e}")
            return None
    
    async def _normalize_content(self, raw_event: RawEvent, parsed_content: Dict[str, Any]) -> Optional[NormRecord]:
        """Normalize parsed content to standard record."""
        source_config = self.sources[raw_event.source_id]
        
        # Create normalized record
        from grace.contracts.ingress_contracts import SourceInfo, QualityMetrics, LineageInfo
        
        norm_record = NormRecord(
            record_id=generate_record_id(),
            contract=source_config.target_contract,
            body=parsed_content,
            source=SourceInfo(
                source_id=raw_event.source_id,
                uri=source_config.uri,
                fetched_at=raw_event.ingestion_ts,
                parser=source_config.parser,
                content_hash=raw_event.hash
            ),
            quality=QualityMetrics(
                validity_score=0.9,  # Will be updated by validators
                completeness=0.85,
                freshness_minutes=0.0,
                trust_score=0.5  # Will be calculated by trust scorer
            ),
            lineage=LineageInfo(
                raw_event_id=raw_event.event_id,
                transforms=["parse", "normalize"]
            )
        )
        
        # Calculate trust score using trust scorer
        trust_score = await self.trust_scorer.compute_trust_score(norm_record, source_config)
        norm_record.quality.trust_score = trust_score
        
        # Emit normalized event
        if self.event_bus:
            event = IngressEvent(
                event_type=IngressEventType.NORMALIZED,
                correlation_id=norm_record.record_id,
                payload=NormalizedPayload(record=norm_record.dict())
            )
            asyncio.create_task(self.event_bus.publish(event.dict()))
        
        return norm_record
    
    async def _validate_record(self, record: NormRecord) -> bool:
        """Validate record against policies."""
        try:
            source_config = self.sources[record.source.source_id]
            passed, violations = await self.policy_guard.enforce_policies(record, source_config)
            
            if not passed:
                logger.warning(f"Record {record.record_id} failed validation: {violations}")
                
                # Update trust score for source
                await self.trust_scorer.update_source_reputation(
                    record.source.source_id, 
                    0.3,  # Low score for validation failure
                    {"validation_failures": violations}
                )
                
                return False
            
            # Validation passed - update source reputation positively
            await self.trust_scorer.update_source_reputation(
                record.source.source_id,
                0.9,  # High score for validation success
                {"validation_success": True}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Validation failed for {record.record_id}: {e}")
            return False
    
    async def _enrich_record(self, record: NormRecord):
        """Enrich record with additional metadata."""
        # Mock enrichment - in real implementation would call NER, geocoding, etc.
        pass
    
    async def _store_silver(self, record: NormRecord):
        """Store normalized record in silver tier."""
        silver_file = self.silver_path / f"{record.record_id}.json"
        with open(silver_file, 'w') as f:
            json.dump(record.dict(), f, default=str)
    
    async def _publish_record(self, record: NormRecord):
        """Publish record to downstream systems."""
        # Mock publishing - would route to Kafka topics, feature store, etc.
        pass
    
    async def _health_monitor(self):
        """Background health monitoring."""
        while self.running:
            try:
                for source_id in self.sources.keys():
                    self.health_status[source_id].update({
                        "last_check": utc_now(),
                        "status": "ok"
                    })
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _process_scheduled_sources(self):
        """Background processing of scheduled sources."""
        while self.running:
            try:
                # Mock scheduled processing
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Scheduled processing error: {e}")
                await asyncio.sleep(300)
    
    async def _stop_job(self, job_id: str):
        """Stop an active job."""
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]