"""
Semantic Bridge system for Cross-Domain Knowledge Linking.

Provides infrastructure to automatically correlate and link entities across
different domains (books, logs, metrics, memories) enabling unified retrieval
and semantic graph overlay capabilities.
"""
import asyncio
import logging
import re
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import json
import hashlib
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities that can be linked."""
    PERSON = "person"
    CONCEPT = "concept"
    LOCATION = "location"
    DOCUMENT = "document"
    EVENT = "event"
    METRIC = "metric"
    MEMORY = "memory"
    POLICY = "policy"
    UNKNOWN = "unknown"


class LinkType(Enum):
    """Types of semantic links between entities."""
    SAME_AS = "same_as"              # Exact same entity
    SIMILAR_TO = "similar_to"        # Similar entities
    RELATES_TO = "relates_to"        # Related entities
    CONTAINS = "contains"            # One entity contains another
    PART_OF = "part_of"             # Entity is part of another
    MENTIONS = "mentions"            # One entity mentions another
    CAUSES = "causes"                # Causal relationship
    PRECEDES = "precedes"            # Temporal relationship
    CONFLICTED_WITH = "conflicts"    # Conflicting information


@dataclass
class EntitySignature:
    """Normalized signature for entity identification."""
    canonical_name: str
    entity_type: EntityType
    aliases: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    def __post_init__(self):
        self.canonical_name = self._normalize_name(self.canonical_name)
        self.aliases = {self._normalize_name(alias) for alias in self.aliases}
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        # Remove extra whitespace, convert to lowercase
        normalized = re.sub(r'\s+', ' ', name.strip().lower())
        # Remove common prefixes/suffixes
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)
        normalized = re.sub(r'\s+(jr|sr|inc|ltd|corp)\.?$', '', normalized)
        return normalized
    
    def matches(self, other: 'EntitySignature', threshold: float = 0.8) -> float:
        """Calculate similarity score with another entity signature."""
        if self.entity_type != other.entity_type and self.entity_type != EntityType.UNKNOWN and other.entity_type != EntityType.UNKNOWN:
            return 0.0
        
        # Check exact matches
        if self.canonical_name == other.canonical_name:
            return 1.0
        
        # Check aliases
        all_names_self = {self.canonical_name} | self.aliases
        all_names_other = {other.canonical_name} | other.aliases
        
        if all_names_self & all_names_other:  # Set intersection
            return 0.95
        
        # Calculate string similarity
        max_similarity = 0.0
        for name1 in all_names_self:
            for name2 in all_names_other:
                similarity = SequenceMatcher(None, name1, name2).ratio()
                max_similarity = max(max_similarity, similarity)
        
        # Attribute similarity boost
        if self.attributes and other.attributes:
            attr_matches = 0
            total_attrs = len(self.attributes) + len(other.attributes)
            for key, value in self.attributes.items():
                if key in other.attributes and str(value).lower() == str(other.attributes[key]).lower():
                    attr_matches += 2
            
            if total_attrs > 0:
                attr_boost = attr_matches / total_attrs
                max_similarity = min(1.0, max_similarity + attr_boost * 0.3)
        
        return max_similarity if max_similarity >= threshold else 0.0


@dataclass
class SemanticLink:
    """Link between entities across domains."""
    source_entity_id: str
    target_entity_id: str
    link_type: LinkType
    confidence: float
    evidence: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_reinforced: datetime = field(default_factory=datetime.utcnow)
    reinforcement_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def reinforce(self, new_evidence: str = None):
        """Reinforce this link with additional evidence."""
        self.reinforcement_count += 1
        self.last_reinforced = datetime.utcnow()
        self.confidence = min(1.0, self.confidence + 0.1)  # Increase confidence
        
        if new_evidence:
            self.evidence.append(new_evidence)
    
    def decay(self, decay_rate: float = 0.01):
        """Apply time-based decay to link confidence."""
        time_since_reinforcement = datetime.utcnow() - self.last_reinforced
        days_since = time_since_reinforcement.days
        
        if days_since > 0:
            decay_factor = decay_rate * days_since
            self.confidence = max(0.1, self.confidence - decay_factor)


@dataclass
class SemanticEntity:
    """Complete entity representation with links."""
    entity_id: str
    signature: EntitySignature
    domain: str  # Source domain (books, logs, metrics, etc.)
    source_reference: str  # Reference to original source
    content_summary: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    tags: Set[str] = field(default_factory=set)
    
    def update_from_mention(self, mention_context: str, attributes: Dict[str, Any] = None):
        """Update entity based on new mention."""
        self.last_updated = datetime.utcnow()
        self.access_count += 1
        
        if attributes:
            self.signature.attributes.update(attributes)
        
        # Update content summary with new context
        if mention_context and mention_context not in self.content_summary:
            if self.content_summary:
                self.content_summary += f" | {mention_context[:200]}"
            else:
                self.content_summary = mention_context[:200]


class EntityExtractor:
    """Extracts entities from various content types."""
    
    def __init__(self):
        # Simple patterns for entity recognition
        self.patterns = {
            EntityType.PERSON: [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:said|wrote|mentioned|discussed|argued)',
                r'\b((?:Dr|Mr|Ms|Mrs|Professor)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+is|\'s|,|\.|:)',
            ],
            EntityType.CONCEPT: [
                r'\b(artificial intelligence|machine learning|neural network|deep learning)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:concept|theory|principle|approach)',
                r'\b(?:concept|idea|theory|principle)\s+of\s+([A-Z][a-z]+(?:\s+[a-z]+)*)',
            ],
            EntityType.LOCATION: [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:City|County|State|Country|University|Lab)',
                r'\b(New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose)\b',
                r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            ],
            EntityType.DOCUMENT: [
                r'\b(paper|article|book|report|study|document)\s+"([^"]+)"',
                r'\b"([^"]+)"\s+(?:paper|article|book|report)',
                r'\breference\s+to\s+"([^"]+)"',
            ],
            EntityType.POLICY: [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:policy|rule|regulation|guideline)',
                r'\b(?:policy|rule)\s+(?:on|about|regarding)\s+([A-Z][a-z]+(?:\s+[a-z]+)*)',
            ]
        }
    
    def extract_entities(self, content: str, domain: str = "unknown") -> List[Tuple[EntitySignature, str]]:
        """Extract entities from content."""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else match[1]
                    
                    if len(match.strip()) > 2:  # Filter out very short matches
                        signature = EntitySignature(
                            canonical_name=match.strip(),
                            entity_type=entity_type,
                            confidence=0.8
                        )
                        
                        # Get context around the match
                        match_pos = content.lower().find(match.lower())
                        if match_pos != -1:
                            context_start = max(0, match_pos - 50)
                            context_end = min(len(content), match_pos + len(match) + 50)
                            context = content[context_start:context_end].strip()
                        else:
                            context = match
                        
                        entities.append((signature, context))
        
        return entities


class SemanticBridge:
    """
    Main semantic bridge system for cross-domain knowledge linking.
    
    Manages entity extraction, linking, and semantic graph maintenance
    across different domains in the Grace system.
    """
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        
        # Core storage
        self.entities: Dict[str, SemanticEntity] = {}
        self.links: Dict[str, SemanticLink] = {}
        
        # Indexes for fast lookup
        self.domain_index: Dict[str, Set[str]] = defaultdict(set)
        self.type_index: Dict[EntityType, Set[str]] = defaultdict(set)
        self.name_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Components
        self.entity_extractor = EntityExtractor()
        
        # Configuration
        self.linking_threshold = 0.7  # Lowered threshold for better matching
        self.max_entities_per_domain = 10000
        self.link_decay_rate = 0.01
        self.cleanup_interval_seconds = 3600  # 1 hour
        
        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Statistics
        self.stats = {
            "entities_created": 0,
            "links_created": 0,
            "extractions_performed": 0,
            "cross_domain_links": 0,
            "start_time": datetime.utcnow()
        }
    
    async def start(self):
        """Start the semantic bridge system."""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting Semantic Bridge...")
        
        # Start maintenance task
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        # Subscribe to relevant events if event bus available
        if self.event_bus:
            await self.event_bus.subscribe("content_ingested", self._handle_content_ingested)
            await self.event_bus.subscribe("memory_stored", self._handle_memory_stored)
            await self.event_bus.subscribe("document_processed", self._handle_document_processed)
        
        logger.info(f"Semantic Bridge started with {len(self.entities)} entities")
    
    async def stop(self):
        """Stop the semantic bridge system.""" 
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping Semantic Bridge...")
        
        # Cancel maintenance task
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Semantic Bridge stopped")
    
    async def process_content(self, content: str, domain: str, source_reference: str, metadata: Dict[str, Any] = None) -> List[str]:
        """
        Process content to extract entities and create links.
        Returns list of entity IDs found.
        """
        if not self.running:
            return []
        
        self.stats["extractions_performed"] += 1
        
        # Extract entities
        extracted_entities = self.entity_extractor.extract_entities(content, domain)
        entity_ids = []
        
        for signature, context in extracted_entities:
            entity_id = await self._add_or_update_entity(signature, domain, source_reference, context, metadata)
            if entity_id:
                entity_ids.append(entity_id)
        
        # Create links between co-occurring entities
        if len(entity_ids) > 1:
            await self._create_co_occurrence_links(entity_ids, content)
        
        # Publish processing event
        if self.event_bus:
            await self.event_bus.publish("semantic_processing_completed", {
                "domain": domain,
                "source_reference": source_reference,
                "entities_found": len(entity_ids),
                "new_links_created": 0  # Would need to track this
            })
        
        logger.debug(f"Processed content from {domain}, found {len(entity_ids)} entities")
        return entity_ids
    
    async def _add_or_update_entity(self, signature: EntitySignature, domain: str, source_reference: str, context: str, metadata: Dict[str, Any] = None) -> Optional[str]:
        """Add new entity or update existing one."""
        
        # Check for existing entity
        existing_entity_id = await self._find_matching_entity(signature)
        
        if existing_entity_id:
            # Update existing entity
            entity = self.entities[existing_entity_id]
            entity.update_from_mention(context, signature.attributes)
            
            # Add new aliases if any
            entity.signature.aliases.update(signature.aliases)
            
            # Update domain index if this is a new domain for the entity
            if entity.domain != domain:
                self.domain_index[domain].add(existing_entity_id)
                # Create implicit cross-domain link for same entity
                await self._create_cross_domain_same_entity_link(existing_entity_id, domain, entity.domain)
            
            return existing_entity_id
        else:
            # Create new entity
            entity_id = f"entity_{uuid.uuid4().hex[:12]}"
            
            entity = SemanticEntity(
                entity_id=entity_id,
                signature=signature,
                domain=domain,
                source_reference=source_reference,
                content_summary=context[:200]
            )
            
            if metadata:
                entity.tags.update(metadata.get("tags", []))
            
            # Store entity
            self.entities[entity_id] = entity
            
            # Update indexes
            self.domain_index[domain].add(entity_id)
            self.type_index[signature.entity_type].add(entity_id)
            self.name_index[signature.canonical_name].add(entity_id)
            
            for alias in signature.aliases:
                self.name_index[alias].add(entity_id)
            
            self.stats["entities_created"] += 1
            return entity_id
    
    async def _create_cross_domain_same_entity_link(self, entity_id: str, domain1: str, domain2: str):
        """Create a link indicating the same entity appears in multiple domains."""
        link_id = f"cross_domain_{entity_id}_{hash(f'{domain1}_{domain2}') % 10000}"
        
        if link_id not in self.links:
            link = SemanticLink(
                source_entity_id=entity_id,
                target_entity_id=entity_id,  # Self-link for cross-domain presence
                link_type=LinkType.SAME_AS,
                confidence=1.0,
                evidence=[f"Same entity appears in {domain1} and {domain2}"],
                metadata={"cross_domain": True, "domains": [domain1, domain2]}
            )
            
            self.links[link_id] = link
            self.stats["cross_domain_links"] += 1
    
    async def _find_matching_entity(self, signature: EntitySignature) -> Optional[str]:
        """Find existing entity that matches the given signature."""
        
        # Check exact name matches first
        candidates = set()
        candidates.update(self.name_index.get(signature.canonical_name, set()))
        
        for alias in signature.aliases:
            candidates.update(self.name_index.get(alias, set()))
        
        # Check type-based candidates
        type_candidates = self.type_index.get(signature.entity_type, set())
        
        # Score all candidates
        best_score = 0.0
        best_entity_id = None
        
        for entity_id in candidates | type_candidates:
            entity = self.entities[entity_id]
            score = signature.matches(entity.signature, self.linking_threshold)
            
            if score > best_score:
                best_score = score
                best_entity_id = entity_id
        
        return best_entity_id if best_score >= self.linking_threshold else None
    
    async def _create_co_occurrence_links(self, entity_ids: List[str], content: str):
        """Create links between entities that co-occur in content."""
        for i, entity_id1 in enumerate(entity_ids):
            for entity_id2 in entity_ids[i+1:]:
                link_id = f"{entity_id1}--{entity_id2}"
                reverse_link_id = f"{entity_id2}--{entity_id1}"
                
                # Check if link already exists
                if link_id in self.links:
                    self.links[link_id].reinforce(f"Co-occurrence in content")
                elif reverse_link_id in self.links:
                    self.links[reverse_link_id].reinforce(f"Co-occurrence in content")
                else:
                    # Create new link
                    link = SemanticLink(
                        source_entity_id=entity_id1,
                        target_entity_id=entity_id2,
                        link_type=LinkType.RELATES_TO,
                        confidence=0.6,
                        evidence=[f"Co-occurrence in content"],
                        metadata={"link_source": "co_occurrence"}
                    )
                    
                    self.links[link_id] = link
                    self.stats["links_created"] += 1
                    
                    # Check if this is a cross-domain link
                    entity1 = self.entities[entity_id1]
                    entity2 = self.entities[entity_id2]
                    if entity1.domain != entity2.domain:
                        self.stats["cross_domain_links"] += 1
    
    async def find_related_entities(self, entity_name: str, max_depth: int = 2, min_confidence: float = 0.5) -> Dict[str, Any]:
        """Find entities related to the given entity name."""
        
        # Find the base entity
        signature = EntitySignature(canonical_name=entity_name, entity_type=EntityType.UNKNOWN)
        base_entity_id = await self._find_matching_entity(signature)
        
        if not base_entity_id:
            return {"entity_name": entity_name, "found": False, "related_entities": []}
        
        # Traverse the semantic graph
        related_entities = []
        visited = set()
        queue = deque([(base_entity_id, 0)])  # (entity_id, depth)
        
        while queue and len(related_entities) < 50:  # Limit results
            current_entity_id, depth = queue.popleft()
            
            if current_entity_id in visited or depth > max_depth:
                continue
            
            visited.add(current_entity_id)
            current_entity = self.entities[current_entity_id]
            
            if current_entity_id != base_entity_id:  # Don't include the base entity
                related_entities.append({
                    "entity_id": current_entity_id,
                    "name": current_entity.signature.canonical_name,
                    "type": current_entity.signature.entity_type.value,
                    "domain": current_entity.domain,
                    "depth": depth,
                    "summary": current_entity.content_summary
                })
            
            # Find linked entities
            if depth < max_depth:
                for link_id, link in self.links.items():
                    if link.confidence < min_confidence:
                        continue
                    
                    next_entity_id = None
                    if link.source_entity_id == current_entity_id:
                        next_entity_id = link.target_entity_id
                    elif link.target_entity_id == current_entity_id:
                        next_entity_id = link.source_entity_id
                    
                    if next_entity_id and next_entity_id not in visited:
                        queue.append((next_entity_id, depth + 1))
        
        return {
            "entity_name": entity_name,
            "base_entity_id": base_entity_id,
            "found": True,
            "related_entities": related_entities,
            "total_found": len(related_entities)
        }
    
    async def get_cross_domain_connections(self, domain1: str, domain2: str) -> List[Dict[str, Any]]:
        """Get connections between two domains."""
        connections = []
        
        domain1_entities = self.domain_index.get(domain1, set())
        domain2_entities = self.domain_index.get(domain2, set())
        
        # Also check for same entities across domains
        for entity_id in domain1_entities:
            entity = self.entities[entity_id]
            # Look for similar entities in domain2 by name/signature matching
            for other_entity_id in domain2_entities:
                other_entity = self.entities[other_entity_id]
                similarity = entity.signature.matches(other_entity.signature, threshold=0.6)
                
                if similarity > 0.6:
                    connections.append({
                        "source_entity": {
                            "id": entity.entity_id,
                            "name": entity.signature.canonical_name,
                            "type": entity.signature.entity_type.value,
                            "domain": entity.domain
                        },
                        "target_entity": {
                            "id": other_entity.entity_id,
                            "name": other_entity.signature.canonical_name,
                            "type": other_entity.signature.entity_type.value,
                            "domain": other_entity.domain
                        },
                        "link_type": "same_as" if similarity > 0.9 else "similar_to",
                        "confidence": similarity,
                        "evidence": [f"Name similarity: {similarity:.2f}"]
                    })
        
        # Check actual links between domains
        for link_id, link in self.links.items():
            source_in_domain1 = link.source_entity_id in domain1_entities
            target_in_domain2 = link.target_entity_id in domain2_entities
            source_in_domain2 = link.source_entity_id in domain2_entities
            target_in_domain1 = link.target_entity_id in domain1_entities
            
            if (source_in_domain1 and target_in_domain2) or (source_in_domain2 and target_in_domain1):
                source_entity = self.entities[link.source_entity_id]
                target_entity = self.entities[link.target_entity_id]
                
                connections.append({
                    "source_entity": {
                        "id": source_entity.entity_id,
                        "name": source_entity.signature.canonical_name,
                        "type": source_entity.signature.entity_type.value,
                        "domain": source_entity.domain
                    },
                    "target_entity": {
                        "id": target_entity.entity_id,
                        "name": target_entity.signature.canonical_name,
                        "type": target_entity.signature.entity_type.value,
                        "domain": target_entity.domain
                    },
                    "link_type": link.link_type.value,
                    "confidence": link.confidence,
                    "evidence": link.evidence
                })
        
        return sorted(connections, key=lambda x: x["confidence"], reverse=True)
    
    async def _maintenance_loop(self):
        """Background maintenance loop."""
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                await self._perform_maintenance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Semantic bridge maintenance error: {e}")
    
    async def _perform_maintenance(self):
        """Perform maintenance tasks."""
        # Apply link decay
        decayed_links = []
        for link_id, link in self.links.items():
            link.decay(self.link_decay_rate)
            if link.confidence < 0.1:
                decayed_links.append(link_id)
        
        # Remove very weak links
        for link_id in decayed_links:
            del self.links[link_id]
        
        if decayed_links:
            logger.info(f"Removed {len(decayed_links)} decayed links")
        
        # Cleanup entities with no links and low access
        if len(self.entities) > self.max_entities_per_domain:
            await self._cleanup_old_entities()
    
    async def _cleanup_old_entities(self):
        """Remove old, unused entities."""
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        
        # Find entities with no links and low usage
        entities_to_remove = []
        for entity_id, entity in self.entities.items():
            if (entity.last_updated < cutoff_time and 
                entity.access_count < 3 and
                not self._entity_has_links(entity_id)):
                entities_to_remove.append(entity_id)
        
        # Remove entities
        for entity_id in entities_to_remove:
            await self._remove_entity(entity_id)
        
        if entities_to_remove:
            logger.info(f"Cleaned up {len(entities_to_remove)} unused entities")
    
    def _entity_has_links(self, entity_id: str) -> bool:
        """Check if entity has any links."""
        for link in self.links.values():
            if link.source_entity_id == entity_id or link.target_entity_id == entity_id:
                return True
        return False
    
    async def _remove_entity(self, entity_id: str):
        """Remove entity and update indexes."""
        if entity_id not in self.entities:
            return
        
        entity = self.entities[entity_id]
        
        # Remove from indexes
        self.domain_index[entity.domain].discard(entity_id)
        self.type_index[entity.signature.entity_type].discard(entity_id)
        self.name_index[entity.signature.canonical_name].discard(entity_id)
        
        for alias in entity.signature.aliases:
            self.name_index[alias].discard(entity_id)
        
        # Remove entity
        del self.entities[entity_id]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        uptime = (datetime.utcnow() - self.stats["start_time"]).total_seconds()
        
        # Domain distribution
        domain_stats = {domain: len(entities) for domain, entities in self.domain_index.items()}
        
        # Type distribution
        type_stats = {entity_type.value: len(entities) for entity_type, entities in self.type_index.items()}
        
        # Link type distribution
        link_type_stats = defaultdict(int)
        for link in self.links.values():
            link_type_stats[link.link_type.value] += 1
        
        return {
            "uptime_seconds": uptime,
            "running": self.running,
            "total_entities": len(self.entities),
            "total_links": len(self.links),
            "domain_distribution": domain_stats,
            "entity_type_distribution": type_stats,
            "link_type_distribution": dict(link_type_stats),
            "processing_stats": self.stats.copy(),
            "average_entity_links": len(self.links) * 2 / max(1, len(self.entities))
        }
    
    # Event handlers
    async def _handle_content_ingested(self, event):
        """Handle content ingestion events."""
        try:
            payload = event.get("payload", {})
            content = payload.get("content", "")
            domain = payload.get("domain", "unknown")
            source_ref = payload.get("source_reference", "")
            
            if content and len(content) > 10:  # Only process substantial content
                await self.process_content(content, domain, source_ref)
        except Exception as e:
            logger.error(f"Error handling content ingestion: {e}")
    
    async def _handle_memory_stored(self, event):
        """Handle memory storage events."""
        try:
            payload = event.get("payload", {})
            content = payload.get("content", "")
            memory_id = payload.get("memory_id", "")
            
            if content:
                await self.process_content(content, "memory", memory_id)
        except Exception as e:
            logger.error(f"Error handling memory storage: {e}")
    
    async def _handle_document_processed(self, event):
        """Handle document processing events."""
        try:
            payload = event.get("payload", {})
            content = payload.get("content", "")
            doc_id = payload.get("document_id", "")
            
            if content:
                await self.process_content(content, "documents", doc_id)
        except Exception as e:
            logger.error(f"Error handling document processing: {e}")