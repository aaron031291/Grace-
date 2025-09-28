"""
Trust Core Kernel - Trust and credibility weighting system for sources, claims, and components.
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import statistics
import math
import logging

from core.contracts import Experience


logger = logging.getLogger(__name__)


class TrustProfile:
    """Profile tracking trust metrics for an entity."""
    
    def __init__(self, entity_id: str, entity_type: str):
        self.entity_id = entity_id
        self.entity_type = entity_type  # "source", "component", "specialist", "user"
        self.base_trust = 0.5  # Initial trust score
        self.current_trust = 0.5
        self.history = []  # List of trust events
        self.last_updated = datetime.now()
        self.interaction_count = 0
        self.positive_outcomes = 0
        self.negative_outcomes = 0
        
        # Specialized metrics
        self.accuracy_history = []
        self.reliability_score = 0.5
        self.consistency_score = 0.5
        self.expertise_relevance = {}  # Domain -> relevance score
    
    def update_trust(self, outcome_score: float, context: Dict[str, Any]):
        """Update trust based on outcome."""
        self.interaction_count += 1
        
        # Record outcome
        if outcome_score > 0.6:
            self.positive_outcomes += 1
        elif outcome_score < 0.4:
            self.negative_outcomes += 1
        
        # Update accuracy history
        self.accuracy_history.append(outcome_score)
        if len(self.accuracy_history) > 100:  # Keep recent history
            self.accuracy_history = self.accuracy_history[-100:]
        
        # Calculate new trust score with exponential moving average
        alpha = 0.1  # Learning rate
        self.current_trust = (1 - alpha) * self.current_trust + alpha * outcome_score
        
        # Update derived metrics
        self._update_derived_metrics()
        
        # Record trust event
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "outcome_score": outcome_score,
            "new_trust": self.current_trust,
            "context": context
        })
        
        self.last_updated = datetime.now()
    
    def _update_derived_metrics(self):
        """Update derived trust metrics."""
        if self.accuracy_history:
            self.reliability_score = statistics.mean(self.accuracy_history)
            
            # Consistency as inverse of standard deviation
            if len(self.accuracy_history) > 1:
                std_dev = statistics.stdev(self.accuracy_history)
                self.consistency_score = max(0.0, 1.0 - std_dev)
            else:
                self.consistency_score = 1.0
    
    def get_weighted_trust(self, domain: Optional[str] = None) -> float:
        """Get domain-weighted trust score."""
        base_trust = self.current_trust
        
        if domain and domain in self.expertise_relevance:
            domain_weight = self.expertise_relevance[domain]
            return base_trust * domain_weight
        
        return base_trust
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "current_trust": self.current_trust,
            "reliability_score": self.reliability_score,
            "consistency_score": self.consistency_score,
            "interaction_count": self.interaction_count,
            "positive_outcomes": self.positive_outcomes,
            "negative_outcomes": self.negative_outcomes,
            "last_updated": self.last_updated.isoformat(),
            "expertise_relevance": self.expertise_relevance
        }


class TrustCoreKernel:
    """
    Trust and credibility weighting system for the governance kernel.
    Manages trust profiles, calculates credibility scores, and provides
    trust-based weighting for governance decisions.
    """
    
    def __init__(self, event_bus, memory_core):
        self.event_bus = event_bus
        self.memory_core = memory_core
        self.trust_profiles: Dict[str, TrustProfile] = {}
        self.trust_decay_rate = 0.01  # Daily decay rate
        self.min_interactions_for_reliability = 5
        
        # Initialize with some baseline trust profiles
        self._initialize_baseline_profiles()
        
        # Setup decay task
        asyncio.create_task(self._periodic_trust_decay())
    
    def _initialize_baseline_profiles(self):
        """Initialize trust profiles for core components."""
        core_components = [
            ("verification_engine", "component", 0.8),
            ("unified_logic", "component", 0.8),
            ("governance_engine", "component", 0.9),
            ("parliament", "component", 0.9),
            ("constitutional_checker", "component", 0.95),
            ("fairness_explainability", "specialist", 0.85),
            ("privacy_security", "specialist", 0.85),
        ]
        
        for entity_id, entity_type, initial_trust in core_components:
            profile = TrustProfile(entity_id, entity_type)
            profile.base_trust = initial_trust
            profile.current_trust = initial_trust
            self.trust_profiles[entity_id] = profile
    
    async def get_trust_score(self, entity_id: str, 
                            domain: Optional[str] = None) -> float:
        """
        Get trust score for an entity, optionally weighted by domain expertise.
        
        Args:
            entity_id: ID of the entity to get trust for
            domain: Optional domain for domain-weighted trust
            
        Returns:
            Trust score between 0.0 and 1.0
        """
        if entity_id not in self.trust_profiles:
            # Create new profile with default trust
            await self._create_trust_profile(entity_id)
        
        profile = self.trust_profiles[entity_id]
        return profile.get_weighted_trust(domain)
    
    async def update_trust(self, entity_id: str, outcome_score: float,
                         context: Optional[Dict[str, Any]] = None) -> float:
        """
        Update trust score based on outcome.
        
        Args:
            entity_id: ID of the entity to update
            outcome_score: Score representing outcome quality (0.0 to 1.0)
            context: Additional context for the trust update
            
        Returns:
            New trust score
        """
        if context is None:
            context = {}
        
        if entity_id not in self.trust_profiles:
            await self._create_trust_profile(entity_id)
        
        profile = self.trust_profiles[entity_id]
        profile.update_trust(outcome_score, context)
        
        # Record trust update experience
        await self._record_trust_experience(entity_id, outcome_score, context)
        
        return profile.current_trust
    
    async def calculate_source_credibility(self, sources: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate credibility scores for a list of sources.
        
        Args:
            sources: List of source dictionaries with 'uri' and optional 'credibility'
            
        Returns:
            Dictionary mapping source URIs to credibility scores
        """
        credibility_scores = {}
        
        for source in sources:
            uri = source.get("uri", "")
            provided_credibility = source.get("credibility")
            
            if provided_credibility is not None:
                # Use provided credibility but validate with our trust system
                trust_score = await self.get_trust_score(uri)
                # Weight provided credibility with our trust assessment
                final_credibility = 0.7 * provided_credibility + 0.3 * trust_score
            else:
                # Calculate credibility based on URI patterns and trust history
                final_credibility = await self._calculate_uri_credibility(uri)
            
            credibility_scores[uri] = min(max(final_credibility, 0.0), 1.0)
        
        return credibility_scores
    
    async def _calculate_uri_credibility(self, uri: str) -> float:
        """Calculate credibility for a URI based on patterns and history."""
        base_credibility = 0.5
        
        # Domain-based credibility patterns
        if any(domain in uri.lower() for domain in ['.edu', '.gov']):
            base_credibility = 0.8
        elif any(domain in uri.lower() for domain in ['.org']):
            base_credibility = 0.7
        elif uri.startswith('https://'):
            base_credibility = 0.6
        elif uri.startswith('http://'):
            base_credibility = 0.4
        
        # Check if we have trust history for this source
        trust_score = await self.get_trust_score(uri)
        
        # Combine base credibility with trust history
        if self.trust_profiles.get(uri, {}).interaction_count > self.min_interactions_for_reliability:
            return 0.3 * base_credibility + 0.7 * trust_score
        else:
            return 0.8 * base_credibility + 0.2 * trust_score
    
    async def calculate_component_trust_weights(self, components: List[str],
                                              domain: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate trust weights for a list of components.
        
        Args:
            components: List of component IDs
            domain: Optional domain for domain-specific weighting
            
        Returns:
            Dictionary mapping component IDs to trust weights
        """
        trust_weights = {}
        
        for component_id in components:
            trust_score = await self.get_trust_score(component_id, domain)
            trust_weights[component_id] = trust_score
        
        return trust_weights
    
    async def validate_claim_trustworthiness(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the trustworthiness of a claim based on its sources and evidence.
        
        Args:
            claim: Claim dictionary with sources and evidence
            
        Returns:
            Trust validation results
        """
        sources = claim.get("sources", [])
        evidence = claim.get("evidence", [])
        
        # Calculate source credibility
        source_credibility = await self.calculate_source_credibility(sources)
        avg_source_credibility = statistics.mean(source_credibility.values()) if source_credibility else 0.3
        
        # Assess evidence trustworthiness
        evidence_trust = await self._assess_evidence_trust(evidence)
        
        # Check for corroborating sources
        corroboration_score = await self._calculate_corroboration_score(sources, evidence)
        
        # Calculate overall trust score
        overall_trust = (
            0.4 * avg_source_credibility +
            0.3 * evidence_trust +
            0.3 * corroboration_score
        )
        
        return {
            "overall_trust": overall_trust,
            "source_credibility": source_credibility,
            "evidence_trust": evidence_trust,
            "corroboration_score": corroboration_score,
            "trust_factors": {
                "source_count": len(sources),
                "evidence_count": len(evidence),
                "avg_source_credibility": avg_source_credibility
            }
        }
    
    async def _assess_evidence_trust(self, evidence: List[Dict[str, Any]]) -> float:
        """Assess trustworthiness of evidence items."""
        if not evidence:
            return 0.2
        
        trust_scores = []
        
        for item in evidence:
            evidence_type = item.get("type", "unknown")
            pointer = item.get("pointer", "")
            
            # Base trust by evidence type
            type_trust = {
                "db": 0.8,      # Database evidence is usually reliable
                "doc": 0.7,     # Document evidence varies
                "api": 0.6,     # API evidence depends on source
                "unknown": 0.3
            }.get(evidence_type, 0.3)
            
            # Adjust based on pointer availability and quality
            if pointer:
                if pointer.startswith("https://"):
                    type_trust *= 1.1
                elif pointer.startswith("http://"):
                    type_trust *= 0.9
            else:
                type_trust *= 0.7  # No pointer reduces trust
            
            trust_scores.append(min(type_trust, 1.0))
        
        return statistics.mean(trust_scores)
    
    async def _calculate_corroboration_score(self, sources: List[Dict[str, Any]],
                                           evidence: List[Dict[str, Any]]) -> float:
        """Calculate corroboration score based on source diversity and evidence alignment."""
        if not sources:
            return 0.1
        
        # Source diversity bonus
        unique_domains = set()
        for source in sources:
            uri = source.get("uri", "")
            if "://" in uri:
                domain = uri.split("://")[1].split("/")[0]
                unique_domains.add(domain)
        
        diversity_score = min(len(unique_domains) / max(len(sources), 1), 1.0)
        
        # Evidence alignment bonus
        alignment_score = 0.5
        if evidence:
            # Simple heuristic: more evidence types = better alignment
            evidence_types = set(item.get("type", "") for item in evidence)
            alignment_score = min(len(evidence_types) / 3.0, 1.0)
        
        return 0.6 * diversity_score + 0.4 * alignment_score
    
    async def _create_trust_profile(self, entity_id: str, 
                                  entity_type: str = "unknown") -> TrustProfile:
        """Create a new trust profile for an entity."""
        profile = TrustProfile(entity_id, entity_type)
        
        # Set initial trust based on entity type
        initial_trust_by_type = {
            "component": 0.6,
            "specialist": 0.5,
            "source": 0.4,
            "user": 0.3,
            "unknown": 0.2
        }
        
        initial_trust = initial_trust_by_type.get(entity_type, 0.2)
        profile.base_trust = initial_trust
        profile.current_trust = initial_trust
        
        self.trust_profiles[entity_id] = profile
        logger.info(f"Created trust profile for {entity_id} ({entity_type})")
        
        return profile
    
    async def _record_trust_experience(self, entity_id: str, outcome_score: float,
                                     context: Dict[str, Any]):
        """Record trust update as learning experience."""
        experience = Experience(
            type="TRUST_UPDATE",
            component_id="trust_core_kernel",
            context={
                "entity_id": entity_id,
                "entity_type": self.trust_profiles[entity_id].entity_type,
                "previous_trust": context.get("previous_trust", 0.5)
            },
            outcome={
                "outcome_score": outcome_score,
                "new_trust": self.trust_profiles[entity_id].current_trust,
                "interaction_count": self.trust_profiles[entity_id].interaction_count
            },
            success_score=outcome_score,
            timestamp=datetime.now()
        )
        
        self.memory_core.store_experience(experience)
        
        # Emit trust update event
        await self.event_bus.publish("TRUST_UPDATED", {
            "entity_id": entity_id,
            "new_trust": self.trust_profiles[entity_id].current_trust,
            "outcome_score": outcome_score
        })
    
    async def _periodic_trust_decay(self):
        """Periodic task to apply trust decay over time."""
        while True:
            await asyncio.sleep(24 * 3600)  # Run daily
            
            current_time = datetime.now()
            
            for profile in self.trust_profiles.values():
                # Calculate days since last update
                days_since_update = (current_time - profile.last_updated).days
                
                if days_since_update > 0:
                    # Apply exponential decay
                    decay_factor = math.exp(-self.trust_decay_rate * days_since_update)
                    
                    # Decay towards base trust level
                    profile.current_trust = (
                        profile.base_trust + 
                        (profile.current_trust - profile.base_trust) * decay_factor
                    )
                    
                    profile.last_updated = current_time
            
            logger.info("Applied periodic trust decay")
    
    def set_domain_expertise(self, entity_id: str, domain: str, relevance: float):
        """Set domain expertise relevance for an entity."""
        if entity_id not in self.trust_profiles:
            asyncio.create_task(self._create_trust_profile(entity_id))
        
        profile = self.trust_profiles[entity_id]
        profile.expertise_relevance[domain] = max(0.0, min(1.0, relevance))
        
        logger.info(f"Set {entity_id} expertise in {domain}: {relevance:.3f}")
    
    def get_trust_profile(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get trust profile for an entity."""
        if entity_id not in self.trust_profiles:
            return None
        
        return self.trust_profiles[entity_id].to_dict()
    
    def get_trust_statistics(self) -> Dict[str, Any]:
        """Get overall trust system statistics."""
        if not self.trust_profiles:
            return {
                "total_profiles": 0,
                "average_trust": 0.0,
                "total_interactions": 0
            }
        
        trust_scores = [p.current_trust for p in self.trust_profiles.values()]
        total_interactions = sum(p.interaction_count for p in self.trust_profiles.values())
        
        return {
            "total_profiles": len(self.trust_profiles),
            "average_trust": statistics.mean(trust_scores),
            "median_trust": statistics.median(trust_scores),
            "trust_std_dev": statistics.stdev(trust_scores) if len(trust_scores) > 1 else 0,
            "total_interactions": total_interactions,
            "profiles_by_type": self._get_profiles_by_type()
        }
    
    def _get_profiles_by_type(self) -> Dict[str, int]:
        """Get count of profiles by entity type."""
        type_counts = {}
        for profile in self.trust_profiles.values():
            entity_type = profile.entity_type
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return type_counts
    
    async def cleanup_stale_profiles(self, days_threshold: int = 90):
        """Remove trust profiles that haven't been updated in a long time."""
        current_time = datetime.now()
        stale_profiles = []
        
        for entity_id, profile in self.trust_profiles.items():
            days_since_update = (current_time - profile.last_updated).days
            if days_since_update > days_threshold and profile.interaction_count == 0:
                stale_profiles.append(entity_id)
        
        for entity_id in stale_profiles:
            del self.trust_profiles[entity_id]
            logger.info(f"Removed stale trust profile: {entity_id}")
        
        return len(stale_profiles)