"""
Trust and Quality Scoring System for Ingress Kernel.
"""
import logging
import hashlib
from typing import Dict, Any, Optional, List
from datetime import timedelta
from grace.utils.time import now_utc
import math

from grace.contracts.ingress_contracts import NormRecord, SourceConfig


logger = logging.getLogger(__name__)


class TrustScorer:
    """Calculates trust scores for records and sources."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize trust scorer with configuration."""
        self.config = config or {}
        
        # Default weights for trust score calculation
        self.default_weights = {
            "w1": 0.25,  # source_reputation
            "w2": 0.20,  # schema_compliance
            "w3": 0.15,  # parser_confidence
            "w4": 0.20,  # cross_source_agreement
            "w5": 0.10,  # freshness_factor
            "w6": 0.10   # pii_risk (subtracted)
        }
        
        self.weights = self.config.get("trust_weights", self.default_weights)
        
        # Source reputation cache
        self.source_reputation: Dict[str, float] = {}
        self.reputation_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Cross-source agreement cache
        self.content_signatures: Dict[str, List[Dict[str, Any]]] = {}
        
    async def compute_trust_score(self, record: NormRecord, 
                                source_config: SourceConfig) -> float:
        """
        Compute trust score for a normalized record.
        
        Formula: trust_score = w1*source_reputation 
                            + w2*(1 - schema_violations_norm)
                            + w3*parser_confidence 
                            + w4*cross_source_agreement
                            + w5*freshness_factor 
                            - w6*pii_risk
        
        Args:
            record: Normalized record to score
            source_config: Source configuration
            
        Returns:
            Trust score between 0.0 and 1.0
        """
        try:
            # Calculate individual factors
            source_rep = await self._get_source_reputation(source_config.source_id)
            schema_compliance = await self._calculate_schema_compliance(record)
            parser_confidence = await self._calculate_parser_confidence(record)
            cross_agreement = await self._calculate_cross_source_agreement(record)
            freshness = await self._calculate_freshness_factor(record)
            pii_risk = await self._calculate_pii_risk(record)
            
            # Apply weighted formula
            trust_score = (
                self.weights["w1"] * source_rep +
                self.weights["w2"] * schema_compliance +
                self.weights["w3"] * parser_confidence +
                self.weights["w4"] * cross_agreement +
                self.weights["w5"] * freshness -
                self.weights["w6"] * pii_risk
            )
            
            # Clamp to [0, 1] range
            trust_score = max(0.0, min(1.0, trust_score))
            
            logger.debug(f"Trust score for {record.record_id}: {trust_score:.3f}")
            
            return trust_score
            
        except Exception as e:
            logger.error(f"Trust score calculation failed: {e}")
            return 0.5  # Default neutral score
    
    async def update_source_reputation(self, source_id: str, outcome_score: float,
                                     context: Optional[Dict[str, Any]] = None):
        """
        Update source reputation based on processing outcome.
        
        Args:
            source_id: Source identifier
            outcome_score: Score based on processing success (0.0-1.0)
            context: Additional context for the update
        """
        try:
            current_rep = self.source_reputation.get(source_id, 0.7)  # Default reputation
            
            # Exponential moving average with learning rate
            learning_rate = self.config.get("reputation_learning_rate", 0.1)
            new_rep = current_rep * (1 - learning_rate) + outcome_score * learning_rate
            
            self.source_reputation[source_id] = max(0.0, min(1.0, new_rep))
            
            # Store in history
            if source_id not in self.reputation_history:
                self.reputation_history[source_id] = []
            
            self.reputation_history[source_id].append({
                "timestamp": now_utc(),
                "old_reputation": current_rep,
                "new_reputation": new_rep,
                "outcome_score": outcome_score,
                "context": context or {}
            })
            
            # Limit history size
            max_history = self.config.get("max_reputation_history", 1000)
            if len(self.reputation_history[source_id]) > max_history:
                self.reputation_history[source_id] = self.reputation_history[source_id][-max_history:]
            
            logger.debug(f"Updated reputation for {source_id}: {current_rep:.3f} -> {new_rep:.3f}")
            
        except Exception as e:
            logger.error(f"Reputation update failed: {e}")
    
    async def _get_source_reputation(self, source_id: str) -> float:
        """Get current source reputation."""
        return self.source_reputation.get(source_id, 0.7)  # Default reputation
    
    async def _calculate_schema_compliance(self, record: NormRecord) -> float:
        """Calculate schema compliance score."""
        try:
            # Use existing validity score from quality metrics
            validity = record.quality.validity_score
            
            # Convert to compliance score (1 - normalized violations)
            return validity
            
        except Exception as e:
            logger.error(f"Schema compliance calculation failed: {e}")
            return 0.5
    
    async def _calculate_parser_confidence(self, record: NormRecord) -> float:
        """Calculate parser confidence score."""
        try:
            # Mock parser confidence based on completeness and content type
            completeness = record.quality.completeness
            
            # Different parsers have different base confidence levels
            parser_type = record.source.parser.lower()
            base_confidence = {
                "json": 0.95,
                "csv": 0.90,
                "html": 0.80,
                "xml": 0.85,
                "pdf": 0.70,
                "audio": 0.65,
                "video": 0.60
            }.get(parser_type, 0.75)
            
            # Combine base confidence with completeness
            confidence = base_confidence * completeness
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Parser confidence calculation failed: {e}")
            return 0.75
    
    async def _calculate_cross_source_agreement(self, record: NormRecord) -> float:
        """Calculate cross-source agreement score."""
        try:
            # Create content signature for similarity matching
            content_sig = await self._create_content_signature(record)
            
            # Look for similar content from other sources
            similar_count = 0
            total_sources = max(1, len(self.content_signatures))
            
            for sig, sources in self.content_signatures.items():
                if self._signatures_similar(content_sig, sig):
                    # Count unique sources with similar content
                    unique_sources = set(s["source_id"] for s in sources)
                    unique_sources.discard(record.source.source_id)  # Exclude self
                    similar_count += len(unique_sources)
            
            # Store this signature
            if content_sig not in self.content_signatures:
                self.content_signatures[content_sig] = []
            
            self.content_signatures[content_sig].append({
                "source_id": record.source.source_id,
                "record_id": record.record_id,
                "timestamp": now_utc()
            })
            
            # Calculate agreement score
            agreement = min(1.0, similar_count / max(1, total_sources * 0.1))
            
            return agreement
            
        except Exception as e:
            logger.error(f"Cross-source agreement calculation failed: {e}")
            return 0.5
    
    async def _calculate_freshness_factor(self, record: NormRecord) -> float:
        """Calculate freshness factor based on data recency."""
        try:
            freshness_minutes = record.quality.freshness_minutes
            
            # Exponential decay: fresh data gets higher scores
            # Half-life of 1 day (1440 minutes)
            half_life_minutes = self.config.get("freshness_half_life", 1440)
            
            freshness_factor = math.exp(-freshness_minutes / half_life_minutes * math.log(2))
            
            return min(1.0, freshness_factor)
            
        except Exception as e:
            logger.error(f"Freshness calculation failed: {e}")
            return 0.5
    
    async def _calculate_pii_risk(self, record: NormRecord) -> float:
        """Calculate PII risk factor."""
        try:
            pii_flags = record.quality.pii_flags
            
            if not pii_flags:
                return 0.0  # No PII risk
            
            # Risk increases with number and severity of PII types
            risk_weights = {
                "ssn": 0.9,
                "credit_card": 0.8,
                "email": 0.3,
                "phone": 0.4,
                "address": 0.5,
                "name": 0.2
            }
            
            total_risk = 0.0
            for flag in pii_flags:
                flag_lower = flag.lower()
                for pii_type, weight in risk_weights.items():
                    if pii_type in flag_lower:
                        total_risk += weight
                        break
                else:
                    total_risk += 0.5  # Unknown PII type
            
            # Normalize and cap risk
            return min(1.0, total_risk / max(1, len(pii_flags)))
            
        except Exception as e:
            logger.error(f"PII risk calculation failed: {e}")
            return 0.2  # Default low risk
    
    async def _create_content_signature(self, record: NormRecord) -> str:
        """Create signature for content similarity matching."""
        try:
            # Create hash from key content fields
            content_parts = []
            
            # Add title-like fields
            for field in ["title", "name", "subject", "headline"]:
                if field in record.body:
                    content_parts.append(str(record.body[field]).lower().strip())
            
            # Add main content
            for field in ["content", "text", "body", "description"]:
                if field in record.body:
                    text = str(record.body[field]).lower().strip()
                    # Use first 500 characters for signature
                    content_parts.append(text[:500])
            
            if not content_parts:
                # Fallback to record hash
                return record.source.content_hash[:16]
            
            # Create signature hash
            combined = "|".join(content_parts)
            signature = hashlib.md5(combined.encode()).hexdigest()[:16]
            
            return signature
            
        except Exception as e:
            logger.error(f"Content signature creation failed: {e}")
            return "unknown"
    
    def _signatures_similar(self, sig1: str, sig2: str, threshold: float = 0.8) -> bool:
        """Check if two content signatures are similar."""
        if sig1 == sig2:
            return True
        
        # Simple Hamming distance for similarity
        if len(sig1) != len(sig2):
            return False
        
        matches = sum(c1 == c2 for c1, c2 in zip(sig1, sig2))
        similarity = matches / len(sig1)
        
        return similarity >= threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trust scorer statistics."""
        return {
            "sources_tracked": len(self.source_reputation),
            "content_signatures": len(self.content_signatures),
            "trust_weights": self.weights,
            "average_reputation": (
                sum(self.source_reputation.values()) / max(1, len(self.source_reputation))
            ),
            "reputation_history_size": sum(
                len(history) for history in self.reputation_history.values()
            )
        }
    
    async def get_source_trust_report(self, source_id: str) -> Dict[str, Any]:
        """Get detailed trust report for a source."""
        reputation = self.source_reputation.get(source_id, 0.7)
        history = self.reputation_history.get(source_id, [])
        
        recent_history = [
            h for h in history 
            if h["timestamp"] > now_utc() - timedelta(days=7)
        ]
        
        return {
            "source_id": source_id,
            "current_reputation": reputation,
            "history_entries": len(history),
            "recent_updates": len(recent_history),
            "trend": self._calculate_reputation_trend(recent_history),
            "last_update": max(h["timestamp"] for h in history) if history else None
        }
    
    def _calculate_reputation_trend(self, recent_history: List[Dict[str, Any]]) -> str:
        """Calculate reputation trend from recent history."""
        if len(recent_history) < 2:
            return "stable"
        
        recent_scores = [h["new_reputation"] for h in recent_history[-10:]]
        
        if len(recent_scores) >= 2:
            trend_slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
            
            if trend_slope > 0.01:
                return "improving"
            elif trend_slope < -0.01:
                return "declining"
        
        return "stable"