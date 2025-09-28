"""
Unified Logic - Cross-layer synthesis and arbitration for Grace governance kernel.
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from ..utils.datetime_utils import utc_now, iso_format, format_for_filename
import logging
import statistics

from ..core.contracts import (
    UnifiedDecision, ComponentSignal, Experience, 
    generate_decision_id, VerifiedClaims
)


logger = logging.getLogger(__name__)


class UnifiedLogic:
    """
    Merges inputs from all components (MLDL specialists, verification engine, etc.)
    into unified, constitutional decisions through weighted synthesis and arbitration.
    """
    
    def __init__(self, event_bus, memory_core):
        self.event_bus = event_bus
        self.memory_core = memory_core
        self.specialist_weights = self._initialize_specialist_weights()
        self.decision_thresholds = self._initialize_decision_thresholds()
    
    def _initialize_specialist_weights(self) -> Dict[str, float]:
        """Initialize weights for different specialist components."""
        return {
            # Classic ML specialists
            "tabular_classification": 0.9,
            "tabular_regression": 0.9,
            "nlp_text": 0.85,
            "vision_cnn": 0.8,
            "time_series": 0.85,
            "recommenders": 0.7,
            "anomaly_detection": 0.9,
            "clustering": 0.75,
            "dimensionality_reduction": 0.7,
            "graph_learning": 0.8,
            
            # Advanced AI specialists
            "reinforcement_learning": 0.85,
            "bayesian_uncertainty": 0.9,
            "causal_inference": 0.95,
            "optimization_hpo": 0.8,
            "fairness_explainability": 1.0,  # High weight for ethical decisions
            "data_quality_drift": 0.85,
            "privacy_security": 1.0,  # High weight for security decisions
            "automl_planner": 0.8,
            "experimentation": 0.75,
            "meta_ensembler": 0.85,
            "governance_liaison": 1.0,  # Highest weight as it interfaces directly
            
            # Core components
            "verification_engine": 1.0,
            "trust_core": 0.95,
            "constitutional_checker": 1.0
        }
    
    def _initialize_decision_thresholds(self) -> Dict[str, float]:
        """Initialize decision confidence thresholds."""
        return {
            "min_confidence_approve": 0.75,
            "min_confidence_reject": 0.6,
            "min_trust_score": 0.7,
            "consensus_threshold": 0.65,
            "constitutional_compliance_min": 0.8
        }
    
    async def synthesize_decision(self, topic: str, inputs: Dict[str, Any],
                                mldl_consensus_id: Optional[str] = None) -> UnifiedDecision:
        """
        Synthesize inputs from multiple components into a unified decision.
        
        Args:
            topic: The decision topic/subject
            inputs: Dictionary containing all component inputs
            mldl_consensus_id: Optional ID from MLDL consensus system
        """
        synthesis_start = utc_now()
        decision_id = generate_decision_id()
        
        try:
            # Extract component signals
            component_signals = await self._extract_component_signals(inputs)
            
            # Apply specialist weights
            weighted_signals = await self._apply_weights(component_signals)
            
            # Check for conflicts and arbitrate
            arbitrated_signals = await self._arbitrate_conflicts(weighted_signals)
            
            # Calculate unified recommendation
            recommendation = await self._calculate_recommendation(arbitrated_signals)
            
            # Generate rationale
            rationale = await self._generate_rationale(
                topic, arbitrated_signals, recommendation
            )
            
            # Calculate confidence and trust scores
            confidence = await self._calculate_confidence(arbitrated_signals)
            trust_score = await self._calculate_trust_score(arbitrated_signals, inputs)
            
            # Create unified decision
            decision = UnifiedDecision(
                decision_id=decision_id,
                topic=topic,
                inputs={
                    "mldl_consensus_id": mldl_consensus_id,
                    "component_signals": [signal.to_dict() for signal in arbitrated_signals],
                    "original_inputs": inputs
                },
                recommendation=recommendation,
                rationale=rationale,
                confidence=confidence,
                trust_score=trust_score,
                timestamp=utc_now()
            )
            
            # Record experience for meta-learning
            await self._record_synthesis_experience(
                len(component_signals), len(arbitrated_signals), 
                recommendation, confidence,
                (utc_now() - synthesis_start).total_seconds()
            )
            
            logger.info(f"Synthesized decision {decision_id}: {recommendation} (conf: {confidence:.3f})")
            return decision
            
        except Exception as e:
            logger.error(f"Error in decision synthesis: {e}")
            return self._create_error_decision(decision_id, topic, inputs, str(e))
    
    async def _extract_component_signals(self, inputs: Dict[str, Any]) -> List[ComponentSignal]:
        """Extract and standardize signals from component inputs."""
        signals = []
        
        # Process MLDL specialist outputs
        if "mldl_outputs" in inputs:
            mldl_outputs = inputs["mldl_outputs"]
            for specialist_name, output in mldl_outputs.items():
                signal = self._convert_mldl_output_to_signal(specialist_name, output)
                if signal:
                    signals.append(signal)
        
        # Process verification engine output
        if "verification_result" in inputs:
            verification = inputs["verification_result"]
            if isinstance(verification, dict):
                confidence = verification.get("overall_confidence", 0.0)
                status = verification.get("verification_status", "unknown")
                
                # Convert verification status to signal
                signal_value = {
                    "verified": 1.0,
                    "probable": 0.7,
                    "inconclusive": 0.3,
                    "refuted": 0.0
                }.get(status, 0.5)
                
                signals.append(ComponentSignal(
                    component="verification_engine",
                    signal=f"verification_{status}",
                    weight=confidence
                ))
        
        # Process trust scores
        if "trust_scores" in inputs:
            trust_data = inputs["trust_scores"]
            avg_trust = statistics.mean(trust_data.values()) if trust_data else 0.5
            signals.append(ComponentSignal(
                component="trust_core",
                signal="average_trust",
                weight=avg_trust
            ))
        
        # Process constitutional compliance
        if "constitutional_compliance" in inputs:
            compliance = inputs["constitutional_compliance"]
            signals.append(ComponentSignal(
                component="constitutional_checker",
                signal="compliance_score",
                weight=compliance
            ))
        
        return signals
    
    def _convert_mldl_output_to_signal(self, specialist_name: str, output: Dict[str, Any]) -> Optional[ComponentSignal]:
        """Convert MLDL specialist output to standardized signal."""
        if not output:
            return None
        
        # Extract confidence/probability from different output formats
        confidence = 0.5  # Default
        signal_type = "prediction"
        
        if "confidence" in output:
            confidence = output["confidence"]
        elif "probability" in output:
            confidence = output["probability"]
        elif "score" in output:
            confidence = output["score"]
        elif "accuracy" in output:
            confidence = output["accuracy"]
        
        # Determine signal type based on specialist
        if "classification" in specialist_name:
            signal_type = f"classification_{output.get('class', 'unknown')}"
        elif "regression" in specialist_name:
            signal_type = f"regression_{output.get('value', 'predicted')}"
        elif "anomaly" in specialist_name:
            signal_type = "anomaly_detected" if output.get("is_anomaly", False) else "normal"
        elif "fairness" in specialist_name:
            signal_type = "fairness_compliant" if confidence > 0.7 else "fairness_concern"
        elif "security" in specialist_name:
            signal_type = "security_cleared" if confidence > 0.8 else "security_risk"
        
        return ComponentSignal(
            component=specialist_name,
            signal=signal_type,
            weight=min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
        )
    
    async def _apply_weights(self, signals: List[ComponentSignal]) -> List[ComponentSignal]:
        """Apply specialist weights to component signals."""
        weighted_signals = []
        
        for signal in signals:
            specialist_weight = self.specialist_weights.get(signal.component, 0.5)
            adjusted_weight = signal.weight * specialist_weight
            
            weighted_signals.append(ComponentSignal(
                component=signal.component,
                signal=signal.signal,
                weight=adjusted_weight
            ))
        
        return weighted_signals
    
    async def _arbitrate_conflicts(self, signals: List[ComponentSignal]) -> List[ComponentSignal]:
        """Resolve conflicts between component signals."""
        # Group signals by component type
        component_groups = {}
        for signal in signals:
            if signal.component not in component_groups:
                component_groups[signal.component] = []
            component_groups[signal.component].append(signal)
        
        arbitrated_signals = []
        
        # For each component, resolve internal conflicts
        for component, component_signals in component_groups.items():
            if len(component_signals) == 1:
                arbitrated_signals.extend(component_signals)
            else:
                # Multiple signals from same component - use weighted average
                total_weight = sum(s.weight for s in component_signals)
                avg_weight = total_weight / len(component_signals)
                
                # Create consolidated signal
                consolidated_signal = ComponentSignal(
                    component=component,
                    signal="consolidated",
                    weight=avg_weight
                )
                arbitrated_signals.append(consolidated_signal)
        
        # Check for cross-component conflicts
        high_confidence_signals = [s for s in arbitrated_signals if s.weight > 0.8]
        low_confidence_signals = [s for s in arbitrated_signals if s.weight < 0.3]
        
        # If we have both high and low confidence signals, investigate
        if high_confidence_signals and low_confidence_signals:
            logger.warning(f"Detected conflicting signals: "
                         f"{len(high_confidence_signals)} high confidence vs "
                         f"{len(low_confidence_signals)} low confidence")
            
            # Weight towards high-confidence signals
            for signal in arbitrated_signals:
                if signal.weight > 0.8:
                    signal.weight *= 1.1  # Boost high confidence
                elif signal.weight < 0.3:
                    signal.weight *= 0.8  # Reduce low confidence
        
        return arbitrated_signals
    
    async def _calculate_recommendation(self, signals: List[ComponentSignal]) -> str:
        """Calculate final recommendation based on arbitrated signals."""
        if not signals:
            return "review"
        
        # Calculate weighted average confidence
        total_weighted_confidence = sum(s.weight for s in signals)
        avg_confidence = total_weighted_confidence / len(signals)
        
        # Check for critical component vetoes
        critical_components = ["privacy_security", "fairness_explainability", "constitutional_checker"]
        for signal in signals:
            if (signal.component in critical_components and 
                signal.weight < self.decision_thresholds["min_trust_score"]):
                return "reject"
        
        # Make recommendation based on confidence thresholds
        if avg_confidence >= self.decision_thresholds["min_confidence_approve"]:
            return "approve"
        elif avg_confidence >= self.decision_thresholds["min_confidence_reject"]:
            return "review"
        else:
            return "reject"
    
    async def _generate_rationale(self, topic: str, signals: List[ComponentSignal],
                                recommendation: str) -> str:
        """Generate human-readable rationale for the decision."""
        rationale_parts = [
            f"Decision on '{topic}': {recommendation.upper()}"
        ]
        
        # Summarize signal contributions
        high_confidence = [s for s in signals if s.weight > 0.7]
        medium_confidence = [s for s in signals if 0.4 <= s.weight <= 0.7]
        low_confidence = [s for s in signals if s.weight < 0.4]
        
        if high_confidence:
            components = [s.component for s in high_confidence]
            rationale_parts.append(
                f"Strong support from: {', '.join(components[:3])}{'...' if len(components) > 3 else ''}"
            )
        
        if low_confidence:
            components = [s.component for s in low_confidence]
            rationale_parts.append(
                f"Concerns raised by: {', '.join(components[:3])}{'...' if len(components) > 3 else ''}"
            )
        
        # Add specific reasoning for recommendation
        if recommendation == "approve":
            rationale_parts.append("All critical compliance checks passed with sufficient confidence.")
        elif recommendation == "reject":
            rationale_parts.append("Critical compliance failures or insufficient confidence detected.")
        else:  # review
            rationale_parts.append("Mixed signals require human review for final determination.")
        
        return " ".join(rationale_parts)
    
    async def _calculate_confidence(self, signals: List[ComponentSignal]) -> float:
        """Calculate overall confidence score."""
        if not signals:
            return 0.0
        
        # Weighted average with bias towards critical components
        weights = []
        confidences = []
        
        for signal in signals:
            component_importance = self.specialist_weights.get(signal.component, 0.5)
            weights.append(component_importance)
            confidences.append(signal.weight)
        
        if not weights:
            return 0.0
        
        weighted_confidence = sum(w * c for w, c in zip(weights, confidences)) / sum(weights)
        return min(max(weighted_confidence, 0.0), 1.0)
    
    async def _calculate_trust_score(self, signals: List[ComponentSignal], 
                                   inputs: Dict[str, Any]) -> float:
        """Calculate trust score based on source reliability and consistency."""
        trust_factors = []
        
        # Component reliability
        for signal in signals:
            component_trust = self.specialist_weights.get(signal.component, 0.5)
            trust_factors.append(component_trust * signal.weight)
        
        # Input source credibility
        if "source_credibility" in inputs:
            trust_factors.append(inputs["source_credibility"])
        
        # Historical consistency
        if "historical_consistency" in inputs:
            trust_factors.append(inputs["historical_consistency"])
        
        if not trust_factors:
            return 0.5
        
        return sum(trust_factors) / len(trust_factors)
    
    def _create_error_decision(self, decision_id: str, topic: str, 
                              inputs: Dict[str, Any], error_msg: str) -> UnifiedDecision:
        """Create an error decision when synthesis fails."""
        return UnifiedDecision(
            decision_id=decision_id,
            topic=topic,
            inputs=inputs,
            recommendation="review",
            rationale=f"Synthesis error: {error_msg}. Manual review required.",
            confidence=0.0,
            trust_score=0.0,
            timestamp=utc_now()
        )
    
    async def _record_synthesis_experience(self, input_count: int, signal_count: int,
                                         recommendation: str, confidence: float,
                                         processing_time: float):
        """Record synthesis experience for meta-learning."""
        # Determine success score based on confidence and decision clarity
        success_score = confidence
        if recommendation == "review":
            success_score *= 0.7  # Penalize indecisive outcomes
        
        experience = Experience(
            type="CONSENSUS_QUALITY",
            component_id="unified_logic",
            context={
                "inputs": input_count,
                "conflicts": max(0, input_count - signal_count),  # Simplified conflict count
                "processing_time": processing_time
            },
            outcome={
                "final_decision": recommendation,
                "delta_vs_thresholds": confidence - self.decision_thresholds["min_confidence_approve"]
            },
            success_score=success_score,
            timestamp=utc_now()
        )
        
        self.memory_core.store_experience(experience)
        
        # Emit learning event
        await self.event_bus.publish("LEARNING_EXPERIENCE", experience.to_dict())
    
    async def update_weights(self, weight_updates: Dict[str, float]):
        """Update specialist weights based on learning feedback."""
        for component, new_weight in weight_updates.items():
            if component in self.specialist_weights:
                old_weight = self.specialist_weights[component]
                self.specialist_weights[component] = max(0.1, min(1.0, new_weight))
                logger.info(f"Updated {component} weight: {old_weight:.3f} -> {new_weight:.3f}")
    
    async def update_thresholds(self, threshold_updates: Dict[str, float]):
        """Update decision thresholds based on learning feedback."""
        for threshold_name, new_value in threshold_updates.items():
            if threshold_name in self.decision_thresholds:
                old_value = self.decision_thresholds[threshold_name]
                self.decision_thresholds[threshold_name] = max(0.0, min(1.0, new_value))
                logger.info(f"Updated {threshold_name}: {old_value:.3f} -> {new_value:.3f}")
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current specialist weights."""
        return self.specialist_weights.copy()
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current decision thresholds."""
        return self.decision_thresholds.copy()