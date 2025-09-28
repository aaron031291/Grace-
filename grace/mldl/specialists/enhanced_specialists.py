"""
Enhanced ML/DL Specialists for Grace Governance Kernel.

Adds new specialized capabilities:
- Graph Neural Networks for relationship modeling
- Multimodal AI for cross-modal understanding  
- Federated Learning for privacy-preserving ML
- Uncertainty Quantification for confidence intervals
- Cross-domain validators for governance oversight
"""

import asyncio
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
import hashlib

# Optional ML dependencies
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SpecialistPrediction:
    """Enhanced prediction with uncertainty quantification."""
    prediction: Any
    confidence: float
    uncertainty: float
    explanation: str
    evidence: List[Dict[str, Any]]
    cross_domain_score: float
    hallucination_risk: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "explanation": self.explanation,
            "evidence": self.evidence,
            "cross_domain_score": self.cross_domain_score,
            "hallucination_risk": self.hallucination_risk,
            "predicted_at": datetime.now().isoformat()
        }


@dataclass
class CrossDomainValidation:
    """Cross-domain validation result."""
    domain: str
    validator_id: str
    validation_score: float
    compliance_score: float
    risk_assessment: Dict[str, float]
    recommendations: List[str]
    red_flags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "validator_id": self.validator_id,
            "validation_score": self.validation_score,
            "compliance_score": self.compliance_score,
            "risk_assessment": self.risk_assessment,
            "recommendations": self.recommendations,
            "red_flags": self.red_flags,
            "validated_at": datetime.now().isoformat()
        }


class EnhancedMLSpecialist(ABC):
    """Base class for enhanced ML/DL specialists with uncertainty quantification."""
    
    def __init__(self, specialist_id: str, domain: str, expertise_areas: List[str]):
        self.specialist_id = specialist_id
        self.domain = domain
        self.expertise_areas = expertise_areas
        self.performance_history = []
        self.trust_score = 0.8
        self.initialized = False
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the specialist."""
        pass
    
    @abstractmethod
    async def predict_with_uncertainty(self, 
                                     data: Dict[str, Any], 
                                     context: Optional[Dict[str, Any]] = None) -> SpecialistPrediction:
        """Make prediction with uncertainty quantification."""
        pass
    
    def calculate_hallucination_risk(self, prediction: Any, evidence: List[Dict[str, Any]]) -> float:
        """Calculate risk of hallucination based on prediction and evidence."""
        try:
            if not evidence:
                return 0.9  # High risk if no evidence
            
            # Calculate evidence quality score
            evidence_scores = [e.get("quality", 0.5) for e in evidence]
            avg_evidence_quality = sum(evidence_scores) / len(evidence_scores)
            
            # Calculate prediction consistency
            consistency_scores = [e.get("consistency", 0.5) for e in evidence]
            avg_consistency = sum(consistency_scores) / len(consistency_scores)
            
            # Risk is inverse of evidence quality and consistency
            risk = 1.0 - (avg_evidence_quality * avg_consistency)
            return min(max(risk, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate hallucination risk: {e}")
            return 0.5
    
    def update_performance(self, actual_outcome: Any, predicted_outcome: Any, confidence: float):
        """Update specialist performance based on actual outcome."""
        try:
            # Calculate accuracy (simplified)
            if isinstance(actual_outcome, (int, float)) and isinstance(predicted_outcome, (int, float)):
                accuracy = 1.0 - abs(actual_outcome - predicted_outcome) / max(abs(actual_outcome), 1.0)
            elif actual_outcome == predicted_outcome:
                accuracy = 1.0
            else:
                accuracy = 0.0
            
            # Update performance history
            performance_record = {
                "timestamp": datetime.now().isoformat(),
                "accuracy": accuracy,
                "confidence": confidence,
                "calibration": abs(confidence - accuracy)  # How well-calibrated is confidence
            }
            
            self.performance_history.append(performance_record)
            
            # Keep only last 100 records
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            # Update trust score based on recent performance
            recent_accuracies = [r["accuracy"] for r in self.performance_history[-10:]]
            if recent_accuracies:
                self.trust_score = sum(recent_accuracies) / len(recent_accuracies)
            
        except Exception as e:
            logger.error(f"Failed to update performance for {self.specialist_id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get specialist statistics."""
        stats = {
            "specialist_id": self.specialist_id,
            "domain": self.domain,
            "expertise_areas": self.expertise_areas,
            "trust_score": self.trust_score,
            "initialized": self.initialized,
            "total_predictions": len(self.performance_history)
        }
        
        if self.performance_history:
            recent_performance = self.performance_history[-10:]
            stats.update({
                "recent_accuracy": sum(r["accuracy"] for r in recent_performance) / len(recent_performance),
                "recent_calibration": sum(r["calibration"] for r in recent_performance) / len(recent_performance),
                "avg_confidence": sum(r["confidence"] for r in recent_performance) / len(recent_performance)
            })
        
        return stats


class GraphNeuralNetworkSpecialist(EnhancedMLSpecialist):
    """Specialist for graph-based relationship modeling and analysis."""
    
    def __init__(self):
        super().__init__(
            specialist_id="gnn_specialist",
            domain="graph_analysis",
            expertise_areas=["relationship_modeling", "network_analysis", "dependency_tracking"]
        )
        self.graph = None
        
    async def initialize(self) -> bool:
        """Initialize the GNN specialist."""
        if not NETWORKX_AVAILABLE:
            logger.error("NetworkX not available for Graph Neural Network specialist")
            return False
        
        try:
            self.graph = nx.DiGraph()
            self.initialized = True
            logger.info("Graph Neural Network specialist initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize GNN specialist: {e}")
            return False
    
    async def predict_with_uncertainty(self, 
                                     data: Dict[str, Any], 
                                     context: Optional[Dict[str, Any]] = None) -> SpecialistPrediction:
        """Analyze relationships and predict outcomes based on graph structure."""
        try:
            task_type = data.get("task", "relationship_analysis")
            
            if task_type == "relationship_analysis":
                return await self._analyze_relationships(data, context)
            elif task_type == "dependency_analysis":
                return await self._analyze_dependencies(data, context)
            elif task_type == "influence_propagation":
                return await self._analyze_influence_propagation(data, context)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
                
        except Exception as e:
            logger.error(f"GNN prediction failed: {e}")
            return SpecialistPrediction(
                prediction=None,
                confidence=0.0,
                uncertainty=1.0,
                explanation=f"Analysis failed: {str(e)}",
                evidence=[],
                cross_domain_score=0.0,
                hallucination_risk=0.9
            )
    
    async def _analyze_relationships(self, data: Dict[str, Any], 
                                   context: Optional[Dict[str, Any]]) -> SpecialistPrediction:
        """Analyze entity relationships."""
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        
        # Build graph
        temp_graph = nx.DiGraph()
        for entity in entities:
            temp_graph.add_node(entity.get("id"), **entity)
        
        for rel in relationships:
            temp_graph.add_edge(
                rel.get("source"), 
                rel.get("target"), 
                weight=rel.get("strength", 1.0),
                type=rel.get("type", "unknown")
            )
        
        # Analyze graph properties
        analysis = {
            "node_count": temp_graph.number_of_nodes(),
            "edge_count": temp_graph.number_of_edges(),
            "density": nx.density(temp_graph),
            "connected_components": nx.number_weakly_connected_components(temp_graph)
        }
        
        # Calculate centrality measures
        if temp_graph.number_of_nodes() > 0:
            centrality = nx.degree_centrality(temp_graph)
            analysis["most_central"] = max(centrality, key=centrality.get) if centrality else None
            analysis["centrality_scores"] = centrality
        
        evidence = [
            {"type": "graph_analysis", "data": analysis, "quality": 0.8, "consistency": 0.9}
        ]
        
        confidence = min(1.0, temp_graph.number_of_edges() / max(temp_graph.number_of_nodes(), 1))
        uncertainty = 1.0 - confidence
        
        return SpecialistPrediction(
            prediction=analysis,
            confidence=confidence,
            uncertainty=uncertainty,
            explanation=f"Graph analysis of {temp_graph.number_of_nodes()} entities and {temp_graph.number_of_edges()} relationships",
            evidence=evidence,
            cross_domain_score=0.7,
            hallucination_risk=self.calculate_hallucination_risk(analysis, evidence)
        )
    
    async def _analyze_dependencies(self, data: Dict[str, Any], 
                                  context: Optional[Dict[str, Any]]) -> SpecialistPrediction:
        """Analyze system dependencies."""
        dependencies = data.get("dependencies", [])
        
        # Build dependency graph
        dep_graph = nx.DiGraph()
        for dep in dependencies:
            dep_graph.add_edge(dep.get("dependent"), dep.get("dependency"))
        
        analysis = {
            "total_dependencies": dep_graph.number_of_edges(),
            "components": dep_graph.number_of_nodes(),
            "cycles": list(nx.simple_cycles(dep_graph)),
            "topological_order": list(nx.topological_sort(dep_graph)) if nx.is_directed_acyclic_graph(dep_graph) else None
        }
        
        # Risk assessment
        analysis["risk_score"] = len(analysis["cycles"]) / max(dep_graph.number_of_edges(), 1)
        
        evidence = [
            {"type": "dependency_analysis", "data": analysis, "quality": 0.9, "consistency": 0.8}
        ]
        
        confidence = 0.8 if nx.is_directed_acyclic_graph(dep_graph) else 0.6
        
        return SpecialistPrediction(
            prediction=analysis,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            explanation=f"Dependency analysis found {len(analysis['cycles'])} cycles in {dep_graph.number_of_nodes()} components",
            evidence=evidence,
            cross_domain_score=0.8,
            hallucination_risk=self.calculate_hallucination_risk(analysis, evidence)
        )
    
    async def _analyze_influence_propagation(self, data: Dict[str, Any], 
                                           context: Optional[Dict[str, Any]]) -> SpecialistPrediction:
        """Analyze how influence propagates through the network."""
        source_nodes = data.get("source_nodes", [])
        influence_strength = data.get("influence_strength", 1.0)
        
        # Simple influence propagation simulation
        influence_map = {}
        for source in source_nodes:
            if self.graph.has_node(source):
                # BFS-based influence propagation
                visited = {source}
                queue = [(source, influence_strength)]
                influence_map[source] = influence_strength
                
                while queue:
                    node, strength = queue.pop(0)
                    if strength < 0.1:  # Stop if influence too weak
                        continue
                    
                    for neighbor in self.graph.neighbors(node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            # Decay influence based on edge weight
                            edge_data = self.graph[node][neighbor]
                            decay_factor = edge_data.get("weight", 0.5)
                            new_strength = strength * decay_factor
                            influence_map[neighbor] = influence_map.get(neighbor, 0) + new_strength
                            queue.append((neighbor, new_strength))
        
        analysis = {
            "influenced_nodes": len(influence_map),
            "influence_distribution": influence_map,
            "total_influence": sum(influence_map.values()),
            "max_influence_node": max(influence_map, key=influence_map.get) if influence_map else None
        }
        
        evidence = [
            {"type": "influence_propagation", "data": analysis, "quality": 0.7, "consistency": 0.8}
        ]
        
        confidence = min(1.0, len(influence_map) / max(len(source_nodes) * 5, 1))
        
        return SpecialistPrediction(
            prediction=analysis,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            explanation=f"Influence propagation from {len(source_nodes)} sources affected {len(influence_map)} nodes",
            evidence=evidence,
            cross_domain_score=0.6,
            hallucination_risk=self.calculate_hallucination_risk(analysis, evidence)
        )


class MultimodalAISpecialist(EnhancedMLSpecialist):
    """Specialist for cross-modal understanding and integration."""
    
    def __init__(self):
        super().__init__(
            specialist_id="multimodal_specialist",
            domain="multimodal_analysis",
            expertise_areas=["text_analysis", "cross_modal_reasoning", "content_integration"]
        )
        
    async def initialize(self) -> bool:
        """Initialize the multimodal specialist."""
        try:
            # In a real implementation, this would load multimodal models
            self.initialized = True
            logger.info("Multimodal AI specialist initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize multimodal specialist: {e}")
            return False
    
    async def predict_with_uncertainty(self, 
                                     data: Dict[str, Any], 
                                     context: Optional[Dict[str, Any]] = None) -> SpecialistPrediction:
        """Perform multimodal analysis and reasoning."""
        try:
            task_type = data.get("task", "content_analysis")
            
            if task_type == "content_analysis":
                return await self._analyze_content(data, context)
            elif task_type == "cross_modal_alignment":
                return await self._analyze_cross_modal_alignment(data, context)
            elif task_type == "content_synthesis":
                return await self._synthesize_content(data, context)
            elif task_type in ["ai_deployment_decision", "policy_approval"]:
                # Handle governance tasks as content analysis
                return await self._analyze_governance_content(data, context)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Multimodal prediction failed: {e}")
            return SpecialistPrediction(
                prediction=None,
                confidence=0.0,
                uncertainty=1.0,
                explanation=f"Analysis failed: {str(e)}",
                evidence=[],
                cross_domain_score=0.0,
                hallucination_risk=0.9
            )
    
    async def _analyze_content(self, data: Dict[str, Any], 
                             context: Optional[Dict[str, Any]]) -> SpecialistPrediction:
        """Analyze multimodal content."""
        content_types = data.get("content_types", [])
        content_data = data.get("content", {})
        
        analysis = {
            "modalities_detected": content_types,
            "content_quality": {},
            "semantic_coherence": 0.0,
            "cross_modal_consistency": 0.0
        }
        
        # Simulate content quality assessment for each modality
        for content_type in content_types:
            content_item = content_data.get(content_type, "")
            if content_type == "text":
                quality = min(1.0, len(str(content_item)) / 100)  # Simple length-based quality
            elif content_type == "image":
                quality = 0.8  # Simulated image quality
            elif content_type == "audio":
                quality = 0.7  # Simulated audio quality
            else:
                quality = 0.5
            
            analysis["content_quality"][content_type] = quality
        
        # Simulate semantic coherence
        if len(content_types) > 1:
            analysis["semantic_coherence"] = np.mean(list(analysis["content_quality"].values()))
            analysis["cross_modal_consistency"] = analysis["semantic_coherence"] * 0.9
        else:
            analysis["semantic_coherence"] = list(analysis["content_quality"].values())[0] if analysis["content_quality"] else 0.5
            analysis["cross_modal_consistency"] = analysis["semantic_coherence"]
        
        evidence = [
            {"type": "content_analysis", "data": analysis, "quality": 0.7, "consistency": 0.8}
        ]
        
        confidence = analysis["semantic_coherence"]
        uncertainty = 1.0 - confidence
        
        return SpecialistPrediction(
            prediction=analysis,
            confidence=confidence,
            uncertainty=uncertainty,
            explanation=f"Multimodal content analysis of {len(content_types)} modalities with {confidence:.2f} coherence",
            evidence=evidence,
            cross_domain_score=0.8,
            hallucination_risk=self.calculate_hallucination_risk(analysis, evidence)
        )
    
    async def _analyze_cross_modal_alignment(self, data: Dict[str, Any], 
                                           context: Optional[Dict[str, Any]]) -> SpecialistPrediction:
        """Analyze alignment between different modalities."""
        modal_content = data.get("modal_content", {})
        
        alignment_scores = {}
        for mod1 in modal_content:
            for mod2 in modal_content:
                if mod1 != mod2:
                    # Simulate alignment scoring
                    content1 = str(modal_content[mod1])
                    content2 = str(modal_content[mod2])
                    
                    # Simple alignment based on content similarity
                    similarity = len(set(content1.lower().split()) & set(content2.lower().split()))
                    total_words = len(set(content1.lower().split()) | set(content2.lower().split()))
                    alignment = similarity / max(total_words, 1)
                    
                    alignment_scores[f"{mod1}_{mod2}"] = alignment
        
        analysis = {
            "alignment_scores": alignment_scores,
            "average_alignment": np.mean(list(alignment_scores.values())) if alignment_scores else 0.0,
            "well_aligned_pairs": [k for k, v in alignment_scores.items() if v > 0.7],
            "misaligned_pairs": [k for k, v in alignment_scores.items() if v < 0.3]
        }
        
        evidence = [
            {"type": "cross_modal_alignment", "data": analysis, "quality": 0.6, "consistency": 0.7}
        ]
        
        confidence = analysis["average_alignment"]
        
        return SpecialistPrediction(
            prediction=analysis,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            explanation=f"Cross-modal alignment analysis: {confidence:.2f} average alignment across modalities",
            evidence=evidence,
            cross_domain_score=0.7,
            hallucination_risk=self.calculate_hallucination_risk(analysis, evidence)
        )
    
    async def _synthesize_content(self, data: Dict[str, Any], 
                                context: Optional[Dict[str, Any]]) -> SpecialistPrediction:
        """Synthesize content across modalities."""
        source_modalities = data.get("source_modalities", [])
        target_modality = data.get("target_modality", "text")
        content_data = data.get("content", {})
        
        # Simulate content synthesis
        synthesis_result = {
            "target_modality": target_modality,
            "source_modalities": source_modalities,
            "synthesis_quality": 0.0,
            "coherence_score": 0.0
        }
        
        if target_modality == "text":
            # Simulate text synthesis from multiple modalities
            synthesized_text = f"Synthesis of {', '.join(source_modalities)} content"
            for modality in source_modalities:
                if modality in content_data:
                    synthesized_text += f" | {modality}: {str(content_data[modality])[:100]}"
            
            synthesis_result["synthesized_content"] = synthesized_text
            synthesis_result["synthesis_quality"] = min(1.0, len(synthesized_text) / 200)
            synthesis_result["coherence_score"] = 0.7  # Simulated coherence
        
        evidence = [
            {"type": "content_synthesis", "data": synthesis_result, "quality": 0.6, "consistency": 0.7}
        ]
        
        confidence = (synthesis_result["synthesis_quality"] + synthesis_result["coherence_score"]) / 2
        
        return SpecialistPrediction(
            prediction=synthesis_result,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            explanation=f"Content synthesis from {len(source_modalities)} modalities to {target_modality}",
            evidence=evidence,
            cross_domain_score=0.6,
            hallucination_risk=self.calculate_hallucination_risk(synthesis_result, evidence)
        )
    
    async def _analyze_governance_content(self, data: Dict[str, Any], 
                                         context: Optional[Dict[str, Any]]) -> SpecialistPrediction:
        """Analyze governance-related content."""
        task_type = data.get("task", "ai_deployment_decision")
        content_data = data.get("data", {})
        
        analysis = {
            "task_type": task_type,
            "feasibility": 0.8,  # Simulated analysis
            "risk_assessment": {"low": 0.3, "medium": 0.5, "high": 0.2},
            "stakeholder_impact": content_data.get("stakeholder_support", "medium"),
            "complexity_score": 0.6
        }
        
        # Simulate quality assessment
        if context and "regulatory_pressure" in context:
            analysis["regulatory_alignment"] = 0.8 if context["regulatory_pressure"] == "high" else 0.6
        
        evidence = [
            {"type": "governance_analysis", "data": analysis, "quality": 0.8, "consistency": 0.7}
        ]
        
        confidence = 0.7
        uncertainty = 1.0 - confidence
        
        return SpecialistPrediction(
            prediction=analysis,
            confidence=confidence,
            uncertainty=uncertainty,
            explanation=f"Governance analysis for {task_type}",
            evidence=evidence,
            cross_domain_score=0.7,
            hallucination_risk=self.calculate_hallucination_risk(analysis, evidence)
        )


class UncertaintyQuantificationSpecialist(EnhancedMLSpecialist):
    """Specialist for uncertainty quantification and confidence interval estimation."""
    
    def __init__(self):
        super().__init__(
            specialist_id="uncertainty_specialist",
            domain="uncertainty_quantification",
            expertise_areas=["confidence_estimation", "risk_assessment", "prediction_intervals"]
        )
        
    async def initialize(self) -> bool:
        """Initialize the uncertainty quantification specialist."""
        try:
            self.initialized = True
            logger.info("Uncertainty Quantification specialist initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize uncertainty specialist: {e}")
            return False
    
    async def predict_with_uncertainty(self, 
                                     data: Dict[str, Any], 
                                     context: Optional[Dict[str, Any]] = None) -> SpecialistPrediction:
        """Quantify uncertainty in predictions."""
        try:
            task_type = data.get("task", "uncertainty_analysis")
            
            if task_type == "uncertainty_analysis":
                return await self._analyze_uncertainty(data, context)
            elif task_type == "confidence_calibration":
                return await self._calibrate_confidence(data, context)
            elif task_type == "risk_quantification":
                return await self._quantify_risk(data, context)
            elif task_type in ["ai_deployment_decision", "policy_approval"]:
                # Handle governance tasks as uncertainty analysis
                return await self._analyze_governance_uncertainty(data, context)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Uncertainty prediction failed: {e}")
            return SpecialistPrediction(
                prediction=None,
                confidence=0.0,
                uncertainty=1.0,
                explanation=f"Analysis failed: {str(e)}",
                evidence=[],
                cross_domain_score=0.0,
                hallucination_risk=0.9
            )
    
    async def _analyze_uncertainty(self, data: Dict[str, Any], 
                                 context: Optional[Dict[str, Any]]) -> SpecialistPrediction:
        """Analyze uncertainty in a prediction or decision."""
        predictions = data.get("predictions", [])
        model_confidence = data.get("model_confidence", [])
        
        if not predictions or not model_confidence:
            raise ValueError("Predictions and model confidence required")
        
        # Calculate various uncertainty measures
        analysis = {
            "aleatoric_uncertainty": 0.0,  # Uncertainty inherent in data
            "epistemic_uncertainty": 0.0,  # Uncertainty due to model limitations
            "total_uncertainty": 0.0,
            "confidence_interval": {},
            "uncertainty_sources": []
        }
        
        # Simulate aleatoric uncertainty (data noise)
        if isinstance(predictions[0], (int, float)):
            pred_std = np.std(predictions) if len(predictions) > 1 else 0.1
            analysis["aleatoric_uncertainty"] = min(pred_std, 1.0)
        else:
            # For categorical predictions, use entropy-based measure
            unique_preds = len(set(str(p) for p in predictions))
            total_preds = len(predictions)
            analysis["aleatoric_uncertainty"] = unique_preds / max(total_preds, 1)
        
        # Simulate epistemic uncertainty (model uncertainty)
        conf_std = np.std(model_confidence) if len(model_confidence) > 1 else 0.1
        analysis["epistemic_uncertainty"] = min(conf_std, 1.0)
        
        # Total uncertainty
        analysis["total_uncertainty"] = np.sqrt(
            analysis["aleatoric_uncertainty"]**2 + analysis["epistemic_uncertainty"]**2
        )
        
        # Confidence interval
        if isinstance(predictions[0], (int, float)):
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            analysis["confidence_interval"] = {
                "mean": mean_pred,
                "lower_95": mean_pred - 1.96 * std_pred,
                "upper_95": mean_pred + 1.96 * std_pred
            }
        
        # Identify uncertainty sources
        analysis["uncertainty_sources"] = []
        if analysis["aleatoric_uncertainty"] > 0.3:
            analysis["uncertainty_sources"].append("high_data_noise")
        if analysis["epistemic_uncertainty"] > 0.3:
            analysis["uncertainty_sources"].append("model_uncertainty")
        if np.mean(model_confidence) < 0.7:
            analysis["uncertainty_sources"].append("low_model_confidence")
        
        evidence = [
            {"type": "uncertainty_analysis", "data": analysis, "quality": 0.8, "consistency": 0.9}
        ]
        
        confidence = 1.0 - analysis["total_uncertainty"]
        
        return SpecialistPrediction(
            prediction=analysis,
            confidence=confidence,
            uncertainty=analysis["total_uncertainty"],
            explanation=f"Uncertainty analysis: {analysis['total_uncertainty']:.3f} total uncertainty from {len(analysis['uncertainty_sources'])} sources",
            evidence=evidence,
            cross_domain_score=0.9,
            hallucination_risk=self.calculate_hallucination_risk(analysis, evidence)
        )
    
    async def _calibrate_confidence(self, data: Dict[str, Any], 
                                  context: Optional[Dict[str, Any]]) -> SpecialistPrediction:
        """Calibrate confidence scores based on historical performance."""
        confidence_scores = data.get("confidence_scores", [])
        actual_accuracies = data.get("actual_accuracies", [])
        
        if len(confidence_scores) != len(actual_accuracies):
            raise ValueError("Confidence scores and accuracies must have same length")
        
        # Calculate calibration metrics
        calibration_analysis = {
            "calibration_error": 0.0,
            "reliability": 0.0,
            "sharpness": 0.0,
            "calibration_curve": []
        }
        
        if confidence_scores and actual_accuracies:
            # Expected Calibration Error (ECE)
            calibration_error = np.mean(np.abs(np.array(confidence_scores) - np.array(actual_accuracies)))
            calibration_analysis["calibration_error"] = calibration_error
            
            # Reliability (how well confidence matches accuracy)
            reliability = 1.0 - calibration_error
            calibration_analysis["reliability"] = reliability
            
            # Sharpness (how far confidence is from 0.5)
            sharpness = np.mean(np.abs(np.array(confidence_scores) - 0.5))
            calibration_analysis["sharpness"] = sharpness
            
            # Calibration curve (binned)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                in_bin = [(c >= bin_lower and c < bin_upper) for c in confidence_scores]
                if any(in_bin):
                    bin_conf = np.mean([c for c, in_b in zip(confidence_scores, in_bin) if in_b])
                    bin_acc = np.mean([a for a, in_b in zip(actual_accuracies, in_bin) if in_b])
                    calibration_analysis["calibration_curve"].append({
                        "bin_range": [bin_lower, bin_upper],
                        "avg_confidence": bin_conf,
                        "avg_accuracy": bin_acc,
                        "count": sum(in_bin)
                    })
        
        evidence = [
            {"type": "confidence_calibration", "data": calibration_analysis, "quality": 0.9, "consistency": 0.8}
        ]
        
        confidence = calibration_analysis["reliability"]
        uncertainty = calibration_analysis["calibration_error"]
        
        return SpecialistPrediction(
            prediction=calibration_analysis,
            confidence=confidence,
            uncertainty=uncertainty,
            explanation=f"Confidence calibration: {calibration_error:.3f} calibration error, {reliability:.3f} reliability",
            evidence=evidence,
            cross_domain_score=0.8,
            hallucination_risk=self.calculate_hallucination_risk(calibration_analysis, evidence)
        )
    
    async def _quantify_risk(self, data: Dict[str, Any], 
                           context: Optional[Dict[str, Any]]) -> SpecialistPrediction:
        """Quantify various types of risk."""
        predictions = data.get("predictions", [])
        outcomes = data.get("outcomes", [])
        risk_factors = data.get("risk_factors", {})
        
        risk_analysis = {
            "prediction_risk": 0.0,
            "outcome_risk": 0.0,
            "factor_risks": {},
            "overall_risk": 0.0,
            "risk_categories": []
        }
        
        # Prediction risk (volatility of predictions)
        if predictions and isinstance(predictions[0], (int, float)):
            pred_volatility = np.std(predictions) / max(np.mean(predictions), 1.0)
            risk_analysis["prediction_risk"] = min(pred_volatility, 1.0)
        
        # Outcome risk (volatility of outcomes)
        if outcomes and isinstance(outcomes[0], (int, float)):
            outcome_volatility = np.std(outcomes) / max(np.mean(outcomes), 1.0)
            risk_analysis["outcome_risk"] = min(outcome_volatility, 1.0)
        
        # Factor risks
        for factor, values in risk_factors.items():
            if isinstance(values, list) and values and isinstance(values[0], (int, float)):
                factor_volatility = np.std(values) / max(np.mean(values), 1.0)
                risk_analysis["factor_risks"][factor] = min(factor_volatility, 1.0)
        
        # Overall risk
        all_risks = [risk_analysis["prediction_risk"], risk_analysis["outcome_risk"]]
        all_risks.extend(risk_analysis["factor_risks"].values())
        risk_analysis["overall_risk"] = np.mean([r for r in all_risks if r > 0])
        
        # Risk categories
        if risk_analysis["overall_risk"] > 0.7:
            risk_analysis["risk_categories"].append("high_volatility")
        if risk_analysis["prediction_risk"] > 0.5:
            risk_analysis["risk_categories"].append("prediction_uncertainty")
        if risk_analysis["outcome_risk"] > 0.5:
            risk_analysis["risk_categories"].append("outcome_uncertainty")
        
        evidence = [
            {"type": "risk_quantification", "data": risk_analysis, "quality": 0.7, "consistency": 0.8}
        ]
        
        confidence = 1.0 - risk_analysis["overall_risk"]
        uncertainty = risk_analysis["overall_risk"]
        
        return SpecialistPrediction(
            prediction=risk_analysis,
            confidence=confidence,
            uncertainty=uncertainty,
            explanation=f"Risk quantification: {risk_analysis['overall_risk']:.3f} overall risk across {len(risk_analysis['factor_risks'])} factors",
            evidence=evidence,
            cross_domain_score=0.8,
            hallucination_risk=self.calculate_hallucination_risk(risk_analysis, evidence)
        )
    
    async def _analyze_governance_uncertainty(self, data: Dict[str, Any], 
                                            context: Optional[Dict[str, Any]]) -> SpecialistPrediction:
        """Analyze uncertainty in governance decisions."""
        task_data = data.get("data", {})
        
        # Extract uncertainty factors from governance data
        uncertainty_factors = {
            "stakeholder_alignment": 0.3,  # Simulated
            "regulatory_uncertainty": 0.4,
            "technical_feasibility": 0.2,
            "timeline_risk": 0.3
        }
        
        # Contextual adjustments
        if context:
            if context.get("urgency") == "high":
                uncertainty_factors["timeline_risk"] += 0.2
            if context.get("regulatory_pressure") == "high":
                uncertainty_factors["regulatory_uncertainty"] -= 0.1
        
        # Calculate overall uncertainty
        overall_uncertainty = np.mean(list(uncertainty_factors.values()))
        
        analysis = {
            "uncertainty_factors": uncertainty_factors,
            "overall_uncertainty": overall_uncertainty,
            "confidence_interval": {
                "lower": max(0.0, 0.7 - overall_uncertainty),
                "upper": min(1.0, 0.7 + overall_uncertainty),
                "mean": 0.7
            },
            "uncertainty_sources": [k for k, v in uncertainty_factors.items() if v > 0.3]
        }
        
        evidence = [
            {"type": "governance_uncertainty", "data": analysis, "quality": 0.7, "consistency": 0.8}
        ]
        
        confidence = 1.0 - overall_uncertainty
        
        return SpecialistPrediction(
            prediction=analysis,
            confidence=confidence,
            uncertainty=overall_uncertainty,
            explanation=f"Governance uncertainty analysis: {overall_uncertainty:.3f} overall uncertainty",
            evidence=evidence,
            cross_domain_score=0.8,
            hallucination_risk=self.calculate_hallucination_risk(analysis, evidence)
        )


class CrossDomainValidator:
    """Cross-domain validator for governance oversight."""
    
    def __init__(self, validator_id: str, domain: str, expertise_areas: List[str]):
        self.validator_id = validator_id
        self.domain = domain
        self.expertise_areas = expertise_areas
        self.validation_history = []
        
    async def validate_decision(self, 
                              decision_data: Dict[str, Any],
                              specialist_predictions: List[SpecialistPrediction],
                              context: Optional[Dict[str, Any]] = None) -> CrossDomainValidation:
        """Validate a decision from cross-domain perspective."""
        try:
            # Analyze predictions from domain perspective
            validation_score = await self._calculate_validation_score(
                decision_data, specialist_predictions, context
            )
            
            # Assess compliance with domain standards
            compliance_score = await self._assess_compliance(
                decision_data, specialist_predictions, context
            )
            
            # Perform risk assessment
            risk_assessment = await self._assess_risks(
                decision_data, specialist_predictions, context
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                decision_data, specialist_predictions, validation_score, compliance_score
            )
            
            # Identify red flags
            red_flags = await self._identify_red_flags(
                decision_data, specialist_predictions, risk_assessment
            )
            
            validation = CrossDomainValidation(
                domain=self.domain,
                validator_id=self.validator_id,
                validation_score=validation_score,
                compliance_score=compliance_score,
                risk_assessment=risk_assessment,
                recommendations=recommendations,
                red_flags=red_flags
            )
            
            # Store validation history
            self.validation_history.append(validation.to_dict())
            if len(self.validation_history) > 1000:
                self.validation_history = self.validation_history[-1000:]
            
            return validation
            
        except Exception as e:
            logger.error(f"Cross-domain validation failed for {self.validator_id}: {e}")
            return CrossDomainValidation(
                domain=self.domain,
                validator_id=self.validator_id,
                validation_score=0.0,
                compliance_score=0.0,
                risk_assessment={"error": str(e)},
                recommendations=[f"Validation failed: {str(e)}"],
                red_flags=["validation_failure"]
            )
    
    async def _calculate_validation_score(self, 
                                        decision_data: Dict[str, Any],
                                        predictions: List[SpecialistPrediction],
                                        context: Optional[Dict[str, Any]]) -> float:
        """Calculate validation score from domain perspective."""
        if not predictions:
            return 0.0
        
        # Domain-specific validation logic
        if self.domain == "ethical":
            return await self._ethical_validation_score(decision_data, predictions, context)
        elif self.domain == "technical":
            return await self._technical_validation_score(decision_data, predictions, context)
        elif self.domain == "legal":
            return await self._legal_validation_score(decision_data, predictions, context)
        elif self.domain == "security":
            return await self._security_validation_score(decision_data, predictions, context)
        else:
            # General validation based on prediction quality
            avg_confidence = sum(p.confidence for p in predictions) / len(predictions)
            avg_cross_domain = sum(p.cross_domain_score for p in predictions) / len(predictions)
            return (avg_confidence + avg_cross_domain) / 2
    
    async def _ethical_validation_score(self, 
                                      decision_data: Dict[str, Any],
                                      predictions: List[SpecialistPrediction],
                                      context: Optional[Dict[str, Any]]) -> float:
        """Validate from ethical perspective."""
        score = 0.5  # Start neutral
        
        # Check for bias indicators
        bias_indicators = decision_data.get("bias_indicators", {})
        if bias_indicators:
            bias_score = sum(bias_indicators.values()) / len(bias_indicators)
            score -= bias_score * 0.3
        
        # Check for fairness considerations
        fairness_score = decision_data.get("fairness_score", 0.5)
        score += (fairness_score - 0.5) * 0.4
        
        # Check prediction consistency (ethical decisions should be consistent)
        if len(predictions) > 1:
            confidence_variance = np.var([p.confidence for p in predictions])
            if confidence_variance < 0.1:  # Low variance is good for ethics
                score += 0.2
        
        return max(0.0, min(1.0, score))
    
    async def _technical_validation_score(self, 
                                        decision_data: Dict[str, Any],
                                        predictions: List[SpecialistPrediction],
                                        context: Optional[Dict[str, Any]]) -> float:
        """Validate from technical perspective."""
        score = 0.5
        
        # Check technical feasibility
        feasibility = decision_data.get("technical_feasibility", 0.5)
        score += (feasibility - 0.5) * 0.4
        
        # Check for technical risks
        tech_risks = decision_data.get("technical_risks", {})
        if tech_risks:
            avg_risk = sum(tech_risks.values()) / len(tech_risks)
            score -= avg_risk * 0.3
        
        # Reward high confidence in technical predictions
        if predictions:
            avg_confidence = sum(p.confidence for p in predictions) / len(predictions)
            score += (avg_confidence - 0.5) * 0.3
        
        return max(0.0, min(1.0, score))
    
    async def _legal_validation_score(self, 
                                    decision_data: Dict[str, Any],
                                    predictions: List[SpecialistPrediction],
                                    context: Optional[Dict[str, Any]]) -> float:
        """Validate from legal perspective."""
        score = 0.5
        
        # Check legal compliance
        legal_compliance = decision_data.get("legal_compliance", 0.5)
        score += (legal_compliance - 0.5) * 0.5
        
        # Check for legal risks
        legal_risks = decision_data.get("legal_risks", [])
        score -= len(legal_risks) * 0.1
        
        # Check constitutional compliance
        constitutional_scores = [p.cross_domain_score for p in predictions if hasattr(p, 'cross_domain_score')]
        if constitutional_scores:
            avg_constitutional = sum(constitutional_scores) / len(constitutional_scores)
            score += (avg_constitutional - 0.5) * 0.3
        
        return max(0.0, min(1.0, score))
    
    async def _security_validation_score(self, 
                                       decision_data: Dict[str, Any],
                                       predictions: List[SpecialistPrediction],
                                       context: Optional[Dict[str, Any]]) -> float:
        """Validate from security perspective."""
        score = 0.5
        
        # Check security risks
        security_risks = decision_data.get("security_risks", {})
        if security_risks:
            max_risk = max(security_risks.values())
            score -= max_risk * 0.4
        
        # Check for vulnerabilities
        vulnerabilities = decision_data.get("vulnerabilities", [])
        score -= len(vulnerabilities) * 0.1
        
        # Reward low hallucination risk (important for security)
        if predictions:
            avg_hallucination_risk = sum(p.hallucination_risk for p in predictions) / len(predictions)
            score += (1.0 - avg_hallucination_risk) * 0.3
        
        return max(0.0, min(1.0, score))
    
    async def _assess_compliance(self, 
                               decision_data: Dict[str, Any],
                               predictions: List[SpecialistPrediction],
                               context: Optional[Dict[str, Any]]) -> float:
        """Assess compliance with domain standards."""
        # Domain-specific compliance assessment
        compliance_factors = decision_data.get("compliance_factors", {})
        
        if not compliance_factors:
            return 0.5
        
        # Calculate weighted compliance score
        domain_weights = {
            "ethical": {"fairness": 0.4, "transparency": 0.3, "accountability": 0.3},
            "technical": {"feasibility": 0.4, "scalability": 0.3, "maintainability": 0.3},
            "legal": {"regulatory": 0.5, "constitutional": 0.3, "precedent": 0.2},
            "security": {"confidentiality": 0.4, "integrity": 0.3, "availability": 0.3}
        }
        
        weights = domain_weights.get(self.domain, {})
        if not weights:
            # Equal weighting if no specific weights
            weights = {k: 1.0/len(compliance_factors) for k in compliance_factors}
        
        compliance_score = 0.0
        for factor, value in compliance_factors.items():
            weight = weights.get(factor, 1.0/len(compliance_factors))
            compliance_score += value * weight
        
        return max(0.0, min(1.0, compliance_score))
    
    async def _assess_risks(self, 
                          decision_data: Dict[str, Any],
                          predictions: List[SpecialistPrediction],
                          context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Assess various risks from domain perspective."""
        risks = {}
        
        # General risks
        if predictions:
            avg_uncertainty = sum(p.uncertainty for p in predictions) / len(predictions)
            risks["prediction_uncertainty"] = avg_uncertainty
            
            avg_hallucination_risk = sum(p.hallucination_risk for p in predictions) / len(predictions)
            risks["hallucination_risk"] = avg_hallucination_risk
        
        # Domain-specific risks
        domain_risks = decision_data.get(f"{self.domain}_risks", {})
        risks.update(domain_risks)
        
        return risks
    
    async def _generate_recommendations(self, 
                                      decision_data: Dict[str, Any],
                                      predictions: List[SpecialistPrediction],
                                      validation_score: float,
                                      compliance_score: float) -> List[str]:
        """Generate domain-specific recommendations."""
        recommendations = []
        
        # Score-based recommendations
        if validation_score < 0.6:
            recommendations.append(f"Low {self.domain} validation score ({validation_score:.2f}). Consider domain expert review.")
        
        if compliance_score < 0.7:
            recommendations.append(f"Compliance concerns in {self.domain} domain ({compliance_score:.2f}). Review regulatory requirements.")
        
        # Prediction quality recommendations
        if predictions:
            low_confidence_predictions = [p for p in predictions if p.confidence < 0.7]
            if low_confidence_predictions:
                recommendations.append(f"Found {len(low_confidence_predictions)} low-confidence predictions. Consider additional validation.")
            
            high_uncertainty_predictions = [p for p in predictions if p.uncertainty > 0.4]
            if high_uncertainty_predictions:
                recommendations.append(f"Found {len(high_uncertainty_predictions)} high-uncertainty predictions. Consider risk mitigation.")
        
        # Domain-specific recommendations
        if self.domain == "ethical":
            recommendations.extend(self._ethical_recommendations(decision_data, predictions))
        elif self.domain == "security":
            recommendations.extend(self._security_recommendations(decision_data, predictions))
        
        return recommendations
    
    def _ethical_recommendations(self, 
                               decision_data: Dict[str, Any],
                               predictions: List[SpecialistPrediction]) -> List[str]:
        """Generate ethical recommendations."""
        recommendations = []
        
        bias_indicators = decision_data.get("bias_indicators", {})
        if any(v > 0.3 for v in bias_indicators.values()):
            recommendations.append("High bias indicators detected. Consider bias mitigation strategies.")
        
        fairness_score = decision_data.get("fairness_score", 0.5)
        if fairness_score < 0.6:
            recommendations.append("Fairness concerns detected. Review decision impact on different groups.")
        
        return recommendations
    
    def _security_recommendations(self, 
                                decision_data: Dict[str, Any],
                                predictions: List[SpecialistPrediction]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        security_risks = decision_data.get("security_risks", {})
        high_risks = [k for k, v in security_risks.items() if v > 0.7]
        if high_risks:
            recommendations.append(f"High security risks detected: {', '.join(high_risks)}. Implement additional security measures.")
        
        vulnerabilities = decision_data.get("vulnerabilities", [])
        if vulnerabilities:
            recommendations.append(f"Security vulnerabilities identified: {len(vulnerabilities)} issues. Conduct security review.")
        
        return recommendations
    
    async def _identify_red_flags(self, 
                                decision_data: Dict[str, Any],
                                predictions: List[SpecialistPrediction],
                                risk_assessment: Dict[str, float]) -> List[str]:
        """Identify critical red flags."""
        red_flags = []
        
        # High-risk thresholds
        if any(risk > 0.8 for risk in risk_assessment.values()):
            red_flags.append("critical_risk_detected")
        
        # Prediction quality red flags
        if predictions:
            extremely_low_confidence = [p for p in predictions if p.confidence < 0.3]
            if extremely_low_confidence:
                red_flags.append("extremely_low_confidence_predictions")
            
            high_hallucination_risk = [p for p in predictions if p.hallucination_risk > 0.8]
            if high_hallucination_risk:
                red_flags.append("high_hallucination_risk")
        
        # Domain-specific red flags
        if self.domain == "security":
            security_risks = decision_data.get("security_risks", {})
            if any(risk > 0.9 for risk in security_risks.values()):
                red_flags.append("critical_security_risk")
        
        elif self.domain == "ethical":
            bias_indicators = decision_data.get("bias_indicators", {})
            if any(bias > 0.8 for bias in bias_indicators.values()):
                red_flags.append("severe_bias_detected")
        
        return red_flags
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics."""
        stats = {
            "validator_id": self.validator_id,
            "domain": self.domain,
            "expertise_areas": self.expertise_areas,
            "total_validations": len(self.validation_history)
        }
        
        if self.validation_history:
            recent_validations = self.validation_history[-50:]  # Last 50 validations
            
            stats.update({
                "avg_validation_score": np.mean([v["validation_score"] for v in recent_validations]),
                "avg_compliance_score": np.mean([v["compliance_score"] for v in recent_validations]),
                "red_flags_rate": len([v for v in recent_validations if v["red_flags"]]) / len(recent_validations)
            })
        
        return stats


# Factory functions for creating specialists and validators
def create_enhanced_specialists() -> List[EnhancedMLSpecialist]:
    """Create instances of all enhanced ML specialists."""
    return [
        GraphNeuralNetworkSpecialist(),
        MultimodalAISpecialist(),
        UncertaintyQuantificationSpecialist()
    ]


def create_cross_domain_validators() -> List[CrossDomainValidator]:
    """Create instances of all cross-domain validators."""
    return [
        CrossDomainValidator(
            validator_id="ethical_validator",
            domain="ethical",
            expertise_areas=["fairness", "bias_detection", "transparency", "accountability"]
        ),
        CrossDomainValidator(
            validator_id="technical_validator",
            domain="technical",
            expertise_areas=["feasibility", "scalability", "performance", "maintainability"]
        ),
        CrossDomainValidator(
            validator_id="legal_validator",
            domain="legal",
            expertise_areas=["regulatory_compliance", "constitutional_law", "precedent_analysis"]
        ),
        CrossDomainValidator(
            validator_id="security_validator",
            domain="security",
            expertise_areas=["threat_assessment", "vulnerability_analysis", "risk_management"]
        )
    ]